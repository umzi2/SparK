# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from timm.models.layers import DropPath


_cur_active: torch.Tensor = None            # B1ff
# todo: try to use `gather` for speed?
def _get_active_ex_or_ii(H, W, returning_active_ex=True):
    h_repeat, w_repeat = H // _cur_active.shape[-2], W // _cur_active.shape[-1]
    active_ex = _cur_active.repeat_interleave(h_repeat, dim=2).repeat_interleave(w_repeat, dim=3)
    return active_ex if returning_active_ex else active_ex.squeeze(1).nonzero(as_tuple=True)  # ii: bi, hi, wi


def sp_conv_forward(self, x: torch.Tensor):
    x = super(type(self), self).forward(x)
    x *= _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], returning_active_ex=True)    # (BCHW) *= (B1HW), mask the output of conv
    return x


def sp_bn_forward(self, x: torch.Tensor):
    ii = _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], returning_active_ex=False)
    
    bhwc = x.permute(0, 2, 3, 1)
    nc = bhwc[ii]                               # select the features on non-masked positions to form a flatten feature `nc`
    nc = super(type(self), self).forward(nc)    # use BN1d to normalize this flatten feature `nc`
    
    bchw = torch.zeros_like(bhwc)
    bchw[ii] = nc
    bchw = bchw.permute(0, 3, 1, 2)
    return bchw


class SparseConv2d(nn.Conv2d):
    forward = sp_conv_forward   # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseMaxPooling(nn.MaxPool2d):
    forward = sp_conv_forward   # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseAvgPooling(nn.AvgPool2d):
    forward = sp_conv_forward   # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseBatchNorm2d(nn.BatchNorm1d):
    forward = sp_bn_forward     # hack: override the forward function; see `sp_bn_forward` above for more details


class SparseSyncBatchNorm2d(nn.SyncBatchNorm):
    forward = sp_bn_forward     # hack: override the forward function; see `sp_bn_forward` above for more details


class SparseConvNeXtLayerNorm(nn.LayerNorm):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last", sparse=True):
        if data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        super().__init__(normalized_shape, eps, elementwise_affine=True)
        self.data_format = data_format
        self.sparse = sparse
    
    def forward(self, x):
        if x.ndim == 4: # BHWC or BCHW
            if self.data_format == "channels_last": # BHWC
                if self.sparse:
                    ii = _get_active_ex_or_ii(H=x.shape[1], W=x.shape[2], returning_active_ex=False)
                    nc = x[ii]
                    nc = super(SparseConvNeXtLayerNorm, self).forward(nc)
    
                    x = torch.zeros_like(x)
                    x[ii] = nc.to(x.dtype)
                    return x
                else:
                    return super(SparseConvNeXtLayerNorm, self).forward(x)
            else:       # channels_first, BCHW
                if self.sparse:
                    ii = _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], returning_active_ex=False)
                    bhwc = x.permute(0, 2, 3, 1)
                    nc = bhwc[ii]
                    nc = super(SparseConvNeXtLayerNorm, self).forward(nc)
                
                    x = torch.zeros_like(bhwc)
                    x[ii] = nc.to(x.dtype)
                    return x.permute(0, 3, 1, 2)
                else:
                    u = x.mean(1, keepdim=True)
                    s = (x - u).pow(2).mean(1, keepdim=True)
                    x = (x - u) / torch.sqrt(s + self.eps)
                    x = self.weight[:, None, None] * x + self.bias[:, None, None]
                    return x
        else:           # BLC or BC
            if self.sparse:
                raise NotImplementedError
            else:
                return super(SparseConvNeXtLayerNorm, self).forward(x)

    def __repr__(self):
        return super(SparseConvNeXtLayerNorm, self).__repr__()[:-1] + f', ch={self.data_format.split("_")[-1]}, sp={self.sparse})'

class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution
    """

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )

class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    copied from timm: https://github.com/huggingface/pytorch-image-models/blob/v0.6.11/timm/models/layers/mlp.py
    """
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
class SparseConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, sparse=True, ks=7,mlp=4):
        super().__init__()
        self.token_mixer = InceptionDWConv2d(dim)  # depthwise conv
        self.norm = SparseBatchNorm2d(dim)
        # self.pwconv1 = nn.Conv2d(dim, 4 * dim,1)  # pointwise/1x1 convs, implemented with linear layers
        # self.act = nn.GELU()
        # self.pwconv2 = nn.Conv2d(4 * dim, dim,1)
        self.mlp = ConvMlp(dim, dim*mlp)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path: nn.Module = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.sparse = sparse
    
    def forward(self, x):
        input = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = self.gamma[None,...,None,None] * x
        
        if self.sparse:
            x *= _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], returning_active_ex=True)
        
        x = input + self.drop_path(x)
        return x
    
    def __repr__(self):
        return super(SparseConvNeXtBlock, self).__repr__()[:-1] + f', sp={self.sparse})'


class SparseEncoder(nn.Module):
    def __init__(self, cnn, input_size, sbn=False, verbose=False):
        super(SparseEncoder, self).__init__()
        self.sp_cnn = SparseEncoder.dense_model_to_sparse(m=cnn, verbose=verbose, sbn=sbn)
        self.input_size, self.downsample_raito, self.enc_feat_map_chs = input_size, cnn.get_downsample_ratio(), cnn.get_feature_map_channels()
    
    @staticmethod
    def dense_model_to_sparse(m: nn.Module, verbose=False, sbn=False):
        oup = m
        if isinstance(m, nn.Conv2d):
            m: nn.Conv2d
            bias = m.bias is not None
            oup = SparseConv2d(
                m.in_channels, m.out_channels,
                kernel_size=m.kernel_size, stride=m.stride, padding=m.padding,
                dilation=m.dilation, groups=m.groups, bias=bias, padding_mode=m.padding_mode,
            )
            oup.weight.data.copy_(m.weight.data)
            if bias:
                oup.bias.data.copy_(m.bias.data)
        elif isinstance(m, nn.MaxPool2d):
            m: nn.MaxPool2d
            oup = SparseMaxPooling(m.kernel_size, stride=m.stride, padding=m.padding, dilation=m.dilation, return_indices=m.return_indices, ceil_mode=m.ceil_mode)
        elif isinstance(m, nn.AvgPool2d):
            m: nn.AvgPool2d
            oup = SparseAvgPooling(m.kernel_size, m.stride, m.padding, ceil_mode=m.ceil_mode, count_include_pad=m.count_include_pad, divisor_override=m.divisor_override)
        elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m: nn.BatchNorm2d
            oup = (SparseSyncBatchNorm2d if sbn else SparseBatchNorm2d)(m.weight.shape[0], eps=m.eps, momentum=m.momentum, affine=m.affine, track_running_stats=m.track_running_stats)
            oup.weight.data.copy_(m.weight.data)
            oup.bias.data.copy_(m.bias.data)
            oup.running_mean.data.copy_(m.running_mean.data)
            oup.running_var.data.copy_(m.running_var.data)
            oup.num_batches_tracked.data.copy_(m.num_batches_tracked.data)
            if hasattr(m, "qconfig"):
                oup.qconfig = m.qconfig
        elif isinstance(m, nn.LayerNorm) and not isinstance(m, SparseConvNeXtLayerNorm):
            m: nn.LayerNorm
            oup = SparseConvNeXtLayerNorm(m.weight.shape[0], eps=m.eps)
            oup.weight.data.copy_(m.weight.data)
            oup.bias.data.copy_(m.bias.data)
        elif isinstance(m, (nn.Conv1d,)):
            raise NotImplementedError
        
        for name, child in m.named_children():
            oup.add_module(name, SparseEncoder.dense_model_to_sparse(child, verbose=verbose, sbn=sbn))
        del m
        return oup
    
    def forward(self, x):
        return self.sp_cnn(x, hierarchical=True)
