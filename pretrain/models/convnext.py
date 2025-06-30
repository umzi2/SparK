# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This file is basically a copy of: https://github.com/facebookresearch/ConvNeXt/blob/06f7b05f922e21914916406141f50f82b4a15852/models/convnext.py
from typing import List

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model

# from pretrain.encoder import SparseConvNeXtLayerNorm, SparseConvNeXtBlock


from encoder import SparseConvNeXtBlock, SparseBatchNorm2d

class MetaNeXtStage(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            ds_stride=2,
            depth=2,
            drop_path_rates=None,
            mlp = 4
    ):
        super().__init__()
        self.grad_checkpointing = False
        if ds_stride > 1:
            self.downsample = nn.Sequential(
                SparseBatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, kernel_size=ds_stride, stride=ds_stride),
            )
        else:
            self.downsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(SparseConvNeXtBlock(
                dim=out_chs,
                drop_path=drop_path_rates[i],
                mlp=mlp,
            ))
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        x = self.downsample(x)
        return self.blocks(x)
class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    
    def __init__(self, in_chans=3, num_classes=0,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6,mlp =(4, 4, 4, 3), head_init_scale=1., global_pool='avg',
                 sparse=True,
                 ):
        super().__init__()
        self.dims: List[int] = dims
        # self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            SparseBatchNorm2d(dims[0])
        )
        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        cur = 0
        prev_chs = dims[0]
        for i in range(4):
            out_chs = dims[i]
            stage = MetaNeXtStage(in_chs=prev_chs,out_chs=out_chs,depth=depths[i], ds_stride = 1 if i ==0 else 2, drop_path_rates=dp_rates[i],mlp = mlp[i])
            self.stages.append(stage)
            cur += depths[i]
            prev_chs = out_chs
        self.depths = depths
        
        self.apply(self._init_weights)
        if num_classes > 0:
            self.norm = SparseBatchNorm2d(dims[-1])  # final norm layer for LE/FT; should not be sparse
            self.fc = nn.Linear(dims[-1], num_classes)
        else:
            self.norm = nn.Identity()
            self.fc = nn.Identity()
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
    
    def get_downsample_ratio(self) -> int:
        return 32
    
    def get_feature_map_channels(self) -> List[int]:
        return self.dims
    
    def forward(self, x, hierarchical=False):
        # if hierarchical:
        ls = []
        x = self.stem(x)
        for i in range(4):
            x = self.stages[i](x)
            ls.append(x)
        return ls
        # else:
        #     return self.fc(self.norm(x.mean([-2, -1]))) # (B, C, H, W) =mean=> (B, C) =norm&fc=> (B, NumCls)
    
    def get_classifier(self):
        return self.fc
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate}, layer_scale_init_value={self.layer_scale_init_value:g}'


@register_model
def convnext_tiny(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    state = torch.load("/home/umzi/PycharmProjects/SparK_fork/pretrain/your_exp_dir/convnext_tiny_1kpretrained_timm_style.pth")
    print(model.load_state_dict(state,strict=False))
    return model


@register_model
def convnext_small(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    return model


@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model


@register_model
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

