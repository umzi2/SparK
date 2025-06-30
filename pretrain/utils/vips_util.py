import random


import numpy as np
import pyvips
import torch
from chainner_ext import resize,ResizeFilter
from torch import Tensor

RESIZEFILTERS = [
    ResizeFilter.CubicCatrom,
    ResizeFilter.Gauss,
    ResizeFilter.Linear,
    ResizeFilter.Box,
    ResizeFilter.CubicBSpline,
    ResizeFilter.CubicMitchell,
    ResizeFilter.Lagrange,
    ResizeFilter.Lanczos,
    ResizeFilter.Nearest
]

def img2rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert a NumPy array image to RGB.

    Parameters:
        image (np.ndarray): The input image array.
                            Expected shape: (H, W), (H, W, 1), (H, W, 3), or (H, W, 4+).
                            The array should have dtype=np.uint8 or similar.

    Returns:
        np.ndarray: RGB image with shape (H, W, 3).
    """

    # preprocessing for grayscale case
    if image.ndim == 3 and image.shape[2] == 1:
        image = image.squeeze(-1)

    # rgb
    if image.ndim == 3 and image.shape[2] == 3:
        return image

    # rgba
    elif image.ndim == 3 and image.shape[2] > 3:
        return image[:, :, :3]

    # grayscale
    elif image.ndim == 2:
        return np.stack((image,) * 3, axis=-1)

    else:
        raise ValueError(
            "Unsupported image shape: expected (H, W), (H, W, 1), (H, W, 3), or (H, W, 4+)"
        )
def vipsimfrompath(path: str) -> pyvips.Image:
    img = pyvips.Image.new_from_file(
        path, access="sequential", fail=True
    ).icc_transform("srgb")  # pyright: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]
    assert isinstance(img, pyvips.Image)
    return img
def single_random_crop_vips(path: str, patch_size: int) -> Tensor:
    img = vipsimfrompath(path)
    h, w = img.height, img.width
    # Вычисляем максимальный возможный размер тайла
    max_tile = min(h, w)
    if max_tile < patch_size:
        raise ValueError(f"Изображение слишком маленькое: min(height, width)={max_tile} < patch_size={patch_size}")
    tile_size = random.randint(patch_size, max_tile)
    y = random.randint(0, h - tile_size)
    x = random.randint(0, w - tile_size)
    region_gt = pyvips.Region.new(img)
    data_gt = region_gt.fetch(x, y, tile_size, tile_size)
    data_gt = img2rgb(
        np.ndarray(
            buffer=data_gt,
            dtype=np.uint8,
            shape=[tile_size, tile_size, img.bands],  # pyright: ignore
        )).astype(np.float32)/255.0
    return torch.tensor(resize(data_gt,(patch_size,patch_size),random.choice(RESIZEFILTERS),False).transpose(2,0,1))
if __name__ == "__main__":
    img = single_random_crop_vips("/run/media/umzi/H/nahuy_pixiv/new/ma_tile/0_0.png",224)
    print(img)