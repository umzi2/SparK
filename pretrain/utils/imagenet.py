# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Callable, Optional, Tuple

import PIL.Image as PImage
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS
from torchvision.transforms import transforms
from torch.utils.data import Dataset

try:
    from torchvision.transforms import InterpolationMode
    interpolation = InterpolationMode.BICUBIC
except:
    import PIL
    interpolation = PIL.Image.BICUBIC


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f: img: PImage.Image = PImage.open(f).convert('RGB')
    return img


class FlatFolderDataset(Dataset):
    def __init__(self, image_folder: str, transform: Optional[Callable] = None):
        self.image_folder = os.path.abspath(image_folder)
        self.transform = transform
        self.extensions = IMG_EXTENSIONS
        self.samples = self._find_images(self.image_folder)
        self.loader = pil_loader

    def _find_images(self, folder):
        return [
            os.path.join(folder, fname)
            for fname in os.listdir(folder)
            if fname.lower().endswith(tuple(ext.lower() for ext in self.extensions))
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Any:
        path = self.samples[index]
        image = self.loader(path)
        if self.transform:
            image = self.transform(image)
        return image


def build_dataset_to_pretrain(dataset_path, input_size) -> Dataset:
    trans_train = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.67, 1.0), interpolation=interpolation),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])

    dataset_path = os.path.abspath(dataset_path)

    dataset_train = FlatFolderDataset(image_folder=dataset_path, transform=trans_train)
    print_transform(trans_train, '[pre-train]')
    return dataset_train


def print_transform(transform, s):
    print(f'Transform {s} = ')
    for t in transform.transforms:
        print(t)
    print('---------------------------\n')
