# https://github.com/pytorch/vision/blob/main/references/video_classification/presets.py

import torch
import torch.nn as nn
from torchvision.transforms import transforms

class ConvertBCHWtoCBHW(nn.Module):
    """Convert tensor from (B, C, H, W) to (C, B, H, W)"""

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(1, 0, 2, 3)

class VideoClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size,
        resize_size,
        mean=(0.43216, 0.394666, 0.37645),
        std=(0.22803, 0.22145, 0.216989),
        hflip_prob=0.5,
    ):
        self.transforms = transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float32),
                transforms.Resize(resize_size, antialias=False),
                transforms.RandomHorizontalFlip(hflip_prob),
                transforms.Normalize(mean=mean, std=std),
                transforms.RandomCrop(crop_size),
                ConvertBCHWtoCBHW(),
            ]
        )

    def __call__(self, x):
        return self.transforms(x)


class VideoClassificationPresetEval:
    def __init__(self, *, crop_size, resize_size, mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)):
        self.transforms = transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float32),
                transforms.Resize(resize_size, antialias=False),
                transforms.Normalize(mean=mean, std=std),
                transforms.CenterCrop(crop_size),
                ConvertBCHWtoCBHW(),
            ]
        )

    def __call__(self, x):
        return self.transforms(x)