from torchvision.transforms import transforms

import torch
import torch.nn as nn

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
        trans = [
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(resize_size, antialias=False),
        ]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        trans.extend([
            transforms.Normalize(mean=mean, std=std), 
            transforms.CenterCrop(crop_size), 
            ConvertBCHWtoCBHW()
        ])
        self.transforms = transforms.Compose(trans)

    def __call__(self, x):
        return self.transforms(x)
