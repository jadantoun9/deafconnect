"""
augmentations.py — temporally-consistent video augmentations.

Critical property: any random parameter (crop window, flip, jitter strength)
is sampled *once per clip* and applied to all T frames. Per-frame randomness
introduces a flicker that destroys video transformer training. The
torchvision augmentations operate per-image and would do exactly that —
hence this module instead.

All transforms here take and return [T, 3, H, W] float tensors in [0, 1].
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


Transform = Callable[[torch.Tensor], torch.Tensor]


def compose(*transforms: Transform) -> Transform:
    def apply(x: torch.Tensor) -> torch.Tensor:
        for t in transforms:
            x = t(x)
        return x
    return apply


def random_horizontal_flip(p: float = 0.5) -> Transform:
    """Flip the entire clip horizontally with probability p."""
    def apply(x: torch.Tensor) -> torch.Tensor:
        if random.random() < p:
            return torch.flip(x, dims=[-1])
        return x
    return apply


def random_resized_crop(
    out_size: int,
    scale: tuple[float, float] = (0.7, 1.0),
    ratio: tuple[float, float] = (0.9, 1.1),
) -> Transform:
    """Pick one crop window per clip, apply to all frames."""
    def apply(x: torch.Tensor) -> torch.Tensor:
        T, C, H, W = x.shape
        s = random.uniform(*scale)
        r = random.uniform(*ratio)
        target_h = int(round(H * (s / r) ** 0.5))
        target_w = int(round(W * (s * r) ** 0.5))
        target_h = max(8, min(target_h, H))
        target_w = max(8, min(target_w, W))
        top = random.randint(0, H - target_h)
        left = random.randint(0, W - target_w)
        x = x[:, :, top:top + target_h, left:left + target_w]
        x = F.interpolate(x, size=(out_size, out_size), mode="bilinear", align_corners=False)
        return x
    return apply


def center_crop(out_size: int) -> Transform:
    def apply(x: torch.Tensor) -> torch.Tensor:
        T, C, H, W = x.shape
        if H != out_size or W != out_size:
            x = F.interpolate(x, size=(out_size, out_size), mode="bilinear", align_corners=False)
        return x
    return apply


def color_jitter(
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
) -> Transform:
    """Per-clip jitter: same multipliers for every frame."""
    def apply(x: torch.Tensor) -> torch.Tensor:
        b = 1 + random.uniform(-brightness, brightness)
        c = 1 + random.uniform(-contrast, contrast)
        s = 1 + random.uniform(-saturation, saturation)

        x = x * b  # brightness

        # contrast: scale around the per-frame mean
        mean = x.mean(dim=(-1, -2, -3), keepdim=True)
        x = (x - mean) * c + mean

        # saturation: scale around per-frame luminance
        lum = (0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3])
        x = (x - lum) * s + lum

        return x.clamp(0.0, 1.0)
    return apply


def normalize_imagenet() -> Transform:
    """Normalise to ImageNet mean/std — required for backbones pretrained on ImageNet/Kinetics."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def apply(x: torch.Tensor) -> torch.Tensor:
        return (x - mean.to(x)) / std.to(x)
    return apply


@dataclass
class TrainAugmentationConfig:
    out_size: int = 224
    crop_scale: tuple[float, float] = (0.7, 1.0)
    crop_ratio: tuple[float, float] = (0.9, 1.1)
    flip_p: float = 0.5
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    imagenet_norm: bool = True


def build_train_pipeline(cfg: TrainAugmentationConfig) -> Transform:
    pipeline: list[Transform] = [
        random_resized_crop(cfg.out_size, scale=cfg.crop_scale, ratio=cfg.crop_ratio),
        random_horizontal_flip(p=cfg.flip_p),
        color_jitter(cfg.brightness, cfg.contrast, cfg.saturation),
    ]
    if cfg.imagenet_norm:
        pipeline.append(normalize_imagenet())
    return compose(*pipeline)


def build_eval_pipeline(out_size: int = 224, imagenet_norm: bool = True) -> Transform:
    pipeline: list[Transform] = [center_crop(out_size)]
    if imagenet_norm:
        pipeline.append(normalize_imagenet())
    return compose(*pipeline)
