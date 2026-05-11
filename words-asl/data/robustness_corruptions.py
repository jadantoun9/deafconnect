"""
robustness_corruptions.py — synthetic corruptions for the robustness eval.

Phase E reports per-track accuracy under realistic deployment failure modes.
Rather than collect a real-world distorted test set (out of scope for a
thesis), we apply controlled synthetic corruptions to the clean WLASL test
set:

  - low_light:  multiplicative darkening + Gaussian noise
  - blur:       Gaussian blur with configurable kernel
  - occlusion:  random rectangular black-out (simulates a hand half off-screen)
  - jpeg:       compress/decompress at low quality (simulates phone-encoded
                video on a slow upload)

Each corruption is parameterised by a single 'severity' integer (1..5) that
plugs into ImageNet-C's published intensity scales. Severity 0 is the
no-op baseline.

All corruptions are applied *temporally consistently*: the same parameters
across all T frames. Per-frame randomness would be unrealistic for low-light
(camera doesn't suddenly brighten between frames) and unfair for occlusion
(would create a flickering box).

Each function takes and returns [T, 3, H, W] float tensors in [0, 1].
"""

from __future__ import annotations

import io
import random
from typing import Callable

import torch
import torch.nn.functional as F


Corruption = Callable[[torch.Tensor], torch.Tensor]


# Severity tables (1..5). Tuned to feel comparable to ImageNet-C; owner can
# swap any of these without touching the call sites.
_LOW_LIGHT_DARKEN = {1: 0.7, 2: 0.5, 3: 0.35, 4: 0.22, 5: 0.12}
_LOW_LIGHT_NOISE_STD = {1: 0.02, 2: 0.04, 3: 0.06, 4: 0.09, 5: 0.13}
_BLUR_KERNEL = {1: 3, 2: 5, 3: 7, 4: 11, 5: 15}
_BLUR_SIGMA = {1: 0.6, 2: 1.0, 3: 1.6, 4: 2.4, 5: 3.4}
_OCCL_FRAC = {1: 0.10, 2: 0.18, 3: 0.27, 4: 0.36, 5: 0.45}
_JPEG_Q = {1: 60, 2: 40, 3: 25, 4: 15, 5: 8}


def low_light(severity: int = 3, seed: int | None = None) -> Corruption:
    rng = random.Random(seed)

    def apply(x: torch.Tensor) -> torch.Tensor:
        if severity == 0:
            return x
        darken = _LOW_LIGHT_DARKEN[severity]
        sigma = _LOW_LIGHT_NOISE_STD[severity]
        x = x * darken
        noise = torch.randn_like(x, generator=None) * sigma
        return (x + noise).clamp(0.0, 1.0)

    return apply


def blur(severity: int = 3) -> Corruption:
    def apply(x: torch.Tensor) -> torch.Tensor:
        if severity == 0:
            return x
        k = _BLUR_KERNEL[severity]
        sigma = _BLUR_SIGMA[severity]
        kernel = _gaussian_kernel(k, sigma).to(x)
        # Per-channel separable conv.
        T, C, H, W = x.shape
        x_flat = x.view(T * C, 1, H, W)
        x_flat = F.pad(x_flat, [k // 2] * 4, mode="reflect")
        x_flat = F.conv2d(x_flat, kernel.view(1, 1, k, 1))
        x_flat = F.conv2d(x_flat, kernel.view(1, 1, 1, k))
        return x_flat.view(T, C, H, W).clamp(0.0, 1.0)

    return apply


def _gaussian_kernel(k: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(k, dtype=torch.float32) - (k - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def occlusion(severity: int = 3, seed: int | None = None) -> Corruption:
    rng = random.Random(seed)

    def apply(x: torch.Tensor) -> torch.Tensor:
        if severity == 0:
            return x
        frac = _OCCL_FRAC[severity]
        T, C, H, W = x.shape
        bh = max(1, int(H * frac ** 0.5))
        bw = max(1, int(W * frac ** 0.5))
        top = rng.randint(0, max(0, H - bh))
        left = rng.randint(0, max(0, W - bw))
        out = x.clone()
        out[..., top:top + bh, left:left + bw] = 0.0
        return out

    return apply


def jpeg(severity: int = 3) -> Corruption:
    """JPEG-compress every frame at quality `severity`'s level then decode."""
    def apply(x: torch.Tensor) -> torch.Tensor:
        if severity == 0:
            return x
        try:
            from PIL import Image
        except ImportError as e:
            raise SystemExit("Pillow is required for JPEG corruption.") from e
        q = _JPEG_Q[severity]
        T, C, H, W = x.shape
        out = torch.empty_like(x)
        for t in range(T):
            arr = (x[t].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
            img = Image.fromarray(arr, mode="RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            decoded = Image.open(buf).convert("RGB")
            arr2 = torch.from_numpy(_pil_to_array(decoded)).float() / 255.0
            out[t] = arr2.permute(2, 0, 1)
        return out

    return apply


def _pil_to_array(img):
    import numpy as np
    return np.asarray(img)


CORRUPTION_REGISTRY: dict[str, Callable[..., Corruption]] = {
    "low_light": low_light,
    "blur": blur,
    "occlusion": occlusion,
    "jpeg": jpeg,
}


def build(name: str, severity: int) -> Corruption:
    if name not in CORRUPTION_REGISTRY:
        raise KeyError(f"unknown corruption {name!r}; options: {list(CORRUPTION_REGISTRY)}")
    return CORRUPTION_REGISTRY[name](severity=severity)
