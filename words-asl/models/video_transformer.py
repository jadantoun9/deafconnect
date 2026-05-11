"""
video_transformer.py — frozen ViT-S/16 backbone for Track 2.

Loads a supervised ImageNet ViT-S/16 from timm and freezes it. The temporal
head + ToMe (in temporal_tome_model.py) are the only trainable parts — keeps
training tractable on 92 clips and a laptop GPU.

Returned tokens drop the CLS token; the per-patch tokens are what feed into
the temporal-aware ToMe stage. With image_size=224 and patch=16 we get
14x14 = 196 tokens per frame at D=384.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from utils.registry import register_model


class FrozenViTBackbone(nn.Module):
    """timm ViT-S/16 with all parameters frozen and BN/dropout in eval mode.

    Forward signature: x [B, 3, H, W] -> tokens [B, N, D].
    """

    def __init__(
        self,
        model_name: str = "vit_small_patch16_224.augreg_in21k_ft_in1k",
        image_size: int = 224,
    ):
        super().__init__()
        import timm
        self.vit = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            img_size=image_size,
        )
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()
        self.embed_dim: int = self.vit.embed_dim
        # Patch grid for downstream sanity-checks. timm returns patch_size as
        # a tuple (h, w); we assume square patches throughout.
        ps = self.vit.patch_embed.patch_size
        self.patch_size: int = ps[0] if isinstance(ps, (list, tuple)) else int(ps)
        self.num_patches: int = (image_size // self.patch_size) ** 2

    def train(self, mode: bool = True) -> "FrozenViTBackbone":
        # Override: keep the backbone in eval mode regardless of trainer state.
        # Otherwise dropout/BN inside the ViT would diverge from pretraining.
        super().train(mode)
        self.vit.eval()
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward_features returns [B, 1+N, D] (CLS + patches). We drop CLS;
        # ToMe operates on patch tokens only.
        feats = self.vit.forward_features(x)
        return feats[:, 1:, :].contiguous()


@register_model("vit_backbone")
def _build_vit_backbone(**kwargs):
    """Registry hook — exposed mostly for ad-hoc inspection. The full Track-2
    model is registered in temporal_tome_model.py."""
    return FrozenViTBackbone(**kwargs)
