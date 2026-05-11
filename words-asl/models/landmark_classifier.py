"""
landmark_classifier.py — Track 1 model.

A small Transformer encoder over per-frame landmark embeddings. Input is the
LandmarkDataset output: [B, T, K, 3] float32 + an optional [B, T, K] bool
valid mask. Output is [B, num_classes] logits.

Why a per-frame Transformer rather than ST-GCN or BiLSTM:

  - Every op in this model is in the green rows of docs/coreml-compat.md
    (linear, matmul, softmax, layer_norm, gelu, reshape, transpose,
    reduce_mean). We expect the converted .mlpackage to land entirely on
    ANE. ST-GCN's adjacency-gather and BiLSTM ops are mixed/CPU-fallback
    on iOS 16+.
  - The pre-norm variant (norm_first=True) converts more reliably through
    coremltools 8.x than the default post-norm.
  - With d_model=128, 2 layers, 4 heads, FFN 256, the param count is
    ~150 k — small enough for the WLASL-100 train set (409 clips) to
    learn without massive overfit, big enough to actually fit.

Decisions deliberately deferred (see docs/decisions.md):
  - Architecture: ST-GCN and BiLSTM are listed as fallbacks if this baseline
    overfits or underperforms.
  - Padding-mask handling: at inference time the iOS path always feeds
    fully-detected frames or zero-fills (see LandmarkModel.swift), so the
    Core ML conversion is traced *without* a valid_mask path. The training
    forward still uses the mask so we don't pollute the loss with padded
    landmarks.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from utils.registry import register_model


@register_model("landmark_classifier")
class LandmarkClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_keypoints: int = 75,
        coord_dims: int = 3,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_frames: int = 64,
        pool: str = "mean",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.coord_dims = coord_dims
        self.d_model = d_model
        self.pool = pool

        # Per-frame embedding: flatten K*3 into a single token vector then
        # project. A 2-layer MLP doesn't help here (the input is already
        # low-dimensional) and would add CPU-fallback ops on the iOS side.
        self.embed = nn.Linear(num_keypoints * coord_dims, d_model)

        # Parameter tensor + broadcast add. nn.Embedding would emit a
        # `gather` op on conversion which we want to avoid.
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, T, K, 3]
        # valid_mask: [B, T, K] bool, optional
        B, T, K, C = x.shape
        if K != self.num_keypoints or C != self.coord_dims:
            raise ValueError(
                f"input shape mismatch: got [B={B},T={T},K={K},C={C}], "
                f"expected K={self.num_keypoints} C={self.coord_dims}"
            )
        if T > self.pos_embed.shape[1]:
            raise ValueError(
                f"sequence length {T} exceeds max_frames={self.pos_embed.shape[1]}; "
                f"increase max_frames in the config or reduce dataset.num_frames."
            )

        tokens = x.reshape(B, T, K * C)
        tokens = self.embed(tokens) + self.pos_embed[:, :T, :]

        # Frame-level padding mask: a frame is "padded" only if literally
        # zero keypoints were detected. The dataset's normaliser already
        # handles low-detection frames; this catches the case where MediaPipe
        # missed everything and the sampler reused a duplicate index.
        key_padding_mask: torch.Tensor | None = None
        if valid_mask is not None:
            frame_valid = valid_mask.any(dim=-1)            # [B, T]
            key_padding_mask = ~frame_valid                 # True = ignore
            # Guard against an entirely-padded sequence: replace with the
            # all-False mask (no positions ignored) so MultiheadAttention
            # doesn't NaN. This shouldn't happen in practice — the dataset
            # already filters caches with too few detections — but the cost
            # of checking is negligible.
            all_padded = key_padding_mask.all(dim=-1)
            if all_padded.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_padded] = False
        else:
            frame_valid = None

        h = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)

        if self.pool == "mean":
            if frame_valid is not None:
                w = frame_valid.float().unsqueeze(-1)       # [B, T, 1]
                pooled = (h * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
            else:
                pooled = h.mean(dim=1)
        elif self.pool == "first":
            pooled = h[:, 0]
        else:
            raise ValueError(f"unknown pool={self.pool!r}")

        return self.head(pooled)
