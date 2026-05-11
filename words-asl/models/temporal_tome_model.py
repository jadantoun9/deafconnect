"""
temporal_tome_model.py — Track 2 main model.

End-to-end video transformer with three configurable token-merging modes:

    tome_mode = 'none'      Baseline. All T*N patch tokens go straight into the
                            temporal head. Most expensive, sets the accuracy
                            ceiling for this architecture.

    tome_mode = 'spatial'   Vanilla ToMe (Bolya et al. 2022). Bipartite soft
                            matching on cosine similarity of token features —
                            same image, time-agnostic. Kept as a published
                            baseline to compare the novelty against.

    tome_mode = 'temporal'  **Novel**: temporal-aware bipartite soft matching.
                            Cosine-similarity score multiplied by an
                            exponential temporal-distance kernel
                            exp(-|Δt| / τ), so patches at the same spatial
                            location at adjacent frames merge first
                            (background, low-motion regions) while patches
                            with high temporal divergence (the moving hands)
                            are preserved.

Architecture:

    [B, T, 3, H, W]
        │
        ▼
    Frozen ViT-S/16 (per-frame)            (/models/video_transformer.py)
        │  per frame: [B*T, N, D]
        ▼
    + temporal positional embedding        (along the T axis, broadcast over N)
        │
        ▼
    Flatten to [B, T*N, D]
        │
        ▼
    Token-merging block ('none'/'spatial'/'temporal')
        │  [B, M, D]   M ≤ T*N
        ▼
    Trainable temporal Transformer (3 layers)
        │
        ▼
    Mean pool + Linear classifier          [B, num_classes]

Only the temporal PE, the temporal Transformer, and the classifier head are
trainable — the ViT backbone is frozen. Trainable params: ~3-5M.

The merging operation is permutation-invariant w.r.t. the temporal head
(self-attention + mean pool both don't care about token order), so we don't
need to track per-token temporal positions through the merge — the temporal
PE added BEFORE ToMe carries the temporal information into each merged
token.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.video_transformer import FrozenViTBackbone
from utils.registry import register_model


# ───────────────────────────────────────────────────────────────────────────
# Bipartite soft matching — the operating mechanism behind every ToMe variant.
# ───────────────────────────────────────────────────────────────────────────

def bipartite_soft_match(
    tokens: torch.Tensor,
    t_idx: torch.Tensor,
    ratio: float,
    mode: str = "spatial",
    tau: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One round of token merging on a flattened spatio-temporal token sequence.

    Args:
        tokens: [B, N, D]. The patch tokens, already with temporal PE added.
        t_idx:  [N] long. Frame index of each token (0..T-1). Shared across
                the batch — every clip lays out tokens in the same T-major order.
        ratio:  fraction of `N` to remove. r = floor(N * ratio / 2) * 2 pairs
                are merged; the merged token replaces the B-set token, the
                A-set partner is dropped after contributing its features.
        mode:   'spatial' = unweighted cosine similarity (vanilla ToMe).
                'temporal' = cosine similarity * exp(-|Δt| / τ).

    Returns:
        merged tokens [B, M, D] where M = N - r,
        new_t_idx    [M] long, suitable for a follow-up temporal-aware op.

    NOTE: in the temporal-head architecture below we don't actually use
    new_t_idx (the head is order-invariant), but emitting it keeps this
    function reusable for stacked ToMe layers if we add them later.
    """
    B, N, D = tokens.shape
    r = (int(N * ratio) // 2) * 2  # round down to even
    if r <= 0:
        return tokens, t_idx

    # Split into bipartite sets A (even indices) and B (odd indices).
    a = tokens[:, 0::2]                              # [B, Na, D]
    b = tokens[:, 1::2]                              # [B, Nb, D]
    a_t = t_idx[0::2]                                # [Na]
    b_t = t_idx[1::2]                                # [Nb]
    Na, Nb = a.shape[1], b.shape[1]

    # Cosine similarity on token features. We use the post-PE feature itself
    # as the merge metric — Bolya et al. use the attention-K vector inside the
    # transformer; here, since our ToMe sits between the frozen backbone and
    # the trainable head, we don't have access to learned K's, and the raw
    # feature is a fine stand-in for "what does this patch represent".
    a_n = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    b_n = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    sim = a_n @ b_n.transpose(-2, -1)                # [B, Na, Nb]

    if mode == "temporal":
        dt = (a_t[:, None] - b_t[None, :]).abs().to(sim).float()  # [Na, Nb]
        kernel = torch.exp(-dt / max(tau, 1e-6)).unsqueeze(0)     # [1, Na, Nb]
        sim = sim * kernel
    elif mode != "spatial":
        raise ValueError(f"unknown ToMe mode {mode!r}")

    # For each A token, the index of its best B match and the score.
    best_score, best_b = sim.max(dim=-1)             # both [B, Na]

    # Pick the r/2 highest-scoring A tokens to merge into their best B partner;
    # keep the remaining Na - r/2 A tokens as-is.
    r_pairs = r // 2
    _, sorted_idx = best_score.sort(dim=-1, descending=True)
    merge_a_idx = sorted_idx[:, :r_pairs]            # [B, r/2]
    keep_a_idx = sorted_idx[:, r_pairs:]             # [B, Na - r/2]

    # Gather A subsets.
    keep_a = a.gather(1, keep_a_idx[..., None].expand(-1, -1, D))   # [B, Na - r/2, D]
    merge_a = a.gather(1, merge_a_idx[..., None].expand(-1, -1, D)) # [B, r/2, D]
    target_b = best_b.gather(1, merge_a_idx)         # [B, r/2]

    # Accumulate merging A's into their target B's. We use a running mean by
    # tracking a count per B slot — multiple A's can land on the same B (this
    # is the standard ToMe "many-to-one" merge).
    merged_b = b.clone()
    counts = torch.ones(B, Nb, 1, device=tokens.device, dtype=tokens.dtype)
    merged_b.scatter_add_(1, target_b[..., None].expand(-1, -1, D), merge_a)
    counts.scatter_add_(
        1,
        target_b[..., None],
        torch.ones(B, r_pairs, 1, device=tokens.device, dtype=tokens.dtype),
    )
    merged_b = merged_b / counts                      # [B, Nb, D]

    # Concat kept-A + merged-B. Order doesn't matter for downstream attention.
    out = torch.cat([keep_a, merged_b], dim=1)        # [B, Na - r/2 + Nb, D]

    # New t_idx: kept A's t_idx (per-batch — but t_idx is shared; we lose
    # exact alignment when batches keep different A tokens). For our use case
    # (order-invariant temporal head) this is fine; we return a representative
    # ordering that uses batch 0's keep ordering.
    keep_a_t = a_t[keep_a_idx[0].cpu()]               # [Na - r/2]
    new_t_idx = torch.cat([keep_a_t, b_t], dim=0)
    return out, new_t_idx


# ───────────────────────────────────────────────────────────────────────────
# Track 2 model
# ───────────────────────────────────────────────────────────────────────────

@register_model("temporal_tome_model")
class TemporalToMeVideoModel(nn.Module):
    """End-to-end video transformer with optional temporal-aware ToMe."""

    def __init__(
        self,
        num_classes: int,
        num_frames: int = 16,
        image_size: int = 224,
        backbone_name: str = "vit_small_patch16_224.augreg_in21k_ft_in1k",
        # Temporal head dims default to the backbone's embed_dim; we set them
        # here so the head doesn't need to introspect the backbone at config
        # parse time.
        d_model: int = 384,
        nhead: int = 6,
        num_layers: int = 3,
        dim_feedforward: int = 768,
        dropout: float = 0.2,
        # Token-merging knobs.
        tome_mode: str = "none",     # 'none' | 'spatial' | 'temporal'
        tome_ratio: float = 0.5,     # fraction of tokens removed by ToMe
        tome_tau: float = 2.0,       # temporal kernel width (frames)
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.tome_mode = tome_mode
        self.tome_ratio = tome_ratio
        self.tome_tau = tome_tau

        self.backbone = FrozenViTBackbone(model_name=backbone_name, image_size=image_size)
        if d_model != self.backbone.embed_dim:
            raise ValueError(
                f"d_model={d_model} must match backbone embed_dim={self.backbone.embed_dim}"
            )
        self.num_patches = self.backbone.num_patches

        # Temporal positional embedding broadcast over patches. Spatial PE is
        # already inside the ViT (on patch tokens before its transformer).
        self.temporal_pe = nn.Parameter(torch.zeros(1, num_frames, 1, d_model))
        nn.init.trunc_normal_(self.temporal_pe, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_head = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    @staticmethod
    def _make_t_idx(T: int, N: int, device: torch.device) -> torch.Tensor:
        """Frame-index per spatio-temporal token, in T-major order."""
        return torch.arange(T, device=device).repeat_interleave(N)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [B, T, 3, H, W]
        B, T, C, H, W = frames.shape
        if T != self.num_frames:
            raise ValueError(f"got T={T} but model was built for num_frames={self.num_frames}")

        # 1) Per-frame backbone forward (frozen, no_grad inside).
        flat = frames.reshape(B * T, C, H, W)
        tokens = self.backbone(flat)                     # [B*T, N, D]
        N = tokens.shape[1]
        D = tokens.shape[2]
        tokens = tokens.reshape(B, T, N, D)

        # 2) Add temporal PE; spatial PE is already on tokens from ViT.
        tokens = tokens + self.temporal_pe[:, :T, :, :]

        # 3) Flatten to a single sequence per clip.
        tokens_flat = tokens.reshape(B, T * N, D)
        t_idx = self._make_t_idx(T, N, tokens.device)

        # 4) Token merging.
        if self.tome_mode != "none":
            tokens_flat, _ = bipartite_soft_match(
                tokens_flat, t_idx, self.tome_ratio,
                mode=self.tome_mode, tau=self.tome_tau,
            )

        # 5) Trainable temporal head.
        h = self.temporal_head(tokens_flat)
        h = self.norm(h)

        # 6) Mean pool over remaining tokens, classify.
        pooled = h.mean(dim=1)
        return self.classifier(pooled)
