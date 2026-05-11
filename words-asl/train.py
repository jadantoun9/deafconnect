"""
train.py — model-agnostic trainer.

Reads a YAML config, builds the model + datasets via the registry, runs the
loop. The same script trains Track 1 (landmark), Track 2 (end-to-end), and
Track 3 (distilled) — only the config changes.

Loop features:
  - AMP autocast + GradScaler when CUDA is available.
  - AdamW + cosine schedule with linear warmup.
  - Per-step gradient clipping (config: training.grad_clip).
  - W&B logging via utils.wandb_setup (no-op stub if disabled).
  - Best-checkpoint tracker on val top-1; periodic full checkpoints under
    checkpoints/<run_name>/.
  - Distillation mode (config: distillation.enabled) loads a frozen teacher
    and combines hard CE + soft KL on logits.

Run:
    python train.py --config configs/track1_landmark.yaml

A --dry-run flag builds everything but stops before the first batch — useful
to validate the config + model + dataset wiring without spending compute.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from utils import checkpoint as ckpt_utils
from utils.registry import build_dataset, build_model
from utils.wandb_setup import init_run


# ────────────────────────────────────────────────────────
# Config loading
# ────────────────────────────────────────────────────────

def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    cfg.setdefault("training", {})
    cfg.setdefault("logging", {})
    cfg.setdefault("distillation", {"enabled": False})
    return cfg


def _resolve_device(pref: str) -> torch.device:
    """Map config training.device into a torch.device.

    "auto" picks cuda > mps > cpu. Anything else is passed through verbatim.
    Apple Silicon laptops need this; without it `device: auto` would crash
    `torch.device("auto")`.
    """
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(pref)


# ────────────────────────────────────────────────────────
# Model / dataset wiring (the only registry-aware section)
# ────────────────────────────────────────────────────────

def build_components(cfg: dict[str, Any]) -> tuple[nn.Module, DataLoader, DataLoader]:
    model_cfg = cfg["model"]
    model = build_model(model_cfg["name"], **model_cfg.get("args", {}))

    ds_cfg = cfg["dataset"]
    train_ds = build_dataset(ds_cfg["name"], **ds_cfg["train_args"])
    val_ds = build_dataset(ds_cfg["name"], **ds_cfg["val_args"])
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=True,
    )
    return model, train_loader, val_loader


# ────────────────────────────────────────────────────────
# Schedule
# ────────────────────────────────────────────────────────

def cosine_warmup_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))


# ────────────────────────────────────────────────────────
# Distillation teacher
# ────────────────────────────────────────────────────────

def load_teacher(cfg: dict[str, Any], device: torch.device) -> nn.Module:
    """Frozen Track-1 model loaded for soft-label distillation."""
    teacher_path = cfg["distillation"]["teacher_checkpoint"]
    payload = ckpt_utils.load(teacher_path, map_location=device)
    teacher_cfg = payload.get("config", {}).get("model")
    if teacher_cfg is None:
        raise RuntimeError(
            "Teacher checkpoint does not embed its config. Re-train Track 1 with "
            "this trainer (which embeds config) or specify model.teacher in the "
            "student config explicitly."
        )
    teacher = build_model(teacher_cfg["name"], **teacher_cfg.get("args", {}))
    teacher.load_state_dict(payload["model"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher.to(device)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    temperature: float,
) -> torch.Tensor:
    """
    Combined hard-label CE + soft-label KL.

    L = alpha * CE(student, target) + (1 - alpha) * T^2 * KL(soft_s || soft_t)

    The T^2 factor preserves the gradient magnitude as the temperature
    softens the targets — see Hinton et al. 2015. Owner can revisit alpha and
    T via config; alternatives (feature-matching / attention-transfer) are
    documented in docs/decisions.md under Track 3.
    """
    hard = F.cross_entropy(student_logits, targets)
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    soft = F.kl_div(s, t, reduction="batchmean") * (temperature ** 2)
    return alpha * hard + (1 - alpha) * soft


# ────────────────────────────────────────────────────────
# Forward / batch unpack — handles either video or landmark batches
# ────────────────────────────────────────────────────────

def forward_batch(model: nn.Module, batch: tuple, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (logits, targets) for a heterogeneous batch.

    VideoDataset yields:    (frames, label, video_id, meta)
    LandmarkDataset yields: (landmarks, valid_mask, label, video_id, meta)

    Both shapes are common enough that gating on tuple length keeps train.py
    free of dataset-specific branches.
    """
    if len(batch) == 4:
        frames, labels, _, _ = batch
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(frames)
    elif len(batch) == 5:
        lm, valid, labels, _, _ = batch
        lm = lm.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        # Models that ignore valid masks should accept the kwarg-less call.
        try:
            logits = model(lm, valid_mask=valid)
        except TypeError:
            logits = model(lm)
    else:
        raise RuntimeError(f"unexpected batch arity: {len(batch)}")
    return logits, labels


# ────────────────────────────────────────────────────────
# Train / eval
# ────────────────────────────────────────────────────────

@torch.no_grad()
def run_validation(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    correct1 = correct5 = total = 0
    loss_sum = 0.0
    for batch in loader:
        logits, labels = forward_batch(model, batch, device)
        loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        _, top5 = logits.topk(min(5, logits.size(-1)), dim=-1)
        correct1 += (top5[:, 0] == labels).sum().item()
        correct5 += (top5 == labels.unsqueeze(-1)).any(dim=-1).sum().item()
        total += labels.numel()
    return {
        "val_loss": loss_sum / max(total, 1),
        "val_top1": correct1 / max(total, 1),
        "val_top5": correct5 / max(total, 1),
    }


def train(cfg: dict[str, Any], dry_run: bool = False) -> None:
    seed = cfg["training"].get("seed", 0)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = _resolve_device(cfg["training"].get("device", "auto"))
    print(f"[train] device={device}")

    model, train_loader, val_loader = build_components(cfg)
    model = model.to(device)
    print(f"[train] model={cfg['model']['name']} params={sum(p.numel() for p in model.parameters()):,}")

    teacher = None
    if cfg["distillation"].get("enabled"):
        teacher = load_teacher(cfg, device)
        print(f"[train] distillation teacher loaded from {cfg['distillation']['teacher_checkpoint']}")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )

    epochs = cfg["training"]["epochs"]
    steps_per_epoch = max(1, len(train_loader))
    total_steps = epochs * steps_per_epoch
    warmup_steps = cfg["training"].get("warmup_epochs", 1) * steps_per_epoch

    use_amp = cfg["training"].get("amp", True) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    run_name = f"{cfg.get('track', 'unknown')}-{cfg.get('dataset_name', 'wlasl')}-s{seed}"
    out_dir = Path(cfg["training"].get("out_dir", "checkpoints")) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_tracker = ckpt_utils.BestTracker(out_dir, mode="max")

    logger = init_run(
        config=cfg,
        track=cfg.get("track", "unknown"),
        dataset=cfg.get("dataset_name", "wlasl"),
        seed=seed,
    )

    if dry_run:
        print("[train] --dry-run set, exiting before first batch.")
        return

    grad_clip = cfg["training"].get("grad_clip", 1.0)
    alpha = cfg["distillation"].get("alpha", 0.5)
    temperature = cfg["distillation"].get("temperature", 4.0)

    # Class-weighted cross-entropy. `auto` derives weights inversely
    # proportional to class frequency in the train manifest — useful for the
    # WLASL-N subsets where local download yield varies per class.
    class_weights: torch.Tensor | None = None
    cw_cfg = cfg["training"].get("class_weights")
    if cw_cfg == "auto":
        import json as _json
        manifest = _json.loads(Path(cfg["dataset"]["train_args"]["manifest_path"]).read_text())
        num_classes = cfg["model"]["args"]["num_classes"]
        counts = [0] * num_classes
        for it in manifest:
            counts[int(it["label"])] += 1
        # Inverse-frequency, normalised so the mean weight is 1.0 — keeps the
        # effective loss scale comparable to the unweighted run.
        inv = [1.0 / max(c, 1) for c in counts]
        mean_inv = sum(inv) / len(inv)
        weights = [w / mean_inv for w in inv]
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        print(f"[train] class_weights=auto -> counts={counts} weights={[round(w,2) for w in weights]}")
    elif isinstance(cw_cfg, list):
        class_weights = torch.tensor(cw_cfg, dtype=torch.float32, device=device)
        print(f"[train] class_weights (manual) = {cw_cfg}")

    step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for batch in train_loader:
            lr = cosine_warmup_lr(step, total_steps, warmup_steps, cfg["training"]["lr"])
            for g in opt.param_groups:
                g["lr"] = lr

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, labels = forward_batch(model, batch, device)
                if teacher is not None:
                    with torch.no_grad():
                        teacher_logits, _ = forward_batch(teacher, batch, device)
                    loss = distillation_loss(logits, teacher_logits, labels, alpha, temperature)
                else:
                    loss = F.cross_entropy(logits, labels, weight=class_weights)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()

            running_loss += loss.item()
            step += 1
            if step % 50 == 0:
                logger.log({"train_loss": loss.item(), "lr": lr, "epoch": epoch}, step=step)

        avg_loss = running_loss / steps_per_epoch
        metrics = run_validation(model, val_loader, device)
        metrics.update({"epoch": epoch, "train_loss_epoch": avg_loss, "epoch_seconds": time.time() - t0})
        logger.log(metrics, step=step)
        print(
            f"[train] epoch {epoch}/{epochs} loss={avg_loss:.4f} "
            f"val_top1={metrics['val_top1']:.4f} val_top5={metrics['val_top5']:.4f} "
            f"({metrics['epoch_seconds']:.0f}s)"
        )

        payload = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch": epoch,
            "metric": metrics["val_top1"],
            "config": cfg,
        }
        ckpt_utils.save(out_dir / "last.pt", payload)
        if best_tracker.observe(metrics["val_top1"], payload):
            print(f"[train]   new best val_top1={metrics['val_top1']:.4f} -> {out_dir / 'best.pt'}")

    logger.finish()
    print(f"[train] done. Checkpoints in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
