"""
checkpoint.py — save/load helpers + best-checkpoint tracking.

Behaviour:
  - `save(path, state)` writes a Torch checkpoint atomically (write to a
    sibling `.tmp` then rename) so a crash mid-write doesn't leave a
    half-finished file at the canonical path.
  - `BestTracker.observe(value, ckpt_payload)` keeps the highest-value
    checkpoint at <out_dir>/best.pt and overwrites it only when a strictly
    better value arrives (configurable via `mode` for min/max metrics).
  - `load(path)` returns the raw payload dict; the caller hydrates state
    dicts.

Checkpoints are deliberately framework-light — just `torch.save` of a dict
with at least:

    {
      "model": <state_dict>,
      "optimizer": <state_dict | None>,
      "epoch": int,
      "metric": float,
      "config": dict | None,   # the parsed YAML, for reproducibility
    }

`config` is included so downstream eval / conversion scripts can read the
settings the model was trained under without needing the YAML file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def save(path: Path | str, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load(path: Path | str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(str(path), map_location=map_location)


class BestTracker:
    """Track the best metric seen and write a single `best.pt`."""

    def __init__(self, out_dir: Path | str, mode: str = "max"):
        if mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")
        self.out_dir = Path(out_dir)
        self.mode = mode
        self.best: float | None = None

    def is_better(self, value: float) -> bool:
        if self.best is None:
            return True
        return (value > self.best) if self.mode == "max" else (value < self.best)

    def observe(self, value: float, payload: dict[str, Any]) -> bool:
        if not self.is_better(value):
            return False
        self.best = value
        save(self.out_dir / "best.pt", payload)
        return True
