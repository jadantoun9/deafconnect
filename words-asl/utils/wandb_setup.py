"""
wandb_setup.py — light wrapper around `wandb.init` with project conventions.

Why this exists rather than calling wandb.init directly:
  - Centralises run-naming + tagging convention so every experiment is
    findable later. Tags are: track, dataset, config-hash, seed.
  - Lets the trainer fall back to a no-op stub when WANDB_DISABLED=1 or the
    `wandb` package isn't installed, so the training loop never has to gate
    every log call on `if wandb_enabled`.

Environment variables read:
    WANDB_API_KEY     — auth token (set by `wandb login` or .env).
    WANDB_PROJECT     — project name; defaults to 'signreader'.
    WANDB_ENTITY      — team / username override.
    WANDB_DISABLED=1  — skip W&B entirely; logger prints to stdout.

The owner sets these by following docs/wandb-setup.md.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any


class _StubLogger:
    """No-op replacement when W&B is unavailable or disabled."""

    def __init__(self, run_name: str):
        self.run_name = run_name
        self.url = ""

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        if step is not None:
            print(f"[stub-wandb step={step}] {data}")
        else:
            print(f"[stub-wandb] {data}")

    def finish(self) -> None:
        pass


def _config_hash(cfg: dict[str, Any]) -> str:
    """Stable short hash of the resolved config — used as a run tag."""
    payload = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha1(payload).hexdigest()[:8]


def init_run(
    config: dict[str, Any],
    track: str,
    dataset: str,
    seed: int,
    extra_tags: list[str] | None = None,
):
    """Returns a logger object with `.log(dict, step=None)` and `.finish()`."""
    if os.environ.get("WANDB_DISABLED") == "1":
        return _StubLogger(run_name=f"{track}-{dataset}-{seed}")
    try:
        import wandb
    except ImportError:
        return _StubLogger(run_name=f"{track}-{dataset}-{seed}")

    project = os.environ.get("WANDB_PROJECT", "signreader")
    entity = os.environ.get("WANDB_ENTITY")
    cfg_hash = _config_hash(config)
    tags = [track, dataset, f"cfg-{cfg_hash}", f"seed-{seed}"]
    if extra_tags:
        tags.extend(extra_tags)

    run = wandb.init(
        project=project,
        entity=entity,
        config=config,
        tags=tags,
        name=f"{track}-{dataset}-{cfg_hash}-s{seed}",
        reinit=True,
    )
    return run
