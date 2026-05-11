"""
registry.py — tiny string-keyed factory for models and datasets.

Keeps train.py / eval.py model-agnostic: instead of hard-coding which model
class belongs to which track, configs reference architectures by name and
the registry resolves the constructor.

Usage:
    @register_model("landmark_classifier")
    class LandmarkClassifier(nn.Module):
        def __init__(self, num_classes: int, ...): ...

    model = build_model("landmark_classifier", num_classes=100, ...)

The registry is populated lazily by `import_track_modules()`, which imports
all known model files so their `@register_model` decorators run. Adding a
new track means: write the module, decorate it, and add the import path to
`_KNOWN_MODEL_MODULES` below.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, TypeVar

T = TypeVar("T")

_MODEL_REGISTRY: dict[str, Callable[..., Any]] = {}
_DATASET_REGISTRY: dict[str, Callable[..., Any]] = {}

_KNOWN_MODEL_MODULES = [
    # Each track adds its model module here once implemented.
    "models.landmark_classifier",     # Track 1 (1.2)
    "models.video_transformer",       # Track 2 (2.1)
    "models.temporal_tome_model",     # Track 2 / 3 wrapper
]

_KNOWN_DATASET_MODULES = [
    "data.video_dataset",
    "data.landmark_dataset",
]


def register_model(name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(cls: Callable[..., T]) -> Callable[..., T]:
        if name in _MODEL_REGISTRY:
            raise ValueError(f"model {name!r} already registered")
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def register_dataset(name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(cls: Callable[..., T]) -> Callable[..., T]:
        if name in _DATASET_REGISTRY:
            raise ValueError(f"dataset {name!r} already registered")
        _DATASET_REGISTRY[name] = cls
        return cls
    return decorator


def _ensure_loaded(modules: list[str]) -> None:
    for mod in modules:
        try:
            importlib.import_module(mod)
        except ImportError:
            # Modules not yet implemented (e.g. Track 2 before Phase 2 lands)
            # are silently skipped. The caller's KeyError surfaces the actual
            # missing model name.
            continue


def build_model(name: str, **kwargs: Any) -> Any:
    _ensure_loaded(_KNOWN_MODEL_MODULES)
    if name not in _MODEL_REGISTRY:
        raise KeyError(
            f"unknown model {name!r}. Registered: {sorted(_MODEL_REGISTRY)}. "
            f"If this model is in a new file, add its import path to "
            f"_KNOWN_MODEL_MODULES in /utils/registry.py."
        )
    return _MODEL_REGISTRY[name](**kwargs)


def build_dataset(name: str, **kwargs: Any) -> Any:
    _ensure_loaded(_KNOWN_DATASET_MODULES)
    if name not in _DATASET_REGISTRY:
        raise KeyError(
            f"unknown dataset {name!r}. Registered: {sorted(_DATASET_REGISTRY)}."
        )
    return _DATASET_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    _ensure_loaded(_KNOWN_MODEL_MODULES)
    return sorted(_MODEL_REGISTRY)


def list_datasets() -> list[str]:
    _ensure_loaded(_KNOWN_DATASET_MODULES)
    return sorted(_DATASET_REGISTRY)
