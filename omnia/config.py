"""Configuration shared by the service and model clients."""

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BarsConfig:
    enabled: bool = False
    min_group_size: int = 2
    alpha: float = 0.5
    beta: float = 1.0
    metrics_timeout_seconds: float = 1.0
    metrics_url: str | None = None

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("bars.enabled must be a boolean")
        if type(self.min_group_size) is not int or self.min_group_size < 1:
            raise ValueError("bars.min_group_size must be a positive integer")
        for name in ("alpha", "beta", "metrics_timeout_seconds"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"bars.{name} must be finite and non-negative")
        if self.metrics_timeout_seconds == 0:
            raise ValueError("bars.metrics_timeout_seconds must be positive")


@dataclass(frozen=True)
class PipelineConfig:
    group_size: int = 2
    initial_fraction: float = 0.2
    stability_fraction: float = 0.1
    min_stable_candidates: int = 2
    enable_pruning: bool = True
    enable_prefill: bool = True
    max_tokens: int = 512

    def __post_init__(self):
        for name in ("group_size", "min_stable_candidates", "max_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("initial_fraction", "stability_fraction"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        for name in ("enable_pruning", "enable_prefill"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")


def load_config(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as source:
        config = json.load(source)
    PipelineConfig(**config.get("pipeline", {}))
    BarsConfig(**config.get("bars", {}))
    if config.get("dataset_path"):
        config["dataset_path"] = str((path.parent / config["dataset_path"]).resolve())
    return config
