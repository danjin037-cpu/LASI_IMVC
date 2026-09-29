from __future__ import annotations

import os
import random
from typing import Any, Dict

import numpy as np
import torch
import yaml


class AverageMeter:
    """Accumulates weighted averages for training statistics."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * int(n)
        self.count += int(n)
        self.avg = self.sum / max(self.count, 1)


class FileLogger:
    """Prints messages to console and appends them to log_train.txt."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(self.file_path, "w", encoding="utf-8") as f:
            f.write("")

    def write(self, msg: str) -> None:
        print(msg)
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")


def set_seed(seed: int) -> None:
    """Fixes training randomness. Missing-mask randomness is controlled separately by dataset.mask_seed."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_device(device: str | None) -> torch.device:
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    return torch.device(device)


def print_config(cfg: Dict[str, Any]) -> str:
    return yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)


def save_checkpoint(state: Dict[str, Any], output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    torch.save(state, path)
    return path


def get_or_create_observed_mask(batch: Dict[str, Any], cfg: Dict[str, Any] | None = None, **_) -> torch.Tensor:
    """
    Returns observed_mask from the dataloader. Kept as a tiny compatibility helper.
    If older data code does not provide a mask, all views are treated as observed.
    """
    if "observed_mask" in batch and batch["observed_mask"] is not None:
        mask = batch["observed_mask"]
        if isinstance(mask, torch.Tensor):
            return mask.to(batch["views"][0].device)
        return torch.as_tensor(mask, dtype=torch.bool, device=batch["views"][0].device)

    batch_size = batch["views"][0].shape[0]
    num_views = len(batch["views"])
    return torch.ones(batch_size, num_views, dtype=torch.bool, device=batch["views"][0].device)
