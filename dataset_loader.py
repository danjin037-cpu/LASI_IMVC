from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class Scene15Dataset(Dataset):
    def __init__(
        self,
        views: list[np.ndarray],
        labels: np.ndarray,
        observed_mask: np.ndarray,
        original_mask: np.ndarray,
    ) -> None:
        self.views = [torch.from_numpy(view.astype(np.float32, copy=False)) for view in views]
        self.labels = torch.from_numpy(labels.astype(np.int64, copy=False))
        self.observed_mask = torch.from_numpy(observed_mask.astype(bool, copy=False))
        self.original_mask = torch.from_numpy(original_mask.astype(bool, copy=False))

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "index": index,
            "views": [view[index] for view in self.views],
            "label": self.labels[index],
            "observed_mask": self.observed_mask[index],
            "original_observed_mask": self.original_mask[index],
            "is_complete": bool(self.original_mask[index].all()),
        }


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    num_views = len(batch[0]["views"])
    return {
        "index": torch.tensor([item["index"] for item in batch], dtype=torch.long),
        "views": [torch.stack([item["views"][v] for item in batch]) for v in range(num_views)],
        "label": torch.stack([item["label"] for item in batch]),
        "observed_mask": torch.stack([item["observed_mask"] for item in batch]),
        "original_observed_mask": torch.stack(
            [item["original_observed_mask"] for item in batch]
        ),
        "is_complete": torch.tensor([item["is_complete"] for item in batch], dtype=torch.bool),
    }


def _load_scene15(path: str) -> tuple[list[np.ndarray], np.ndarray]:
    mat_path = Path(path)
    if not mat_path.is_file():
        raise FileNotFoundError(f"Scene15 data not found: {mat_path}")
    mat = sio.loadmat(mat_path)
    if "X" not in mat or "Y" not in mat:
        raise KeyError("Scene_15.mat must contain 'X' views and 'Y' labels")

    cells = np.asarray(mat["X"], dtype=object).ravel()
    if len(cells) < 2:
        raise ValueError("Scene15 requires at least two feature views")

    # The published LASI-IMVC protocol uses the first two Scene15 feature views.
    views = [np.asarray(cells[i], dtype=np.float32) for i in (0, 1)]
    labels = np.asarray(mat["Y"]).reshape(-1)
    _, labels = np.unique(labels, return_inverse=True)
    for index, view in enumerate(views):
        if view.ndim != 2 or view.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Scene15 view {index} has shape {view.shape}; expected [N, D] with N={labels.shape[0]}"
            )
    return views, labels.astype(np.int64)


def _load_clip_features(path: str, num_samples: int) -> np.ndarray:
    feature_path = Path(path)
    if not feature_path.is_file():
        raise FileNotFoundError(f"CLIP cache not found: {feature_path}")
    try:
        payload = torch.load(feature_path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0
        payload = torch.load(feature_path, map_location="cpu")

    if isinstance(payload, dict):
        candidates = [value for value in payload.values() if hasattr(value, "shape")]
        payload = next(
            (value for value in candidates if len(value.shape) == 2 and num_samples in value.shape),
            None,
        )
    if payload is None:
        raise ValueError("No two-dimensional feature tensor was found in the CLIP cache")

    features = payload.detach().cpu().float() if isinstance(payload, torch.Tensor) else torch.as_tensor(payload).float()
    if features.ndim != 2:
        raise ValueError(f"CLIP features must be 2-D, got {tuple(features.shape)}")
    if features.shape[0] != num_samples and features.shape[1] == num_samples:
        features = features.t()
    if features.shape[0] != num_samples:
        raise ValueError(
            f"CLIP feature count {features.shape[0]} does not match Scene15 sample count {num_samples}"
        )
    return F.normalize(features, dim=1).numpy().astype(np.float32, copy=False)


def _build_sample_wise_mask(num_samples: int, missing_rate: float, seed: int) -> np.ndarray:
    if not 0.0 <= missing_rate <= 1.0:
        raise ValueError("dataset.missing_rate must be in [0, 1]")
    rng = np.random.default_rng(seed)
    mask = np.ones((num_samples, 2), dtype=bool)
    incomplete_count = int(np.floor(num_samples * missing_rate))
    if incomplete_count:
        rows = rng.permutation(num_samples)[:incomplete_count]
        kept_view = rng.integers(0, 2, size=incomplete_count)
        mask[rows] = False
        mask[rows, kept_view] = True
    return mask


def _mask_statistics(mask: np.ndarray) -> dict[str, float]:
    counts = mask.sum(axis=1)
    pair_ratios = [
        float((mask[:, left] & mask[:, right]).mean())
        for left in range(mask.shape[1])
        for right in range(left + 1, mask.shape[1])
    ]
    pair_ratio = float(np.mean(pair_ratios)) if pair_ratios else 0.0
    return {
        "avg_observed_views_per_sample": float(counts.mean()),
        "single_view_sample_ratio": float((counts == 1).mean()),
        "two_or_more_view_sample_ratio": float((counts >= 2).mean()),
        "complete_sample_ratio": float((counts == mask.shape[1]).mean()),
        "valid_pair_ratio": pair_ratio,
    }


def get_dataloaders(cfg: dict[str, Any]):
    dataset_cfg = cfg["dataset"]
    if dataset_cfg["name"] != "Scene15":
        raise ValueError("This release supports only dataset.name=Scene15")
    if dataset_cfg.get("missing_protocol") != "sample_wise":
        raise ValueError("This release supports only the sample_wise missing protocol")

    views, labels = _load_scene15(dataset_cfg["file_path"])
    actual_num_samples = int(labels.shape[0])
    actual_num_classes = int(np.unique(labels).size)
    expected_num_samples = int(dataset_cfg["num_samples_total"])
    expected_num_classes = int(dataset_cfg["num_classes"])
    if actual_num_samples != expected_num_samples:
        raise ValueError(
            f"Scene15 sample count mismatch: config={expected_num_samples}, data={actual_num_samples}"
        )
    if actual_num_classes != expected_num_classes:
        raise ValueError(
            f"Scene15 class count mismatch: config={expected_num_classes}, data={actual_num_classes}"
        )
    views = [StandardScaler().fit_transform(view).astype(np.float32) for view in views]
    clip = _load_clip_features(cfg["clip_view"]["feature_path"], len(labels))

    mask_seed = int(dataset_cfg.get("mask_seed", cfg["seed"]))
    original_mask = _build_sample_wise_mask(
        num_samples=len(labels),
        missing_rate=float(dataset_cfg["missing_rate"]),
        seed=mask_seed,
    )
    # CLIP is derived from the image and is legal only when both source views are observed.
    clip_mask = original_mask.all(axis=1, keepdims=True)
    observed_mask = np.concatenate([original_mask, clip_mask], axis=1)
    all_views = [*views, clip]

    dataset = Scene15Dataset(all_views, labels, observed_mask, original_mask)
    train_loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=bool(cfg["train"]["pin_mem"]),
        collate_fn=_collate,
    )
    eval_loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["eval_batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=bool(cfg["train"]["pin_mem"]),
        collate_fn=_collate,
    )

    original_stats = _mask_statistics(original_mask)
    all_stats = _mask_statistics(observed_mask)
    meta = {
        "input_dims": [view.shape[1] for view in all_views],
        "original_input_dims": [view.shape[1] for view in views],
        "num_views": len(all_views),
        "original_num_views": len(views),
        "num_classes": actual_num_classes,
        "num_samples_total": actual_num_samples,
        "train_size": len(dataset),
        "eval_size": len(dataset),
        "mask_seed": mask_seed,
        "missing_rate": float(dataset_cfg["missing_rate"]),
        "missing_protocol": "sample_wise",
        "strict_imvc": True,
        "clip_view_enabled": True,
        "clip_view_index": 2,
        "semantic_view_specs": [{"name": "clip", "view_index": 2, "source_view_indices": [0, 1]}],
        **all_stats,
        **{f"original_{key}": value for key, value in original_stats.items()},
    }
    return train_loader, eval_loader, meta
