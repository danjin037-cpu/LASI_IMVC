from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Method constants live here instead of being repeated in experiment YAML files.
DEFAULT_CONFIG: dict[str, Any] = {
    "device": "auto",
    "output_dir": "outputs/scene15",
    "seeds": [0, 1, 2, 3, 4],
    "fairness": {
        "strict_imvc": True,
        "allow_external_always_observed": False,
        "allow_complete_view_pretrain": False,
    },
    "model": {
        "shared_dim": 128,
        "encoder_hidden_dims": [1024, 512],
        "decoder_hidden_dims": [512, 1024],
        "dropout": 0.2,
        "norm": "bn",
        "cluster_temperature": 0.1,
        "use_sinkhorn": False,
        "evidence_hidden_dim": 64,
        "evidence_dropout": 0.1,
        "fusion_router_hidden_dim": 128,
        "fusion_router_temperature": 0.5,
        "beta_self_uncertainty": 0.5,
        "beta_self_entropy": 0.5,
        "lambda_self_reliability": 0.7,
        "lambda_agreement": 0.3,
        "discount_rho": 1.0,
    },
    "train": {
        "batch_size": 1024,
        "eval_batch_size": 1024,
        "num_workers": 0,
        "pin_mem": True,
        "pretrain_epochs": 50,
        "joint_epochs": 100,
        "cluster_start_epoch": 10,
        "pretrain_use_complete_views": False,
    },
    "optimizer": {
        "name": "adam",
        "pretrain_lr": 5.0e-4,
        "joint_lr": 1.0e-4,
        "betas": [0.9, 0.99],
        "weight_decay": 0.0,
    },
    "loss": {
        "lambda_cons": 0.5,
        "lambda_cluster": 1.0,
        "lambda_balance": 0.05,
        "lambda_hg": 0.0,
        "cluster_on_reliable_samples": True,
        "cluster_min_observed_views": 2,
        "balance_on_reliable_samples": True,
        "lambda_view_distill": 0.4,
        "lambda_single_proto": 0.2,
        "r3plus_distill_start_epoch": 15,
        "r3plus_proto_start_epoch": 25,
        "r3plus_teacher_conf_threshold": 0.6,
        "r3plus_clip_conflict_filter": True,
        "r3plus_clip_conflict_mode": "pred",
        "r3plus_clip_conflict_kl_threshold": 0.5,
        "r3plus_proto_bank_fallback_to_unfiltered": False,
        "r3plus_proto_bank_rebuild_interval": 0,
        "r3plus_proto_teacher_temperature": 0.2,
        "r3plus_proto_student_temperature": 1.0,
        "r3plus_proto_conf_threshold": 0.45,
        "r3plus_proto_margin_threshold": 0.1,
        "r3plus_single_proto_require_no_clip": False,
        "r3plus_view_student_temperature": 1.0,
    },
    "eval": {
        "eval_freq": 5,
        "eval_protocol": "sklearn_ninit10",
        "kmeans_n_init": 10,
        "kmeans_repeats": 10,
    },
    "logging": {
        "print_freq": 1,
        "save_best": True,
        "save_last": True,
    },
    "dataset": {
        "name": "Scene15",
        "file_path": "data/Scene_15.mat",
        "data_norm": "standard",
        "missing_rate": 0.5,
        "missing_protocol": "sample_wise",
        "ensure_at_least_one_view": True,
    },
    "clip_view": {
        "enabled": True,
        "as_view": True,
        "feature_path": "data/clip_cache/scene15_vitb32/clip_img_feat.pt",
        "normalize": "l2",
        "derived_from_raw_image": True,
        "mask_policy": "all_sources_observed",
        "source_view_indices": [0, 1],
        "original_num_views": 2,
    },
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key not in base:
            raise KeyError(f"Unknown configuration key: {key}")
        if isinstance(value, dict):
            if not isinstance(base[key], dict):
                raise TypeError(f"Configuration key '{key}' is not a section")
            merged[key] = _merge(base[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_path(value: str) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}
    if not isinstance(user_config, dict):
        raise TypeError("The YAML root must be a mapping")

    config = _merge(DEFAULT_CONFIG, user_config)
    config["dataset"]["file_path"] = _resolve_path(config["dataset"]["file_path"])
    config["clip_view"]["feature_path"] = _resolve_path(config["clip_view"]["feature_path"])
    config["output_dir"] = _resolve_path(config["output_dir"])
    return config
