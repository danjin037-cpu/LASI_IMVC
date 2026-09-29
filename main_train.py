from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from dataset_loader import get_dataloaders
from engine_train import train_pipeline
from models import LASIIMVC
from utils import FileLogger, prepare_device, print_config, set_seed


PROJECT_ROOT = Path(__file__).resolve().parent


# Stable implementation details are defined here. Dataset facts, architecture sizes,
# and the four paper-facing semantic-transfer parameters remain explicit in Scene15.yaml.
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
        "shared_dim": None,
        "encoder_hidden_dims": None,
        "decoder_hidden_dims": None,
        "norm": None,
        "dropout": 0.2,
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
        "lambda_view_distill": None,
        "lambda_single_proto": None,
        "tau_t": None,
        "tau_p": None,
        "lambda_cons": 0.5,
        "lambda_cluster": 1.0,
        "lambda_balance": 0.05,
        "lambda_hg": 0.0,
        "cluster_on_reliable_samples": True,
        "cluster_min_observed_views": 2,
        "balance_on_reliable_samples": True,
        "r3plus_distill_start_epoch": 15,
        "r3plus_proto_start_epoch": 25,
        "r3plus_clip_conflict_filter": True,
        "r3plus_clip_conflict_mode": "pred",
        "r3plus_clip_conflict_kl_threshold": 0.5,
        "r3plus_proto_bank_fallback_to_unfiltered": False,
        "r3plus_proto_bank_rebuild_interval": 0,
        "r3plus_proto_teacher_temperature": 0.2,
        "r3plus_proto_student_temperature": 1.0,
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
        "num_classes": None,
        "num_samples_total": None,
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

REQUIRED_YAML_KEYS = (
    ("dataset", "num_classes"),
    ("dataset", "num_samples_total"),
    ("model", "shared_dim"),
    ("model", "encoder_hidden_dims"),
    ("model", "decoder_hidden_dims"),
    ("model", "norm"),
    ("loss", "lambda_view_distill"),
    ("loss", "lambda_single_proto"),
    ("loss", "tau_t"),
    ("loss", "tau_p"),
)


def _merge_config(base: dict[str, Any], override: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if key not in base:
            raise KeyError(f"Unknown configuration key: {full_key}")
        if isinstance(value, dict):
            if not isinstance(base[key], dict):
                raise TypeError(f"Configuration key '{full_key}' is not a section")
            merged[key] = _merge_config(base[key], value, full_key)
        else:
            merged[key] = value
    return merged


def _resolve_project_path(value: str) -> str:
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

    config = _merge_config(DEFAULT_CONFIG, user_config)
    missing = [".".join(keys) for keys in REQUIRED_YAML_KEYS if config[keys[0]][keys[1]] is None]
    if missing:
        raise ValueError(f"Missing required YAML values: {', '.join(missing)}")
    config["dataset"]["file_path"] = _resolve_project_path(config["dataset"]["file_path"])
    config["clip_view"]["feature_path"] = _resolve_project_path(config["clip_view"]["feature_path"])
    config["output_dir"] = _resolve_project_path(config["output_dir"])
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LASI-IMVC on Scene15")
    parser.add_argument("--config", default="config/Scene15.yaml")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seeds", default=None, help="Comma-separated training seeds")
    parser.add_argument("--mask-seed", type=int, default=None)
    parser.add_argument("--missing-rate", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _apply_cli(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.device is not None:
        cfg["device"] = args.device
    if args.seeds is not None:
        cfg["seeds"] = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not cfg["seeds"]:
        raise ValueError("At least one training seed is required")
    if args.mask_seed is not None:
        cfg["dataset"]["fixed_mask_seed"] = int(args.mask_seed)
    if args.missing_rate is not None:
        cfg["dataset"]["missing_rate"] = float(args.missing_rate)
    if args.output_dir is not None:
        cfg["output_dir"] = str(Path(args.output_dir).expanduser().resolve())
    return cfg


def _log_dataset(logger: FileLogger, cfg: dict[str, Any], meta: dict[str, Any]) -> None:
    logger.write(
        "Dataset: Scene15 | samples={} | original_views={} | model_views={} | "
        "classes={} | input_dims={} | missing_rate={:.2f} | mask_seed={}".format(
            meta["num_samples_total"],
            meta["original_num_views"],
            meta["num_views"],
            meta["num_classes"],
            meta["input_dims"],
            cfg["dataset"]["missing_rate"],
            cfg["dataset"]["mask_seed"],
        )
    )
    logger.write(
        "Original-view mask: avg_observed={:.4f} single={:.4f} complete={:.4f}".format(
            meta["original_avg_observed_views_per_sample"],
            meta["original_single_view_sample_ratio"],
            meta["original_complete_sample_ratio"],
        )
    )


def run_seed(base_cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["seed"] = int(seed)
    cfg["train_seed"] = int(seed)
    cfg["dataset"]["mask_seed"] = int(cfg["dataset"].get("fixed_mask_seed", seed))
    run_dir = Path(cfg["output_dir"]) / (
        f"scene15_mr{cfg['dataset']['missing_rate']:.1f}_ts{seed}_ms{cfg['dataset']['mask_seed']}"
    )
    cfg["output_dir"] = str(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = FileLogger(str(run_dir / "train.log"))
    logger.write(print_config(cfg))
    set_seed(seed)
    device = prepare_device(cfg["device"])
    logger.write(f"Device: {device}")

    train_loader, eval_loader, meta = get_dataloaders(cfg)
    cfg["dataset"].update(
        {
            "n_views": meta["num_views"],
            "input_dims": meta["input_dims"],
            "original_num_views": meta["original_num_views"],
            "original_input_dims": meta["original_input_dims"],
            "clip_view_enabled": True,
            "clip_view_index": meta["clip_view_index"],
        }
    )
    cfg["clip_view"]["view_index"] = meta["clip_view_index"]
    _log_dataset(logger, cfg, meta)

    model = LASIIMVC(cfg, meta["input_dims"]).to(device)
    started = time.time()
    metrics, best_epoch = train_pipeline(
        cfg, model, train_loader, eval_loader, device, logger
    )
    logger.write(f"Elapsed seconds: {time.time() - started:.1f}")
    result = {
        "train_seed": seed,
        "mask_seed": cfg["dataset"]["mask_seed"],
        "best_epoch": best_epoch,
        **{
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        },
        "output_dir": str(run_dir),
    }

    del model, train_loader, eval_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def write_summary(results: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for result in results for key in result})
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    core = ("acc", "nmi", "ari", "purity")
    summary = {
        metric: {
            "mean": float(torch.tensor([row[metric] for row in results]).mean()),
            "std": float(torch.tensor([row[metric] for row in results]).std(unbiased=False)),
        }
        for metric in core
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"runs": results, "aggregate": summary}, handle, indent=2)

    for metric, values in summary.items():
        print(f"{metric.upper()}: {values['mean']:.6f} ± {values['std']:.6f}")


def main() -> None:
    args = parse_args()
    cfg = _apply_cli(load_config(args.config), args)
    root_output = Path(cfg["output_dir"])
    results = [run_seed(cfg, int(seed)) for seed in cfg["seeds"]]
    write_summary(results, root_output)


if __name__ == "__main__":
    main()
