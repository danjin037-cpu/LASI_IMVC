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

from lasi_imvc.config import load_config
from lasi_imvc.data import get_dataloaders
from lasi_imvc.engine_train import train_pipeline
from lasi_imvc.models import LASIIMVC
from lasi_imvc.utils import FileLogger, prepare_device, print_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LASI-IMVC on Scene15")
    parser.add_argument("--config", default="configs/scene15.yaml")
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
            "n_classes": meta["num_classes"],
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
