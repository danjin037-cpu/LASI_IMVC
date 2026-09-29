from __future__ import annotations

from typing import Dict

import torch

from .trainers.joint_trainer import JointTrainer
from .trainers.pretrain_trainer import PretrainTrainer
from .utils import save_checkpoint


def _build_optimizer(model, optimizer_cfg: Dict, lr: float):
    name = optimizer_cfg.get("name", "adam").lower()
    betas = tuple(optimizer_cfg.get("betas", [0.9, 0.99]))
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def _should_log(epoch: int, total_epochs: int, print_freq: int) -> bool:
    return (epoch + 1) % max(print_freq, 1) == 0 or (epoch + 1) == total_epochs


def _should_eval(epoch: int, total_epochs: int, cluster_start_epoch: int, eval_freq: int) -> bool:
    current = epoch + 1
    if current == total_epochs:
        return True
    if current < cluster_start_epoch:
        return False
    return (current - cluster_start_epoch) % max(eval_freq, 1) == 0


def _format_view_stats(stats: Dict) -> str:
    view_ids = sorted(int(k.replace("mean_w_view", "")) for k in stats if k.startswith("mean_w_view"))
    parts = []
    for v in view_ids:
        tokens = []
        mapping = [
            (f"mean_w_view{v}", f"w{v}"),
            (f"mean_u_view{v}", f"u{v}"),
            (f"mean_H_view{v}", f"H{v}"),
            (f"mean_agree_view{v}", f"agree{v}"),
            (f"mean_comb_rel_view{v}", f"combRel{v}"),
            (f"mean_obs_view{v}", f"obs{v}"),
        ]
        for key, short in mapping:
            if key in stats:
                tokens.append(f"{short}={stats[key]:.4f}")
        parts.append(" ".join(tokens))
    return " ".join(parts)


def _format_mask_stats(stats: Dict) -> str:
    mapping = [
        ("avg_observed_views_per_sample", "avgObs"),
        ("single_view_sample_ratio", "singleView"),
        ("two_or_more_view_sample_ratio", "ge2View"),
        ("complete_sample_ratio", "complete"),
        ("valid_pair_ratio", "validPair"),
        ("cluster_reliable_sample_ratio", "clusterRel"),
    ]
    return " ".join(f"{short}={stats[key]:.4f}" for key, short in mapping if key in stats)


def _format_scalar_or_na(stats: Dict, key: str, fmt: str = ".6f") -> str:
    """Format a scalar metric from stats; return NA if JointTrainer did not provide it."""
    if key not in stats:
        return "NA"
    value = stats[key]
    if isinstance(value, torch.Tensor):
        value = value.detach().float().mean().item()
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return str(value)


def _format_int_or_na(stats: Dict, key: str) -> str:
    """Format an integer/bool diagnostic flag from stats; return NA if absent."""
    if key not in stats:
        return "NA"
    value = stats[key]
    if isinstance(value, torch.Tensor):
        value = value.detach().float().mean().item()
    try:
        return str(int(round(float(value))))
    except (TypeError, ValueError):
        return str(value)


def _format_r3plus_stats(stats: Dict) -> str:
    """
    R3+ diagnostics.

    If a field is printed as NA, engine_train.py is already ready, but the current
    trainers/joint_trainer.py did not put that key into the returned stats dict.
    """
    mapping = [
        ("loss_view_distill", "loss_view_distill", ".6f", "float"),
        ("loss_single_proto", "loss_single_proto", ".6f", "float"),
        ("clip_source_violation_count", "clip_source_violation_count", ".0f", "float"),
        ("proto_bank_active", "proto_bank_active", "", "int"),
    ]

    parts = []
    for key, short, fmt, kind in mapping:
        if kind == "int":
            value = _format_int_or_na(stats, key)
        else:
            value = _format_scalar_or_na(stats, key, fmt)
        parts.append(f"{short}={value}")
    return " ".join(parts)


def _format_subset_eval_metrics(metrics: Dict) -> str:
    """Compact subset metrics for strict-IMVC analysis."""
    if "single_acc" not in metrics:
        return ""
    parts = [
        f"singleACC={metrics.get('single_acc', 0.0):.6f}",
        f"singleNMI={metrics.get('single_nmi', 0.0):.6f}",
        f"singleARI={metrics.get('single_ari', 0.0):.6f}",
        f"singleN={metrics.get('single_count', 0.0):.0f}",
        f"completeACC={metrics.get('complete_acc', 0.0):.6f}",
        f"completeNMI={metrics.get('complete_nmi', 0.0):.6f}",
        f"completeARI={metrics.get('complete_ari', 0.0):.6f}",
        f"completeN={metrics.get('complete_count', 0.0):.0f}",
        f"incompleteACC={metrics.get('incomplete_acc', 0.0):.6f}",
        f"incompleteN={metrics.get('incomplete_count', 0.0):.0f}",
    ]
    if metrics.get("multi_incomplete_count", 0.0) > 0:
        parts.extend([
            f"multiIncACC={metrics.get('multi_incomplete_acc', 0.0):.6f}",
            f"multiIncN={metrics.get('multi_incomplete_count', 0.0):.0f}",
        ])
    return " | " + " ".join(parts)


def run_pretrain(cfg, model, train_loader, eval_loader, device, logger) -> None:
    epochs = int(cfg["train"].get("pretrain_epochs", 0))
    if epochs <= 0:
        logger.write(">> Skip pretraining")
        return

    logger.write(">> Stage 0: reconstruction pretraining")
    trainer = PretrainTrainer(cfg)
    optimizer = _build_optimizer(model, cfg["optimizer"], float(cfg["optimizer"]["pretrain_lr"]))
    logger.write(str(optimizer))

    best_eval = float("inf")
    print_freq = int(cfg.get("logging", {}).get("print_freq", 1))
    for epoch in range(epochs):
        train_stats = trainer.train_one_epoch(model, train_loader, optimizer, device, epoch)
        eval_stats = trainer.evaluate(model, eval_loader, device)
        best_eval = min(best_eval, eval_stats["loss"])
        if _should_log(epoch, epochs, print_freq):
            logger.write(
                f"[Pretrain][Epoch {epoch + 1}/{epochs}] "
                f"train_loss={train_stats['loss']:.6f} eval_loss={eval_stats['loss']:.6f}"
            )
    logger.write(f"Finished pretraining. Best eval loss = {best_eval:.6f}")
    # Opt-in snapshot for representation-stage figures; no training/RNG changes.
    if bool(cfg.get("logging", {}).get("save_pretrain", False) or
            cfg.get("logging", {}).get("save_stage_snapshots", False)):
        save_checkpoint(
            {"model": model.state_dict(), "epoch": epochs, "stage": "pretrain", "cfg": cfg},
            cfg["output_dir"],
            "checkpoint_pretrain_last.pt",
        )


def run_joint_train(cfg, model, train_loader, eval_loader, device, logger):
    epochs = int(cfg["train"].get("joint_epochs", 100))
    trainer = JointTrainer(cfg)
    optimizer = _build_optimizer(model, cfg["optimizer"], float(cfg["optimizer"]["joint_lr"]))

    # The evaluation loader is deterministic, so prototype initialization is reproducible.
    model.init_prototypes_from_loader(eval_loader, device)

    logger.write(">> Stage 1: joint incomplete multi-view clustering")
    logger.write(str(optimizer))

    best_metrics = None
    best_epoch = -1
    best_acc = -1.0
    print_freq = int(cfg.get("logging", {}).get("print_freq", 1))
    eval_freq = int(cfg.get("eval", {}).get("eval_freq", 5))
    cluster_start_epoch = int(cfg["train"].get("cluster_start_epoch", 10))

    transfer_starts = [start for weight, start in (
        (trainer.lambda_view_distill, trainer.distill_start_epoch),
        (trainer.lambda_single_proto, trainer.proto_start_epoch),
    ) if weight > 0]
    first_transfer_epoch = min(transfer_starts) if transfer_starts else None

    for epoch in range(epochs):
        # Capture BEFORE the first transfer-enabled update. epoch counts completed
        # joint epochs here (e.g. 15 means epochs 1..15 have finished).
        if (bool(cfg.get("logging", {}).get("save_stage_snapshots", False))
                and epoch == first_transfer_epoch):
            save_checkpoint(
                {"model": model.state_dict(), "epoch": epoch,
                 "stage": "before_semantic_transfer", "cfg": cfg},
                cfg["output_dir"], "checkpoint_before_semantic_transfer.pt",
            )
        stats = trainer.train_one_epoch(model, train_loader, optimizer, device, epoch)

        eval_text = ""
        if _should_eval(epoch, epochs, cluster_start_epoch, eval_freq):
            metrics = trainer.evaluate_cluster(model, eval_loader, device)
            eval_text = (
                f" ACC={metrics['acc']:.6f} NMI={metrics['nmi']:.6f} "
                f"ARI={metrics['ari']:.6f} PUR={metrics['purity']:.6f}"
                f"{_format_subset_eval_metrics(metrics)}"
            )
            if metrics["acc"] > best_acc:
                best_acc = metrics["acc"]
                best_metrics = metrics
                best_epoch = epoch + 1
                if bool(cfg.get("logging", {}).get("save_best", True)):
                    save_checkpoint(
                        {"model": model.state_dict(), "epoch": best_epoch, "metrics": best_metrics, "cfg": cfg},
                        cfg["output_dir"],
                        "checkpoint_joint_best.pt",
                    )
        elif (epoch + 1) < cluster_start_epoch:
            eval_text = " (skip clustering eval: starts from epoch {})".format(cluster_start_epoch)
        else:
            eval_text = " (skip clustering eval: eval every {} epochs)".format(eval_freq)

        if _should_log(epoch, epochs, print_freq):
            logger.write(
                f"[Joint][Epoch {epoch + 1}/{epochs}] "
                f"mr={float(cfg['dataset'].get('missing_rate', 0.0)):.2f} "
                f"train_loss={stats['loss']:.6f} "
                f"loss_cons={stats['loss_cons']:.6f} "
                f"loss_cluster={stats['loss_cluster']:.6f} "
                f"loss_balance={stats['loss_balance']:.6f} "
                f"loss_hg={stats['loss_hg']:.6f} "
                f"{_format_r3plus_stats(stats)} "
                f"{_format_view_stats(stats)} {_format_mask_stats(stats)}"
                f"{eval_text}"
            )

        if bool(cfg.get("logging", {}).get("save_last", True)) and (epoch + 1) == epochs:
            save_checkpoint({"model": model.state_dict(), "epoch": epoch + 1, "cfg": cfg}, cfg["output_dir"], "checkpoint_joint_last.pt")

    if best_metrics is None:
        best_metrics = trainer.evaluate_cluster(model, eval_loader, device)
        best_epoch = epochs

    logger.write(
        f"Finished joint training. Best Evaluation at Epoch {best_epoch}: "
        f"ACC={best_metrics['acc']:.6f} NMI={best_metrics['nmi']:.6f} "
        f"ARI={best_metrics['ari']:.6f} PUR={best_metrics['purity']:.6f}"
        f"{_format_subset_eval_metrics(best_metrics)}"
    )
    return best_metrics, best_epoch


def train_pipeline(cfg, model, train_loader, eval_loader, device, logger):
    run_pretrain(cfg, model, train_loader, eval_loader, device, logger)
    best_metrics, best_epoch = run_joint_train(cfg, model, train_loader, eval_loader, device, logger)
    logger.write(
        f"Training finished! Best Performance (Epoch {best_epoch}): "
        f"ACC={best_metrics['acc']:.6f} | NMI={best_metrics['nmi']:.6f} | "
        f"ARI={best_metrics['ari']:.6f} | PUR={best_metrics['purity']:.6f}"
        f"{_format_subset_eval_metrics(best_metrics)}"
    )
    return best_metrics, best_epoch
