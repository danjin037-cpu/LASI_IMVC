from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from ..models.losses import (
    cluster_balance_regularization,
    cross_view_contrastive_loss,
    prototype_clustering_loss,
    prototype_teacher_kl_loss,
    view_distribution_distillation_loss,
)
from ..utils import AverageMeter


def _hungarian_acc(labels: np.ndarray, preds: np.ndarray, n_clusters: int) -> float:
    labels = np.asarray(labels).astype(np.int64)
    preds = np.asarray(preds).astype(np.int64)
    n_true = max(int(labels.max()) + 1, n_clusters)
    n_pred = max(int(preds.max()) + 1, n_clusters)
    mat = np.zeros((n_pred, n_true), dtype=np.int64)
    for p, y in zip(preds, labels):
        if 0 <= int(p) < n_pred and 0 <= int(y) < n_true:
            mat[int(p), int(y)] += 1
    row, col = linear_sum_assignment(mat.max() - mat)
    return float(mat[row, col].sum()) / max(float(labels.shape[0]), 1.0)


def _purity_score(labels: np.ndarray, preds: np.ndarray, n_clusters: int) -> float:
    labels = np.asarray(labels).astype(np.int64)
    preds = np.asarray(preds).astype(np.int64)
    n_true = max(int(labels.max()) + 1, n_clusters)
    n_pred = max(int(preds.max()) + 1, n_clusters)
    mat = np.zeros((n_pred, n_true), dtype=np.int64)
    for p, y in zip(preds, labels):
        if 0 <= int(p) < n_pred and 0 <= int(y) < n_true:
            mat[int(p), int(y)] += 1
    return float(mat.max(axis=1).sum()) / max(float(labels.shape[0]), 1.0)


def _metrics_from_pred(labels: np.ndarray, preds: np.ndarray, n_clusters: int) -> Dict[str, float]:
    return {
        "acc": _hungarian_acc(labels, preds, n_clusters),
        "nmi": float(normalized_mutual_info_score(labels, preds)),
        "ari": float(adjusted_rand_score(labels, preds)),
        "purity": _purity_score(labels, preds, n_clusters),
    }


def _global_label_map(labels: np.ndarray, preds: np.ndarray, n_clusters: int) -> np.ndarray:
    labels = np.asarray(labels).astype(np.int64)
    preds = np.asarray(preds).astype(np.int64)
    n_true = max(int(labels.max()) + 1, n_clusters)
    n_pred = max(int(preds.max()) + 1, n_clusters)
    mat = np.zeros((n_pred, n_true), dtype=np.int64)
    for p, y in zip(preds, labels):
        if 0 <= int(p) < n_pred and 0 <= int(y) < n_true:
            mat[int(p), int(y)] += 1
    row, col = linear_sum_assignment(mat.max() - mat)
    mapper = np.full(n_pred, -1, dtype=np.int64)
    mapper[row] = col
    for r in range(n_pred):
        if mapper[r] < 0:
            mapper[r] = int(mat[r].argmax()) if mat[r].sum() > 0 else 0
    return mapper


def _subset_metrics_global_map(labels: np.ndarray, preds: np.ndarray, original_masks: np.ndarray, n_clusters: int) -> Dict[str, float]:
    original_masks = np.asarray(original_masks).astype(bool)
    if original_masks.ndim != 2:
        return {}
    obs_count = original_masks.sum(axis=1)
    num_original_views = original_masks.shape[1]
    mapper = _global_label_map(labels, preds, n_clusters)
    out: Dict[str, float] = {}
    subset_defs = {
        "complete": obs_count == num_original_views,
        "single": obs_count == 1,
        "incomplete": obs_count < num_original_views,
        "multi_incomplete": (obs_count >= 2) & (obs_count < num_original_views),
    }
    for name, mask in subset_defs.items():
        mask = np.asarray(mask).astype(bool)
        out[f"{name}_count"] = float(mask.sum())
        if mask.sum() == 0:
            out[f"{name}_acc"] = 0.0
            out[f"{name}_nmi"] = 0.0
            out[f"{name}_ari"] = 0.0
            out[f"{name}_purity"] = 0.0
            continue
        safe = np.clip(preds[mask], 0, len(mapper) - 1)
        mapped = mapper[safe]
        out[f"{name}_acc"] = float((mapped == labels[mask]).mean())
        out[f"{name}_nmi"] = float(normalized_mutual_info_score(labels[mask], preds[mask]))
        out[f"{name}_ari"] = float(adjusted_rand_score(labels[mask], preds[mask]))
        out[f"{name}_purity"] = _purity_score(labels[mask], preds[mask], n_clusters)
    return out


def _evaluate_kmeans_label_free(
    features: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    seed: int,
    protocol: str = "sklearn_ninit10",
    n_init: int = 10,
    repeats: int = 10,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Evaluate KMeans without selecting a run by label-based ACC."""
    protocol = str(protocol).lower()
    if protocol == "sklearn_ninit10":
        km = KMeans(n_clusters=n_clusters, n_init=int(n_init), random_state=int(seed))
        pred = km.fit_predict(features)
        metrics = _metrics_from_pred(labels, pred, n_clusters)
        metrics.update({
            "inertia": float(km.inertia_),
            "selected_repeat": -1.0,
            "kmeans_n_init": float(n_init),
        })
        return metrics, pred

    if protocol in {"single", "single_ninit1"}:
        km = KMeans(n_clusters=n_clusters, n_init=1, random_state=int(seed))
        pred = km.fit_predict(features)
        metrics = _metrics_from_pred(labels, pred, n_clusters)
        metrics.update({"inertia": float(km.inertia_), "selected_repeat": 0.0, "kmeans_n_init": 1.0})
        return metrics, pred

    if protocol in {"min_inertia", "inertia"}:
        candidates = []
        for r in range(max(1, int(repeats))):
            km = KMeans(n_clusters=n_clusters, n_init=1, random_state=int(seed) + r)
            pred = km.fit_predict(features)
            candidates.append((r, float(km.inertia_), pred))
        r, inertia, pred = min(candidates, key=lambda x: x[1])
        metrics = _metrics_from_pred(labels, pred, n_clusters)
        metrics.update({"inertia": float(inertia), "selected_repeat": float(r), "kmeans_n_init": 1.0})
        return metrics, pred

    raise ValueError(
        f"Unsupported eval_protocol={protocol}. Use sklearn_ninit10, min_inertia, or single. "
        "Do not use source_oracle for paper-facing training/evaluation."
    )


class JointTrainer:
    """
    Baseline-clean joint trainer + R3+ strict CLIP-guided prototype distillation.

    Base losses:
    - L_cons: paired cross-view contrastive loss, computed only on co-observed view pairs;
    - L_cluster: DEC-style prototype clustering loss;
    - L_balance: cluster-balance regularization;
    - L_hg: optional KL(q_global || p_star) consensus regularization.

    R3+ additions, all strict-IMVC safe:
    - L_view_distill: only on legal observed-complete samples. The fused q_global with CLIP
      acts as a detached teacher; each original view is trained to approach this teacher.
    - L_single_proto: only on real single-original-view samples. Teacher prototypes are built
      only from legal observed-complete samples, then used as cluster-level anchors.

    Important anti-leakage rule:
    - This trainer never constructs a teacher for a real single-view sample by restoring its
      missing view. CLIP can influence single-view samples only through parameters/prototypes
      learned from other legal observed-complete samples.
    """

    def __init__(self, cfg: Dict) -> None:
        self.cfg = cfg
        train_cfg = cfg["train"]
        loss_cfg = cfg["loss"]
        self.cluster_start_epoch = int(train_cfg.get("cluster_start_epoch", 10))
        self.cluster_on_reliable_samples = bool(loss_cfg.get("cluster_on_reliable_samples", False))
        self.cluster_min_observed_views = int(loss_cfg.get("cluster_min_observed_views", 2))
        self.balance_on_reliable_samples = bool(loss_cfg.get("balance_on_reliable_samples", False))

        # R3+ switches and state.
        self.lambda_view_distill = float(loss_cfg.get("lambda_view_distill", 0.0))
        self.lambda_single_proto = float(loss_cfg.get("lambda_single_proto", 0.0))
        self.distill_start_epoch = int(loss_cfg.get("r3plus_distill_start_epoch", 15))
        self.proto_start_epoch = int(loss_cfg.get("r3plus_proto_start_epoch", 25))
        self.teacher_conf_threshold = float(loss_cfg.get("r3plus_teacher_conf_threshold", 0.60))
        self.view_distill_student_temperature = float(loss_cfg.get("r3plus_view_student_temperature", 1.0))
        self.proto_teacher_temperature = float(loss_cfg.get("r3plus_proto_teacher_temperature", 0.2))
        self.proto_student_temperature = float(loss_cfg.get("r3plus_proto_student_temperature", 1.0))
        self.proto_conf_threshold = float(loss_cfg.get("r3plus_proto_conf_threshold", 0.45))
        self.proto_margin_threshold = float(loss_cfg.get("r3plus_proto_margin_threshold", 0.10))
        self.proto_bank_fallback_to_unfiltered = bool(loss_cfg.get("r3plus_proto_bank_fallback_to_unfiltered", False))
        self.proto_bank_rebuild_interval = int(loss_cfg.get("r3plus_proto_bank_rebuild_interval", 0))
        self.single_proto_require_no_clip = bool(loss_cfg.get("r3plus_single_proto_require_no_clip", False))
        self.use_clip_conflict_filter = bool(loss_cfg.get("r3plus_clip_conflict_filter", True))
        self.clip_conflict_mode = str(loss_cfg.get("r3plus_clip_conflict_mode", "pred"))
        self.clip_conflict_kl_threshold = float(loss_cfg.get("r3plus_clip_conflict_kl_threshold", 0.5))

        self.teacher_prototypes: torch.Tensor | None = None
        self.teacher_proto_epoch: int | None = None
        self.last_proto_bank_stats: Dict[str, float] = {
            "proto_bank_size": 0.0,
            "proto_bank_complete_count": 0.0,
            "proto_bank_filtered_count": 0.0,
            "proto_bank_filter_keep_ratio": 0.0,
            "proto_bank_conf_mean": 0.0,
            "proto_bank_conflict_keep_ratio": 0.0,
            "proto_bank_active": 0.0,
        }

    def get_stage_weights(self, epoch: int) -> Dict[str, float]:
        loss_cfg = self.cfg["loss"]
        if epoch < self.cluster_start_epoch:
            return {
                "lambda_cons": float(loss_cfg.get("lambda_cons", 0.5)),
                "lambda_cluster": 0.0,
                "lambda_balance": 0.0,
                "lambda_hg": 0.0,
            }
        return {
            "lambda_cons": float(loss_cfg.get("lambda_cons", 0.5)),
            "lambda_cluster": float(loss_cfg.get("lambda_cluster", 1.0)),
            "lambda_balance": float(loss_cfg.get("lambda_balance", 0.05)),
            "lambda_hg": float(loss_cfg.get("lambda_hg", 1.0)),
        }

    def _original_num_views(self, num_views: int) -> int:
        clip_cfg = self.cfg.get("clip_view", {}) or {}
        clip_as_view = bool(clip_cfg.get("enabled", False)) and bool(clip_cfg.get("as_view", False))
        if not clip_as_view:
            return num_views
        # If not explicitly configured, CLIP is assumed to be appended as the last view.
        return int(clip_cfg.get("original_num_views", max(num_views - 1, 1)))

    def _clip_view_index(self, num_views: int) -> int | None:
        clip_cfg = self.cfg.get("clip_view", {}) or {}
        clip_as_view = bool(clip_cfg.get("enabled", False)) and bool(clip_cfg.get("as_view", False))
        if not clip_as_view or num_views <= 1:
            return None
        idx = int(clip_cfg.get("view_index", num_views - 1))
        if idx < 0 or idx >= num_views:
            return None
        return idx

    def _original_view_indices(self, num_views: int) -> List[int]:
        original_num_views = self._original_num_views(num_views)
        return list(range(min(original_num_views, num_views)))

    def _complete_teacher_mask(self, observed_mask: torch.Tensor) -> torch.Tensor:
        """
        Legal teacher samples for R3+.

        Default strict rule: all currently available model views must be observed. With the
        current CLIP config mask_policy=all_sources_observed, this means [view0, view1, CLIP]
        are all observed. No real single-view sample is ever converted back to full view.
        """
        return observed_mask.bool().all(dim=1)

    def _single_original_view_mask(self, observed_mask: torch.Tensor) -> torch.Tensor:
        """Real single-view samples under original views, not counting derived CLIP."""
        mask = observed_mask.bool()
        num_views = mask.shape[1]
        original_indices = self._original_view_indices(num_views)
        if len(original_indices) == 0:
            return torch.zeros(mask.shape[0], dtype=torch.bool, device=mask.device)
        orig_count = mask[:, original_indices].sum(dim=1)
        single_mask = orig_count == 1
        if self.single_proto_require_no_clip:
            clip_idx = self._clip_view_index(num_views)
            if clip_idx is not None:
                single_mask = single_mask & (~mask[:, clip_idx])
        return single_mask

    @staticmethod
    def _collect_mask_stats(observed_mask: torch.Tensor) -> Dict[str, float]:
        mask = observed_mask.bool()
        obs_count = mask.sum(dim=1).float()
        batch_size, num_views = mask.shape
        stats = {
            "avg_observed_views_per_sample": float(obs_count.mean().item()),
            "single_view_sample_ratio": float((obs_count == 1).float().mean().item()),
            "two_or_more_view_sample_ratio": float((obs_count >= 2).float().mean().item()),
            "complete_sample_ratio": float((obs_count == num_views).float().mean().item()),
        }
        if num_views >= 2:
            pair_count = 0.0
            for i in range(num_views):
                for j in range(i + 1, num_views):
                    pair_count += float((mask[:, i] & mask[:, j]).float().sum().item())
            stats["valid_pair_ratio"] = pair_count / max(float(batch_size * num_views * (num_views - 1) / 2), 1.0)
        else:
            stats["valid_pair_ratio"] = 0.0
        return stats

    def _collect_original_mask_stats(self, observed_mask: torch.Tensor) -> Dict[str, float]:
        """
        Collect original-view mask stats and strict CLIP source-violation count.

        The source-violation count is always returned. If no source indices are
        configured, all original views are treated as the legal CLIP sources.
        """
        mask = observed_mask.bool()
        num_views = mask.shape[1]
        original_indices = self._original_view_indices(num_views)

        stats = {
            "clip_source_violation_count": 0.0,
        }

        if len(original_indices) > 0:
            orig_count = mask[:, original_indices].sum(dim=1).float()
            stats.update({
                "original_avg_observed_views": float(orig_count.mean().item()),
                "original_single_view_ratio": float((orig_count == 1).float().mean().item()),
                "original_complete_ratio": float((orig_count == len(original_indices)).float().mean().item()),
            })

        clip_idx = self._clip_view_index(num_views)
        if clip_idx is None:
            return stats

        stats["clip_observed_ratio"] = float(mask[:, clip_idx].float().mean().item())

        clip_cfg = self.cfg.get("clip_view", {}) or {}
        configured_sources = clip_cfg.get("source_view_indices", None)

        if configured_sources is None or len(configured_sources) == 0:
            # Default strict rule: CLIP is legal only when all original source
            # views are observed. This matches mask_policy=all_sources_observed.
            source_indices = original_indices
        else:
            source_indices = [int(i) for i in configured_sources]

        valid_sources = [
            int(i)
            for i in source_indices
            if 0 <= int(i) < num_views and int(i) != clip_idx
        ]

        if len(valid_sources) == 0:
            # Keep the key present. No valid source configuration means we cannot
            # identify a source-side violation here, so report zero instead of NA.
            return stats

        source_all = mask[:, valid_sources].all(dim=1)
        violation = mask[:, clip_idx] & (~source_all)
        stats["clip_source_violation_count"] = float(violation.sum().item())
        return stats

    @staticmethod
    def _collect_view_stats(out: Dict, observed_mask: torch.Tensor) -> Dict[str, float]:
        stats = {}
        for v in range(out["fusion_weights"].shape[1]):
            stats[f"mean_w_view{v}"] = float(out["fusion_weights"][:, v].mean().item())
            stats[f"mean_u_view{v}"] = float(out["view_uncertainty"][:, v, 0].mean().item())
            stats[f"mean_H_view{v}"] = float(out["view_entropy"][:, v, 0].mean().item())
            stats[f"mean_agree_view{v}"] = float(out["view_agreement"][:, v].mean().item())
            stats[f"mean_comb_rel_view{v}"] = float(out["view_combined_reliability"][:, v].mean().item())
            stats[f"mean_obs_view{v}"] = float(observed_mask[:, v].float().mean().item())
        return stats

    def _reliable_mask(self, observed_mask: torch.Tensor) -> torch.Tensor:
        return observed_mask.bool().sum(dim=1) >= self.cluster_min_observed_views

    def _cluster_loss(self, q_global: torch.Tensor, observed_mask: torch.Tensor, device) -> torch.Tensor:
        if not self.cluster_on_reliable_samples:
            return prototype_clustering_loss(q_global)
        mask = self._reliable_mask(observed_mask)
        if int(mask.sum().item()) <= 1:
            return torch.tensor(0.0, device=device)
        return prototype_clustering_loss(q_global[mask])

    def _balance_loss(self, q_global: torch.Tensor, observed_mask: torch.Tensor, device) -> torch.Tensor:
        if not self.balance_on_reliable_samples:
            return cluster_balance_regularization(q_global)
        mask = self._reliable_mask(observed_mask)
        if int(mask.sum().item()) <= 1:
            return torch.tensor(0.0, device=device)
        return cluster_balance_regularization(q_global[mask])

    def _original_fused_logits_for_conflict_filter(self, model, out: Dict, observed_mask: torch.Tensor) -> torch.Tensor | None:
        num_views = observed_mask.shape[1]
        original_indices = self._original_view_indices(num_views)
        if len(original_indices) == 0:
            return None
        original_features = torch.stack([out["shared_dict"][f"view_{v}"] for v in original_indices], dim=1)
        original_mask = observed_mask[:, original_indices].float().unsqueeze(-1)
        h_orig = (original_features * original_mask).sum(dim=1) / original_mask.sum(dim=1).clamp_min(1.0)
        return model.cluster_head(h_orig)["q_global"]

    def _clip_conflict_keep_mask(self, model, out: Dict, observed_mask: torch.Tensor, complete_mask: torch.Tensor) -> torch.Tensor:
        """Keep complete teacher samples where CLIP and original-view prediction are not conflicting."""
        idx = torch.nonzero(complete_mask.bool(), as_tuple=False).squeeze(1)
        if idx.numel() == 0 or (not self.use_clip_conflict_filter):
            return torch.ones(idx.numel(), dtype=torch.bool, device=observed_mask.device)

        num_views = observed_mask.shape[1]
        clip_idx = self._clip_view_index(num_views)
        if clip_idx is None:
            return torch.ones(idx.numel(), dtype=torch.bool, device=observed_mask.device)

        q_orig = self._original_fused_logits_for_conflict_filter(model, out, observed_mask)
        if q_orig is None:
            return torch.ones(idx.numel(), dtype=torch.bool, device=observed_mask.device)

        q_clip = model.cluster_head(out["shared_dict"][f"view_{clip_idx}"])["q_global"]
        q_orig_i = q_orig[idx].detach().clamp_min(1e-8)
        q_clip_i = q_clip[idx].detach().clamp_min(1e-8)

        if self.clip_conflict_mode == "kl":
            kl = (q_clip_i * (q_clip_i.log() - q_orig_i.log())).sum(dim=-1)
            return kl <= self.clip_conflict_kl_threshold
        if self.clip_conflict_mode == "pred_or_kl":
            same_pred = q_clip_i.argmax(dim=-1) == q_orig_i.argmax(dim=-1)
            kl = (q_clip_i * (q_clip_i.log() - q_orig_i.log())).sum(dim=-1)
            return same_pred | (kl <= self.clip_conflict_kl_threshold)
        # Default: strict but stable. Keep samples where CLIP and original views predict the same cluster.
        return q_clip_i.argmax(dim=-1) == q_orig_i.argmax(dim=-1)

    @torch.no_grad()
    def _build_observed_complete_proto_bank(self, model, loader, device, epoch: int) -> None:
        """
        Build CLIP-enhanced prototype bank only from legal observed-complete samples.

        No single-view sample contributes missing-view features to this bank. If CLIP is configured
        as all_sources_observed, complete_mask means [view0, view1, clip] are all available.
        """
        from sklearn.cluster import KMeans

        n_classes = int(self.cfg["dataset"]["n_classes"])
        model.eval()
        selected_features = []
        unfiltered_features = []
        total_complete = 0
        total_filtered = 0
        conf_values = []
        conflict_keep_values = []

        for batch in loader:
            views = [x.to(device, non_blocking=True) for x in batch["views"]]
            observed_mask = batch["observed_mask"].to(device, non_blocking=True)
            out = model({**batch, "views": views, "observed_mask": observed_mask}, stage="joint")

            complete_mask = self._complete_teacher_mask(observed_mask)
            idx = torch.nonzero(complete_mask.bool(), as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue

            q_teacher = out["q_global"].detach()
            conf = q_teacher[idx].max(dim=-1).values
            conf_keep = conf >= self.teacher_conf_threshold
            conflict_keep = self._clip_conflict_keep_mask(model, out, observed_mask, complete_mask)
            keep = conf_keep & conflict_keep

            h_complete = F.normalize(out["h_fused"][idx].detach(), dim=-1)
            unfiltered_features.append(h_complete.cpu().numpy())
            if int(keep.sum().item()) > 0:
                selected_features.append(h_complete[keep].cpu().numpy())

            total_complete += int(idx.numel())
            total_filtered += int(keep.sum().item())
            conf_values.append(conf.cpu().numpy())
            conflict_keep_values.append(conflict_keep.float().cpu().numpy())

        model.train()

        conf_mean = float(np.concatenate(conf_values).mean()) if len(conf_values) > 0 else 0.0
        conflict_keep_ratio = float(np.concatenate(conflict_keep_values).mean()) if len(conflict_keep_values) > 0 else 0.0
        keep_ratio = float(total_filtered / max(total_complete, 1))

        features_for_kmeans = None
        if len(selected_features) > 0:
            selected = np.concatenate(selected_features, axis=0)
            if selected.shape[0] >= n_classes:
                features_for_kmeans = selected

        if features_for_kmeans is None and self.proto_bank_fallback_to_unfiltered and len(unfiltered_features) > 0:
            unfiltered = np.concatenate(unfiltered_features, axis=0)
            if unfiltered.shape[0] >= n_classes:
                features_for_kmeans = unfiltered

        self.last_proto_bank_stats = {
            "proto_bank_size": float(0 if features_for_kmeans is None else features_for_kmeans.shape[0]),
            "proto_bank_complete_count": float(total_complete),
            "proto_bank_filtered_count": float(total_filtered),
            "proto_bank_filter_keep_ratio": float(keep_ratio),
            "proto_bank_conf_mean": float(conf_mean),
            "proto_bank_conflict_keep_ratio": float(conflict_keep_ratio),
            "proto_bank_active": 0.0,
        }

        if features_for_kmeans is None:
            self.teacher_prototypes = None
            self.teacher_proto_epoch = epoch
            return

        km = KMeans(n_clusters=n_classes, n_init=20, random_state=int(self.cfg.get("seed", 0)))
        km.fit(features_for_kmeans)
        centers = torch.as_tensor(km.cluster_centers_, dtype=torch.float32, device=device)
        self.teacher_prototypes = F.normalize(centers, dim=-1)
        self.teacher_proto_epoch = epoch
        self.last_proto_bank_stats["proto_bank_active"] = 1.0

    def _maybe_build_proto_bank(self, model, loader, device, epoch: int) -> None:
        if self.lambda_single_proto <= 0 or epoch < self.proto_start_epoch:
            return
        should_build = self.teacher_prototypes is None
        if (
            self.proto_bank_rebuild_interval > 0
            and epoch >= self.proto_start_epoch
            and (epoch - self.proto_start_epoch) % self.proto_bank_rebuild_interval == 0
            and self.teacher_proto_epoch != epoch
        ):
            should_build = True
        if should_build:
            self._build_observed_complete_proto_bank(model, loader, device, epoch)

    def _compute_view_distill_loss(self, model, out: Dict, observed_mask: torch.Tensor, epoch: int, device) -> Tuple[torch.Tensor, Dict[str, float]]:
        if self.lambda_view_distill <= 0 or epoch < self.distill_start_epoch:
            return torch.tensor(0.0, device=device), {"view_distill_count": 0.0, "view_distill_terms": 0.0}

        complete_mask = self._complete_teacher_mask(observed_mask)
        original_indices = self._original_view_indices(observed_mask.shape[1])
        student_logits_list = []
        for v in original_indices:
            # Each original view is projected by the shared clustering head. This trains the
            # original-view encoder to enter the CLIP-enhanced fused semantic space.
            student_logits_list.append(model.cluster_head(out["shared_dict"][f"view_{v}"])["logits"])

        return view_distribution_distillation_loss(
            q_teacher=out["q_global"],
            student_logits_list=student_logits_list,
            sample_mask=complete_mask,
            student_temperature=self.view_distill_student_temperature,
        )

    def _compute_single_proto_loss(self, out: Dict, observed_mask: torch.Tensor, epoch: int, device) -> Tuple[torch.Tensor, Dict[str, float]]:
        if self.lambda_single_proto <= 0 or epoch < self.proto_start_epoch:
            return torch.tensor(0.0, device=device), {
                "proto_count": 0.0,
                "proto_keep_ratio": 0.0,
                "proto_conf_mean": 0.0,
                "proto_margin_mean": 0.0,
            }

        single_mask = self._single_original_view_mask(observed_mask)
        return prototype_teacher_kl_loss(
            h_fused=out["h_fused"],
            student_logits=out["logits_global"],
            teacher_prototypes=self.teacher_prototypes,
            sample_mask=single_mask,
            teacher_temperature=self.proto_teacher_temperature,
            student_temperature=self.proto_student_temperature,
            confidence_threshold=self.proto_conf_threshold,
            margin_threshold=self.proto_margin_threshold,
        )

    def train_one_epoch(self, model, loader, optimizer, device, epoch: int) -> Dict[str, float]:
        model.train()
        weights = self.get_stage_weights(epoch)
        self._maybe_build_proto_bank(model, loader, device, epoch)

        meters = {name: AverageMeter() for name in [
            "loss", "loss_cons", "loss_cluster", "loss_balance", "loss_hg",
            "loss_view_distill", "loss_single_proto",
        ]}
        stat_meters: Dict[str, AverageMeter] = {}
        clip_source_violation_total = 0.0

        for batch in loader:
            views = [x.to(device, non_blocking=True) for x in batch["views"]]
            observed_mask = batch["observed_mask"].to(device, non_blocking=True)
            out = model({**batch, "views": views, "observed_mask": observed_mask}, stage="joint")

            loss_cons = torch.tensor(0.0, device=device)
            valid_pair_count = 0
            if weights["lambda_cons"] > 0:
                for i in range(len(views)):
                    for j in range(i + 1, len(views)):
                        pair_mask = observed_mask[:, i] & observed_mask[:, j]
                        if int(pair_mask.sum().item()) > 1:
                            q = out["q_global"][pair_mask].detach()
                            loss_cons = loss_cons + cross_view_contrastive_loss(
                                out["shared_dict"][f"view_{i}"][pair_mask],
                                out["shared_dict"][f"view_{j}"][pair_mask],
                                q1=q,
                                q2=q,
                            )
                            valid_pair_count += 1
                if valid_pair_count > 0:
                    loss_cons = loss_cons / valid_pair_count

            loss_cluster = self._cluster_loss(out["q_global"], observed_mask, device)
            loss_balance = self._balance_loss(out["q_global"], observed_mask, device)

            loss_hg = torch.tensor(0.0, device=device)
            if weights["lambda_hg"] > 0:
                loss_hg = F.kl_div(out["q_global"].clamp_min(1e-8).log(), out["p_star"], reduction="batchmean")

            loss_view_distill, view_distill_stats = self._compute_view_distill_loss(
                model=model,
                out=out,
                observed_mask=observed_mask,
                epoch=epoch,
                device=device,
            )
            loss_single_proto, proto_stats = self._compute_single_proto_loss(
                out=out,
                observed_mask=observed_mask,
                epoch=epoch,
                device=device,
            )

            total_loss = (
                weights["lambda_cons"] * loss_cons
                + weights["lambda_cluster"] * loss_cluster
                + weights["lambda_balance"] * loss_balance
                + weights["lambda_hg"] * loss_hg
                + self.lambda_view_distill * loss_view_distill
                + self.lambda_single_proto * loss_single_proto
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            batch_size = views[0].size(0)
            values = {
                "loss": total_loss.item(),
                "loss_cons": loss_cons.item(),
                "loss_cluster": loss_cluster.item(),
                "loss_balance": loss_balance.item(),
                "loss_hg": loss_hg.item(),
                "loss_view_distill": loss_view_distill.item(),
                "loss_single_proto": loss_single_proto.item(),
            }
            for k, v in values.items():
                meters[k].update(float(v), n=batch_size)

            batch_stats = {}
            batch_stats.update(self._collect_view_stats(out, observed_mask))
            batch_stats.update(self._collect_mask_stats(observed_mask))
            batch_stats.update(self._collect_original_mask_stats(observed_mask))
            batch_stats.update(view_distill_stats)
            batch_stats.update(proto_stats)
            batch_stats.update(self.last_proto_bank_stats)
            batch_stats["cluster_reliable_sample_ratio"] = float(self._reliable_mask(observed_mask).float().mean().item())
            batch_stats["r3plus_lambda_view_distill"] = float(self.lambda_view_distill)
            batch_stats["r3plus_lambda_single_proto"] = float(self.lambda_single_proto)
            batch_stats["r3plus_teacher_complete_ratio"] = float(self._complete_teacher_mask(observed_mask).float().mean().item())
            batch_stats["r3plus_single_original_ratio"] = float(self._single_original_view_mask(observed_mask).float().mean().item())

            # Count should be an epoch-level sum, not a batch-size-weighted average.
            clip_source_violation_total += float(batch_stats.get("clip_source_violation_count", 0.0))

            for k, v in batch_stats.items():
                if k == "clip_source_violation_count":
                    continue
                stat_meters.setdefault(k, AverageMeter()).update(float(v), n=batch_size)

        stats = {k: meter.avg for k, meter in meters.items()}
        stats.update({
            "lambda_cons": weights["lambda_cons"],
            "lambda_cluster": weights["lambda_cluster"],
            "lambda_balance": weights["lambda_balance"],
            "lambda_hg": weights["lambda_hg"],
            "lambda_view_distill": self.lambda_view_distill,
            "lambda_single_proto": self.lambda_single_proto,
        })
        for k, meter in stat_meters.items():
            stats[k] = meter.avg
        stats["clip_source_violation_count"] = float(clip_source_violation_total)
        return stats

    @torch.no_grad()
    def evaluate_cluster(self, model, loader, device) -> Dict[str, float]:
        """
        Evaluate learned fused representations with a label-free KMeans protocol.

        Default protocol is sklearn_ninit10:
        KMeans(n_clusters=K, n_init=10, random_state=train_seed).
        The final KMeans assignment is selected by the internal inertia objective, not by ACC.
        ACC/NMI/ARI/PUR are computed only after the clustering assignment is fixed.
        """
        model.eval()
        features, labels, original_masks = [], [], []
        for batch in loader:
            views = [x.to(device, non_blocking=True) for x in batch["views"]]
            observed_mask = batch["observed_mask"].to(device, non_blocking=True)
            out = model({**batch, "views": views, "observed_mask": observed_mask}, stage="joint")
            features.append(F.normalize(out["h_fused"], dim=-1).cpu().numpy())
            labels.append(batch["label"].cpu().numpy())
            original_masks.append(batch.get("original_observed_mask", batch["observed_mask"]).cpu().numpy().astype(bool))

        feats = np.concatenate(features, axis=0)
        ys = np.concatenate(labels, axis=0)
        masks = np.concatenate(original_masks, axis=0).astype(bool)
        eval_cfg = self.cfg.get("eval", {}) or {}
        protocol = str(eval_cfg.get("eval_protocol", "sklearn_ninit10"))
        n_init = int(eval_cfg.get("kmeans_n_init", 10))
        repeats = int(eval_cfg.get("kmeans_repeats", 10))
        seed = int(self.cfg.get("train_seed", self.cfg.get("seed", 0)))
        n_clusters = int(self.cfg["dataset"]["n_classes"])

        metrics, pred = _evaluate_kmeans_label_free(
            features=feats,
            labels=ys,
            n_clusters=n_clusters,
            seed=seed,
            protocol=protocol,
            n_init=n_init,
            repeats=repeats,
        )
        metrics.update(_subset_metrics_global_map(ys, pred, masks, n_clusters))
        metrics["eval_protocol"] = protocol
        return metrics
