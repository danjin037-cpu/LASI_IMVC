from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .clustering import PrototypeClusteringHead
from .decoders import build_view_decoders
from .encoders import build_view_encoders
from .evidence_heads import build_view_evidence_heads
from .uncertainty_fusion import UncertaintyAwareFusionRouter


class LASIIMVC(nn.Module):
    """LASI-IMVC with strict observed-view encoding and reliability fusion."""

    def __init__(self, cfg: dict[str, Any], input_dims: list[int]) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_views = len(input_dims)
        self.shared_dim = int(cfg["model"]["shared_dim"])
        self.num_clusters = int(cfg["dataset"]["num_classes"])
        self.encoders = build_view_encoders(cfg, input_dims)
        self.decoders = build_view_decoders(cfg, input_dims)
        self.cluster_head = PrototypeClusteringHead(
            feature_dim=self.shared_dim,
            num_clusters=self.num_clusters,
            temperature=float(cfg["model"]["cluster_temperature"]),
            use_sinkhorn=bool(cfg["model"]["use_sinkhorn"]),
        )
        self.view_evidence_heads = build_view_evidence_heads(
            cfg, self.num_views, self.shared_dim, self.num_clusters
        )
        self.fusion = UncertaintyAwareFusionRouter(
            feature_dim=self.shared_dim,
            num_clusters=self.num_clusters,
            router_hidden_dim=int(cfg["model"]["fusion_router_hidden_dim"]),
            temperature=float(cfg["model"]["fusion_router_temperature"]),
            beta_self_uncertainty=float(cfg["model"]["beta_self_uncertainty"]),
            beta_self_entropy=float(cfg["model"]["beta_self_entropy"]),
            lambda_self=float(cfg["model"]["lambda_self_reliability"]),
            lambda_agreement=float(cfg["model"]["lambda_agreement"]),
            discount_rho=float(cfg["model"]["discount_rho"]),
        )

    def encode_views(
        self, views: list[torch.Tensor], observed_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if len(views) != self.num_views or observed_mask.shape[1] != self.num_views:
            raise ValueError("View count and observed mask do not match the model")
        batch_size = observed_mask.shape[0]
        shared: dict[str, torch.Tensor] = {}
        for view_index, view in enumerate(views):
            latent = torch.zeros(batch_size, self.shared_dim, device=view.device)
            observed = torch.nonzero(observed_mask[:, view_index].bool(), as_tuple=False).squeeze(1)
            if observed.numel():
                encoded = self.encoders[f"view_{view_index}"](view.index_select(0, observed))["s"]
                latent.index_copy_(0, observed, encoded)
            shared[f"view_{view_index}"] = latent
        return shared

    def reconstruct_views(self, shared: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            f"view_{v}": self.decoders[f"view_{v}"](shared[f"view_{v}"])
            for v in range(self.num_views)
        }

    def _stack(self, shared: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack([shared[f"view_{v}"] for v in range(self.num_views)], dim=1)

    @staticmethod
    def _masked_mean(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        expanded = mask.float().unsqueeze(-1)
        return (features * expanded).sum(dim=1) / expanded.sum(dim=1).clamp_min(1.0)

    def _view_statistics(self, shared: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = [self.view_evidence_heads[f"view_{v}"](shared[f"view_{v}"]) for v in range(self.num_views)]
        return {
            key: torch.stack([output[key] for output in outputs], dim=1)
            for key in ("prob", "uncertainty", "entropy")
        }

    def _consensus(
        self, view_prob: torch.Tensor, features: torch.Tensor, observed_mask: torch.Tensor
    ) -> torch.Tensor:
        masked_prob = view_prob.detach().masked_fill(~observed_mask.bool().unsqueeze(-1), -1.0)
        with torch.no_grad():
            mean_q = self.cluster_head(self._masked_mean(features, observed_mask))["q_global"]
        target = torch.cat([masked_prob, mean_q.detach().unsqueeze(1)], dim=1).max(dim=1)[0]
        target = target.clamp_min(1e-8).pow(2)
        return (target / target.sum(dim=-1, keepdim=True)).detach()

    @torch.no_grad()
    def init_prototypes_from_loader(self, loader, device: torch.device) -> None:
        from sklearn.cluster import KMeans

        self.eval()
        chunks = []
        for batch in loader:
            views = [view.to(device, non_blocking=True) for view in batch["views"]]
            mask = batch["observed_mask"].to(device, non_blocking=True)
            features = self._stack(self.encode_views(views, mask))
            chunks.append(F.normalize(self._masked_mean(features, mask), dim=-1).cpu().numpy())
        array = np.concatenate(chunks, axis=0)
        kmeans = KMeans(n_clusters=self.num_clusters, n_init=20, random_state=int(self.cfg["seed"]))
        kmeans.fit(array)
        centers = torch.as_tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
        self.cluster_head.init_from_centers(centers)

    def forward_pretrain(self, batch: dict[str, Any]) -> dict[str, Any]:
        shared = self.encode_views(batch["views"], batch["observed_mask"])
        return {"shared_dict": shared, "recon_dict": self.reconstruct_views(shared)}

    def forward_joint(self, batch: dict[str, Any]) -> dict[str, Any]:
        observed_mask = batch["observed_mask"]
        shared = self.encode_views(batch["views"], observed_mask)
        stacked = self._stack(shared)
        stats = self._view_statistics(shared)
        consensus = self._consensus(stats["prob"], stacked, observed_mask)
        fusion = self.fusion(
            features=stacked,
            probs=stats["prob"],
            uncertainties=stats["uncertainty"],
            entropies=stats["entropy"],
            base_reliability=observed_mask.float(),
            valid_mask=observed_mask,
            target_consensus=consensus,
        )
        clustering = self.cluster_head(fusion["fused"])
        return {
            "shared_dict": shared,
            "h_fused": fusion["fused"],
            "q_global": clustering["q_global"],
            "logits_global": clustering["logits"],
            "prototypes": clustering["prototypes"],
            "fusion_valid_mask": observed_mask,
            "available_mask": observed_mask,
            "view_prob": stats["prob"],
            "view_uncertainty": stats["uncertainty"],
            "view_entropy": stats["entropy"],
            "view_agreement": fusion["agreement"],
            "view_combined_reliability": fusion["combined_reliability"],
            "fusion_weights": fusion["weights"],
            "p_star": consensus,
        }

    def forward(self, batch: dict[str, Any], stage: str = "joint") -> dict[str, Any]:
        return self.forward_pretrain(batch) if stage == "pretrain" else self.forward_joint(batch)
