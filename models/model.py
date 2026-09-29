from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float, norm: str) -> None:
        super().__init__()
        if norm == "bn":
            norm_layer = nn.BatchNorm1d(out_dim)
        elif norm == "ln":
            norm_layer = nn.LayerNorm(out_dim)
        else:
            norm_layer = nn.Identity()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            norm_layer,
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ViewEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        shared_dim: int,
        dropout: float,
        norm: str,
    ) -> None:
        super().__init__()
        dims = [in_dim, *hidden_dims]
        self.trunk = nn.Sequential(
            *[
                MLPBlock(dims[i - 1], dims[i], dropout=dropout, norm=norm)
                for i in range(1, len(dims))
            ]
        )
        self.shared_head = nn.Linear(dims[-1], shared_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(x)
        return {"h": hidden, "s": self.shared_head(hidden)}


class ViewDecoder(nn.Module):
    def __init__(
        self,
        shared_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        dims = [shared_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for i in range(1, len(dims)):
            layers.extend(
                [
                    nn.Linear(dims[i - 1], dims[i]),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                ]
            )
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class EvidenceHead(nn.Module):
    """Map a shared latent vector to evidence and uncertainty statistics."""

    def __init__(
        self,
        feature_dim: int,
        num_clusters: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_clusters = num_clusters
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_clusters),
        )

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        evidence = F.softplus(self.net(z))
        alpha = evidence + 1.0
        concentration = alpha.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        probability = alpha / concentration
        uncertainty = float(self.num_clusters) / concentration
        entropy = -(probability * probability.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
        entropy = entropy / math.log(self.num_clusters)
        return {
            "evidence": evidence,
            "alpha": alpha,
            "prob": probability,
            "uncertainty": uncertainty,
            "entropy": entropy,
        }


class PrototypeClusteringHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_clusters: int,
        temperature: float,
        use_sinkhorn: bool,
    ) -> None:
        super().__init__()
        self.num_clusters = num_clusters
        self.temperature = temperature
        self.use_sinkhorn = use_sinkhorn
        self.prototypes = nn.Parameter(torch.empty(num_clusters, feature_dim))
        nn.init.xavier_uniform_(self.prototypes)

    @staticmethod
    def _sinkhorn(logits: torch.Tensor, iterations: int = 3) -> torch.Tensor:
        assignment = torch.exp(logits / 0.05).t()
        assignment /= assignment.sum()
        num_clusters, batch_size = assignment.shape
        for _ in range(iterations):
            assignment /= assignment.sum(dim=1, keepdim=True)
            assignment /= num_clusters
            assignment /= assignment.sum(dim=0, keepdim=True)
            assignment /= batch_size
        return (assignment * batch_size).t()

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        features = F.normalize(z, dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        logits = features @ prototypes.t() / self.temperature
        assignments = self._sinkhorn(logits) if self.use_sinkhorn else torch.softmax(logits, dim=-1)
        return {
            "q_global": assignments,
            "logits": logits,
            "prototypes": self.prototypes,
        }

    @torch.no_grad()
    def init_from_centers(self, centers: torch.Tensor) -> None:
        if centers.shape != self.prototypes.shape:
            raise ValueError(f"Center shape mismatch: {centers.shape} vs {self.prototypes.shape}")
        self.prototypes.copy_(centers)


class UncertaintyAwareFusionRouter(nn.Module):
    """Learn sample-wise view weights from uncertainty and consensus agreement."""

    def __init__(
        self,
        feature_dim: int,
        num_clusters: int,
        router_hidden_dim: int,
        temperature: float,
        beta_self_uncertainty: float,
        beta_self_entropy: float,
        lambda_self: float,
        lambda_agreement: float,
        discount_rho: float,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.beta_self_uncertainty = beta_self_uncertainty
        self.beta_self_entropy = beta_self_entropy
        self.lambda_self = lambda_self
        self.lambda_agreement = lambda_agreement
        self.discount_rho = discount_rho
        self.router = nn.Sequential(
            nn.Linear(feature_dim + num_clusters + 6, router_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(router_hidden_dim, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        probs: torch.Tensor,
        uncertainties: torch.Tensor,
        entropies: torch.Tensor,
        base_reliability: torch.Tensor,
        valid_mask: torch.Tensor,
        target_consensus: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mask = valid_mask.float().unsqueeze(-1)
        base_reliability = base_reliability.unsqueeze(-1)
        consensus = target_consensus.detach().unsqueeze(1)
        kl_divergence = (
            consensus * (consensus.clamp_min(1e-8).log() - probs.clamp_min(1e-8).log())
        ).sum(dim=-1, keepdim=True)
        agreement = torch.exp(-kl_divergence).clamp(min=1e-8, max=1.0)
        self_reliability = (
            self.beta_self_uncertainty * (1.0 - uncertainties)
            + self.beta_self_entropy * (1.0 - entropies)
        )
        raw_reliability = self.lambda_self * self_reliability + self.lambda_agreement * agreement
        discount = agreement.pow(max(float(self.discount_rho), 0.0))
        combined_reliability = raw_reliability * base_reliability * discount

        router_input = torch.cat(
            [
                F.normalize(features, dim=-1),
                probs,
                uncertainties,
                entropies,
                agreement,
                raw_reliability,
                base_reliability,
                combined_reliability,
            ],
            dim=-1,
        )
        logits = self.router(router_input) + combined_reliability.clamp_min(1e-8).log()
        logits = logits.masked_fill(~valid_mask.unsqueeze(-1), -1e4)
        weights = torch.softmax(logits / self.temperature, dim=1)
        return {
            "fused": (weights * features * mask).sum(dim=1),
            "weights": weights.squeeze(-1),
            "agreement": agreement.squeeze(-1),
            "combined_reliability": combined_reliability.squeeze(-1),
        }


class LASIIMVC(nn.Module):
    """LASI-IMVC with strict observed-view encoding and reliability fusion."""

    def __init__(self, cfg: dict[str, Any], input_dims: list[int]) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_views = len(input_dims)
        model_cfg = cfg["model"]
        self.shared_dim = int(model_cfg["shared_dim"])
        self.num_clusters = int(cfg["dataset"]["num_classes"])
        dropout = float(model_cfg["dropout"])

        self.encoders = nn.ModuleDict(
            {
                f"view_{index}": ViewEncoder(
                    in_dim=input_dim,
                    hidden_dims=model_cfg["encoder_hidden_dims"],
                    shared_dim=self.shared_dim,
                    dropout=dropout,
                    norm=model_cfg["norm"],
                )
                for index, input_dim in enumerate(input_dims)
            }
        )
        self.decoders = nn.ModuleDict(
            {
                f"view_{index}": ViewDecoder(
                    shared_dim=self.shared_dim,
                    hidden_dims=model_cfg["decoder_hidden_dims"],
                    out_dim=input_dim,
                    dropout=dropout,
                )
                for index, input_dim in enumerate(input_dims)
            }
        )
        self.cluster_head = PrototypeClusteringHead(
            feature_dim=self.shared_dim,
            num_clusters=self.num_clusters,
            temperature=float(model_cfg["cluster_temperature"]),
            use_sinkhorn=bool(model_cfg["use_sinkhorn"]),
        )
        self.view_evidence_heads = nn.ModuleDict(
            {
                f"view_{index}": EvidenceHead(
                    feature_dim=self.shared_dim,
                    num_clusters=self.num_clusters,
                    hidden_dim=int(model_cfg["evidence_hidden_dim"]),
                    dropout=float(model_cfg["evidence_dropout"]),
                )
                for index in range(self.num_views)
            }
        )
        self.fusion = UncertaintyAwareFusionRouter(
            feature_dim=self.shared_dim,
            num_clusters=self.num_clusters,
            router_hidden_dim=int(model_cfg["fusion_router_hidden_dim"]),
            temperature=float(model_cfg["fusion_router_temperature"]),
            beta_self_uncertainty=float(model_cfg["beta_self_uncertainty"]),
            beta_self_entropy=float(model_cfg["beta_self_entropy"]),
            lambda_self=float(model_cfg["lambda_self_reliability"]),
            lambda_agreement=float(model_cfg["lambda_agreement"]),
            discount_rho=float(model_cfg["discount_rho"]),
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
            f"view_{index}": self.decoders[f"view_{index}"](shared[f"view_{index}"])
            for index in range(self.num_views)
        }

    def _stack(self, shared: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack([shared[f"view_{index}"] for index in range(self.num_views)], dim=1)

    @staticmethod
    def _masked_mean(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        expanded = mask.float().unsqueeze(-1)
        return (features * expanded).sum(dim=1) / expanded.sum(dim=1).clamp_min(1.0)

    def _view_statistics(self, shared: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = [
            self.view_evidence_heads[f"view_{index}"](shared[f"view_{index}"])
            for index in range(self.num_views)
        ]
        return {
            key: torch.stack([output[key] for output in outputs], dim=1)
            for key in ("prob", "uncertainty", "entropy")
        }

    def _consensus(
        self,
        view_prob: torch.Tensor,
        features: torch.Tensor,
        observed_mask: torch.Tensor,
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
        kmeans = KMeans(
            n_clusters=self.num_clusters,
            n_init=20,
            random_state=int(self.cfg["seed"]),
        )
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
        statistics = self._view_statistics(shared)
        consensus = self._consensus(statistics["prob"], stacked, observed_mask)
        fusion = self.fusion(
            features=stacked,
            probs=statistics["prob"],
            uncertainties=statistics["uncertainty"],
            entropies=statistics["entropy"],
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
            "view_prob": statistics["prob"],
            "view_uncertainty": statistics["uncertainty"],
            "view_entropy": statistics["entropy"],
            "view_agreement": fusion["agreement"],
            "view_combined_reliability": fusion["combined_reliability"],
            "fusion_weights": fusion["weights"],
            "p_star": consensus,
        }

    def forward(self, batch: dict[str, Any], stage: str = "joint") -> dict[str, Any]:
        return self.forward_pretrain(batch) if stage == "pretrain" else self.forward_joint(batch)
