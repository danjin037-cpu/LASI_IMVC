from typing import Dict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidenceHead(nn.Module):
    """Map a shared latent vector to evidence and uncertainty statistics."""
    def __init__(
        self,
        feature_dim: int,
        num_clusters: int,
        hidden_dim: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_clusters = num_clusters

        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, num_clusters),
            )
        else:
            self.net = nn.Linear(feature_dim, num_clusters)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Softplus keeps evidence non-negative.
        evidence = F.softplus(self.net(z))
        alpha = evidence + 1.0
        S = alpha.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        prob = alpha / S
        uncertainty = float(self.num_clusters) / S
        entropy = -(prob * torch.log(prob.clamp_min(1e-8))).sum(dim=-1, keepdim=True)
        entropy = entropy / math.log(self.num_clusters)

        return {
            "evidence": evidence,
            "alpha": alpha,
            "prob": prob,
            "uncertainty": uncertainty,
            "entropy": entropy,
        }


def build_view_evidence_heads(cfg, num_views: int, feature_dim: int, num_clusters: int) -> nn.ModuleDict:
    model_cfg = cfg["model"]
    hidden_dim = int(model_cfg.get("evidence_hidden_dim", 0))
    dropout = float(model_cfg.get("evidence_dropout", 0.0))

    heads = nn.ModuleDict()
    for v in range(num_views):
        heads[f"view_{v}"] = EvidenceHead(
            feature_dim=feature_dim,
            num_clusters=num_clusters,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    return heads
