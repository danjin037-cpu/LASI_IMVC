from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        base_rel = base_reliability.unsqueeze(-1)
        consensus = target_consensus.detach().unsqueeze(1)
        kl = (consensus * (consensus.clamp_min(1e-8).log() - probs.clamp_min(1e-8).log())).sum(
            dim=-1, keepdim=True
        )
        agreement = torch.exp(-kl).clamp(min=1e-8, max=1.0)
        self_reliability = (
            self.beta_self_uncertainty * (1.0 - uncertainties)
            + self.beta_self_entropy * (1.0 - entropies)
        )
        raw_reliability = self.lambda_self * self_reliability + self.lambda_agreement * agreement
        discount = agreement.pow(max(float(self.discount_rho), 0.0))
        combined = raw_reliability * base_rel * discount

        router_input = torch.cat(
            [
                F.normalize(features, dim=-1),
                probs,
                uncertainties,
                entropies,
                agreement,
                raw_reliability,
                base_rel,
                combined,
            ],
            dim=-1,
        )
        logits = self.router(router_input) + combined.clamp_min(1e-8).log()
        logits = logits.masked_fill(~valid_mask.unsqueeze(-1), -1e4)
        weights = torch.softmax(logits / self.temperature, dim=1)
        fused = (weights * features * mask).sum(dim=1)
        return {
            "fused": fused,
            "weights": weights.squeeze(-1),
            "agreement": agreement.squeeze(-1),
            "combined_reliability": combined.squeeze(-1),
        }
