from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeClusteringHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_clusters: int,
        temperature: float = 0.1,
        use_sinkhorn: bool = False,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_clusters = num_clusters
        self.temperature = temperature
        self.use_sinkhorn = use_sinkhorn
        self.prototypes = nn.Parameter(torch.randn(num_clusters, feature_dim))
        nn.init.xavier_uniform_(self.prototypes)

    def _sinkhorn(self, logits: torch.Tensor, iters: int = 3) -> torch.Tensor:
        Q = torch.exp(logits / 0.05).t()  # [K, B]
        Q /= Q.sum()
        K, B = Q.shape
        for _ in range(iters):
            Q /= Q.sum(dim=1, keepdim=True)
            Q /= K
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= B
        Q *= B
        return Q.t()

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = F.normalize(z, dim=-1)
        p = F.normalize(self.prototypes, dim=-1)
        logits = (z @ p.t()) / self.temperature
        if self.use_sinkhorn:
            q = self._sinkhorn(logits)
        else:
            q = torch.softmax(logits, dim=-1)
        return {
            "q_global": q,
            "logits": logits,
            "prototypes": self.prototypes,
        }

    @torch.no_grad()
    def init_from_centers(self, centers: torch.Tensor) -> None:
        if centers.shape != self.prototypes.data.shape:
            raise ValueError(f"center shape mismatch: {centers.shape} vs {self.prototypes.data.shape}")
        self.prototypes.data.copy_(centers)
