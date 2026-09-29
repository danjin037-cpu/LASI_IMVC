from typing import Dict, List

import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0, norm: str = "bn") -> None:
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
        hidden_dims: List[int],
        shared_dim: int,
        dropout: float = 0.0,
        norm: str = "bn",
    ) -> None:
        super().__init__()
        dims = [in_dim] + list(hidden_dims)
        layers = []
        for i in range(1, len(dims)):
            layers.append(MLPBlock(dims[i - 1], dims[i], dropout=dropout, norm=norm))
        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()
        last_dim = dims[-1]
        self.shared_head = nn.Linear(last_dim, shared_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(x)
        s = self.shared_head(h)
        return {"h": h, "s": s}


def build_view_encoders(cfg: Dict, input_dims: List[int]) -> nn.ModuleDict:
    model_cfg = cfg["model"]
    encoders = nn.ModuleDict()
    for i, in_dim in enumerate(input_dims):
        encoders[f"view_{i}"] = ViewEncoder(
            in_dim=in_dim,
            hidden_dims=model_cfg["encoder_hidden_dims"],
            shared_dim=model_cfg["shared_dim"],
            dropout=model_cfg.get("dropout", 0.0),
            norm=model_cfg.get("norm", "bn"),
        )
    return encoders
