from typing import Dict, List

import torch
import torch.nn as nn


class ViewDecoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        dims = [in_dim] + list(hidden_dims)
        layers = []
        for i in range(1, len(dims)):
            layers.extend([
                nn.Linear(dims[i - 1], dims[i]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            ])
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def build_view_decoders(cfg: Dict, input_dims: List[int]) -> nn.ModuleDict:
    model_cfg = cfg["model"]
    in_dim = model_cfg["shared_dim"]
    decoders = nn.ModuleDict()
    for i, out_dim in enumerate(input_dims):
        decoders[f"view_{i}"] = ViewDecoder(
            in_dim=in_dim,
            hidden_dims=model_cfg["decoder_hidden_dims"],
            out_dim=out_dim,
            dropout=model_cfg.get("dropout", 0.0),
        )
    return decoders
