from __future__ import annotations

from typing import Dict

import torch

from models.losses import reconstruction_loss
from utils import AverageMeter, get_or_create_observed_mask


class PretrainTrainer:
    """
    Reconstruction pretraining.

    Missing views are never used as reconstruction targets under the strict
    incomplete multi-view protocol.
    """

    def __init__(self, cfg: Dict) -> None:
        self.cfg = cfg
        fairness_cfg = cfg.get("fairness", {}) or {}
        dataset_cfg = cfg.get("dataset", {}) or {}
        self.strict_imvc = bool(fairness_cfg.get("strict_imvc", dataset_cfg.get("strict_imvc", False)))
        self.pretrain_use_complete_views = bool(cfg.get("train", {}).get("pretrain_use_complete_views", False))
        allow_complete = bool(fairness_cfg.get("allow_complete_view_pretrain", False))
        if self.strict_imvc and self.pretrain_use_complete_views and not allow_complete:
            raise ValueError(
                "Strict IMVC forbids train.pretrain_use_complete_views=true because it reconstructs "
                "views that are marked missing by observed_mask. Set pretrain_use_complete_views=false "
                "for fair experiments, or explicitly set fairness.allow_complete_view_pretrain=true "
                "for a non-strict diagnostic run."
            )

    def _reconstruction_mask(self, observed_mask: torch.Tensor) -> torch.Tensor:
        if self.pretrain_use_complete_views:
            return torch.ones_like(observed_mask, dtype=torch.bool)
        return observed_mask.bool()

    def train_one_epoch(self, model, loader, optimizer, device, epoch: int) -> Dict[str, float]:
        model.train()
        meter = AverageMeter()

        for batch_idx, batch in enumerate(loader):
            views = [x.to(device, non_blocking=True) for x in batch["views"]]
            observed_mask = get_or_create_observed_mask({**batch, "views": views}, cfg=self.cfg, epoch=epoch, batch_index=batch_idx)
            out = model({**batch, "views": views, "observed_mask": observed_mask}, stage="pretrain")
            rec_mask = self._reconstruction_mask(observed_mask)

            loss = torch.tensor(0.0, device=device)
            for v in range(len(views)):
                loss = loss + reconstruction_loss(views[v], out["recon_dict"][f"view_{v}"], rec_mask[:, v])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            meter.update(float(loss.item()), n=views[0].size(0))

        return {"loss": meter.avg}

    @torch.no_grad()
    def evaluate(self, model, loader, device) -> Dict[str, float]:
        model.eval()
        meter = AverageMeter()

        for batch_idx, batch in enumerate(loader):
            views = [x.to(device, non_blocking=True) for x in batch["views"]]
            observed_mask = get_or_create_observed_mask({**batch, "views": views}, cfg=self.cfg, batch_index=batch_idx)
            out = model({**batch, "views": views, "observed_mask": observed_mask}, stage="pretrain")
            rec_mask = self._reconstruction_mask(observed_mask)

            loss = torch.tensor(0.0, device=device)
            for v in range(len(views)):
                loss = loss + reconstruction_loss(views[v], out["recon_dict"][f"view_{v}"], rec_mask[:, v])
            meter.update(float(loss.item()), n=views[0].size(0))

        return {"loss": meter.avg}
