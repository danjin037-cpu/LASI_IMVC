import pytest

torch = pytest.importorskip("torch")

from lasi_imvc.config import load_config
from lasi_imvc.data import get_dataloaders


def test_scene15_and_clip_cache_are_aligned() -> None:
    cfg = load_config("configs/scene15.yaml")
    cfg["seed"] = 0
    cfg["dataset"]["mask_seed"] = 0
    train_loader, _, meta = get_dataloaders(cfg)
    batch = next(iter(train_loader))
    assert meta["num_samples_total"] == 4485
    assert meta["num_classes"] == 15
    assert meta["original_num_views"] == 2
    assert meta["num_views"] == 3
    assert len(batch["views"]) == 3
    assert batch["observed_mask"].shape[1] == 3
    assert not (batch["observed_mask"][:, 2] & ~batch["observed_mask"][:, :2].all(dim=1)).any()
