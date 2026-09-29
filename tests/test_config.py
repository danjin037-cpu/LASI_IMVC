from pathlib import Path

import pytest

from lasi_imvc.config import load_config


def test_minimal_config_uses_scene15_defaults() -> None:
    cfg = load_config(Path("configs/scene15.yaml"))
    assert cfg["dataset"]["name"] == "Scene15"
    assert cfg["clip_view"]["source_view_indices"] == [0, 1]
    assert cfg["model"]["shared_dim"] == 128


def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("obsolete_option: true\n", encoding="utf-8")
    with pytest.raises(KeyError, match="obsolete_option"):
        load_config(path)
