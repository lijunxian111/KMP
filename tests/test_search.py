from pathlib import Path
import pytest
import yaml
from kmp.search import candidates
from kmp.cli import config_from_args


def test_paper_grid_has_1620_settings():
    grid = yaml.safe_load((Path(__file__).parents[1] / "configs/grid.yaml").read_text())
    assert len(candidates(grid)) == 1620
    with pytest.raises(ValueError):
        candidates({"test_accuracy": [.5]})


def test_yaml_config_and_cli_override(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("pe_dim: 16\nseeds: [0, 1]\nlr: 0.005\n")
    c = config_from_args(["--config", str(path), "--pe-dim", "4"])
    assert c["pe_dim"] == 4 and c["lr"] == .005 and c["seeds"] == [0, 1]
