"""Config loading, including ``extends``."""

import pytest

from rtcb_gym.config import DEFAULT_CONFIG, REPO_ROOT, load_config


def test_default_config_loads():
    cfg = load_config()
    assert cfg["market"]["location"] == "HB_WEST"
    assert "extends" not in cfg


def test_extends_overrides_only_given_keys(tmp_path):
    child = tmp_path / "child.yaml"
    child.write_text(
        f"extends: {DEFAULT_CONFIG}\n"
        "market:\n  location: HB_HOUSTON\n"
        "battery:\n  power_mw: 50\n"
    )
    base, cfg = load_config(), load_config(child)
    assert cfg["market"]["location"] == "HB_HOUSTON"
    assert cfg["market"]["start_date"] == base["market"]["start_date"]  # nested merge
    assert cfg["battery"]["power_mw"] == 50
    assert cfg["battery"]["energy_mwh"] == base["battery"]["energy_mwh"]
    assert cfg["splits"] == base["splits"]
    assert "extends" not in cfg


def test_extends_path_is_relative_to_the_config_file(tmp_path):
    (tmp_path / "base.yaml").write_text(DEFAULT_CONFIG.read_text())
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "child.yaml").write_text("extends: ../base.yaml\nenv:\n  history_len: 48\n")
    cfg = load_config(sub / "child.yaml")
    assert cfg["env"]["history_len"] == 48
    assert cfg["env"]["episode_days"] == 1


def test_lists_are_replaced_not_merged(tmp_path):
    child = tmp_path / "child.yaml"
    child.write_text(
        f"extends: {DEFAULT_CONFIG}\n"
        "baselines:\n  threshold_rule:\n    low_quantiles: [0.25]\n"
    )
    cfg = load_config(child)
    assert cfg["baselines"]["threshold_rule"]["low_quantiles"] == [0.25]
    assert cfg["baselines"]["threshold_rule"]["high_quantiles"] == [0.6, 0.7, 0.8, 0.9]


def test_extends_cycle_is_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ValueError, match="extends itself"):
        load_config(tmp_path / "a.yaml")


def test_missing_section_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("market:\n  location: HB_WEST\n")
    with pytest.raises(ValueError, match="missing section"):
        load_config(p)


def test_shipped_configs_load():
    for path in sorted((REPO_ROOT / "configs").glob("*.yaml")):
        cfg = load_config(path)
        assert cfg["market"]["location"].startswith("HB_")
