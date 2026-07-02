from __future__ import annotations

import pytest

from scripts import structured_config as script_config
from src.experiments.structured_config import (
    load_structured_config,
    render_toml,
    write_structured_config,
)


def test_structured_config_round_trips_toml_and_json(tmp_path):
    payload = {
        "section": {"name": "demo", "enabled": True},
        "values": [1, 2, 3],
    }

    toml_path = write_structured_config(tmp_path / "config.toml", payload)
    json_path = write_structured_config(tmp_path / "config.json", payload)

    assert load_structured_config(toml_path) == payload
    assert load_structured_config(json_path) == payload


def test_structured_config_script_shim_reexports_public_module(tmp_path):
    payload = {"x": 1}
    path = script_config.write_structured_config(tmp_path / "shim.json", payload)

    assert script_config.load_structured_config(path) == payload
    assert script_config.render_toml is render_toml


def test_structured_config_rejects_null_toml_value():
    with pytest.raises(TypeError):
        render_toml({"unsupported": None})
