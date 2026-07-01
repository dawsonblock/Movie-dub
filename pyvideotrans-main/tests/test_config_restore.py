"""Tests for config creation and restoration (Blocker 4)."""

import json
import sys
from pathlib import Path


def test_config_restore_removes_missing_keys(tmp_path):
    """Config restore should remove keys that didn't exist before (Blocker 4).

    Simulates the logic in run_personal_dub.py: if a key was not in the
    original config, it should be removed (not set to empty string) after
    restoration.
    """
    cfg_path = tmp_path / "cfg.json"
    # Original config has only "some_key"
    original = {"some_key": "value"}
    cfg_path.write_text(json.dumps(original), encoding="utf-8")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    # Track original state of keys we will modify
    cfg_original = {}
    for key in ("tts_default_reference", "tts_preserve_dir"):
        cfg_original[key] = {"existed": key in cfg, "value": cfg.get(key)}

    # Set the keys (simulating the run)
    cfg["tts_default_reference"] = "/ref.wav"
    cfg["tts_preserve_dir"] = "/preserve"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    # Simulate pyVideoTrans adding its own keys during the run
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["pyvt_internal_key"] = "something"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    # Restore: read current, apply original state
    current = json.loads(cfg_path.read_text(encoding="utf-8"))
    for key, state in cfg_original.items():
        if state["existed"]:
            current[key] = state["value"]
        else:
            current.pop(key, None)
    cfg_path.write_text(json.dumps(current), encoding="utf-8")

    final = json.loads(cfg_path.read_text(encoding="utf-8"))
    # Keys that didn't exist should be removed, not empty strings
    assert "tts_default_reference" not in final
    assert "tts_preserve_dir" not in final
    # Keys that did exist should be restored
    assert final.get("some_key") == "value"
    # pyVideoTrans keys should be preserved
    assert final.get("pyvt_internal_key") == "something"


def test_config_restore_preserves_existing_keys(tmp_path):
    """Config restore should preserve existing key values (Blocker 4)."""
    cfg_path = tmp_path / "cfg.json"
    original = {
        "tts_default_reference": "/original_ref.wav",
        "tts_preserve_dir": "/original_preserve",
        "other_key": "keep_me",
    }
    cfg_path.write_text(json.dumps(original), encoding="utf-8")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg_original = {}
    for key in ("tts_default_reference", "tts_preserve_dir"):
        cfg_original[key] = {"existed": key in cfg, "value": cfg.get(key)}

    # Modify during run
    cfg["tts_default_reference"] = "/new_ref.wav"
    cfg["tts_preserve_dir"] = "/new_preserve"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    # Restore
    current = json.loads(cfg_path.read_text(encoding="utf-8"))
    for key, state in cfg_original.items():
        if state["existed"]:
            current[key] = state["value"]
        else:
            current.pop(key, None)
    cfg_path.write_text(json.dumps(current), encoding="utf-8")

    final = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert final["tts_default_reference"] == "/original_ref.wav"
    assert final["tts_preserve_dir"] == "/original_preserve"
    assert final["other_key"] == "keep_me"


def test_config_creation_when_missing(tmp_path):
    """Config should be created if it doesn't exist (Blocker 4)."""
    cfg_path = tmp_path / "subdir" / "cfg.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    else:
        cfg = {}

    cfg["tts_default_reference"] = "/ref.wav"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    assert cfg_path.is_file()
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert data["tts_default_reference"] == "/ref.wav"
