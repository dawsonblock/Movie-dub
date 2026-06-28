import importlib.util
from pathlib import Path

import pytest


BRIDGE_PATH = Path(__file__).resolve().parents[2] / "bridge" / "openvoice_segment_tts.py"
SPEC = importlib.util.spec_from_file_location("openvoice_segment_tts", BRIDGE_PATH)
openvoice_segment_tts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(openvoice_segment_tts)


def test_normalize_openvoice_language_aliases():
    assert openvoice_segment_tts.normalize_language("en-us") == "EN"
    assert openvoice_segment_tts.normalize_language("zh-cn") == "ZH"
    assert openvoice_segment_tts.normalize_language("ja") == "JP"
    assert openvoice_segment_tts.normalize_language("ko-kr") == "KR"


def test_duration_status_buckets():
    assert openvoice_segment_tts.duration_status(1.0, 1.0) == (1.0, "accept")
    assert openvoice_segment_tts.duration_status(1.25, 1.0) == (1.25, "light_speedup_candidate")
    assert openvoice_segment_tts.duration_status(1.5, 1.0) == (1.5, "rewrite_shorter")
    assert openvoice_segment_tts.duration_status(0.5, 1.0) == (0.5, "padding_or_slowdown_candidate")


def test_start_time_end_time_are_milliseconds():
    start, end = openvoice_segment_tts.segment_times_seconds({"start_time": 0, "end_time": 800})
    assert start == 0.0
    assert end == 0.8


def test_start_end_are_seconds():
    start, end = openvoice_segment_tts.segment_times_seconds({"start": 0, "end": 0.8})
    assert start == 0.0
    assert end == 0.8


def test_missing_segment_times_are_unknown():
    assert openvoice_segment_tts.segment_times_seconds({"text": "hello"}) == (None, None)


def test_validate_checkpoint_dir_requires_converter_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        openvoice_segment_tts.validate_checkpoint_dir(tmp_path.as_posix())

    converter = tmp_path / "converter"
    converter.mkdir()
    (converter / "config.json").write_text("{}", encoding="utf-8")
    (converter / "checkpoint.pth").write_bytes(b"fake")

    assert openvoice_segment_tts.validate_checkpoint_dir(tmp_path.as_posix()) == converter


def test_return_code_for_manifest_counts():
    assert openvoice_segment_tts.return_code_for_counts(1, 0) == 0
    assert openvoice_segment_tts.return_code_for_counts(1, 1) == 2
    assert openvoice_segment_tts.return_code_for_counts(0, 0) == 3
    assert openvoice_segment_tts.return_code_for_counts(0, 1) == 3
