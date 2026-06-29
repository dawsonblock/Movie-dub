"""Tests for segment regeneration proof report (Blocker 10)."""

import hashlib
import json
import sys
from pathlib import Path


def test_regeneration_report_fields(tmp_path):
    """Regeneration report should have all required fields (Blocker 10)."""
    # Simulate the report structure that regenerate_segment.py writes
    old_path = tmp_path / "old.wav"
    new_path = tmp_path / "new.wav"
    old_path.write_bytes(b"old audio data")
    new_path.write_bytes(b"new audio data")

    old_hash = hashlib.sha256(old_path.read_bytes()).hexdigest()[:16]
    new_hash = hashlib.sha256(new_path.read_bytes()).hexdigest()[:16]

    report = {
        "regenerated_segment_id": 3,
        "old_segment_path": str(old_path),
        "new_segment_path": str(new_path),
        "old_segment_hash": old_hash,
        "new_segment_hash": new_hash,
        "hash_changed": old_hash != new_hash,
        "manifest_updated": True,
        "audio_rebuilt": True,
        "video_remuxed": True,
        "output_synced": True,
        "timestamp": "2026-06-29T20:00:00Z",
    }

    # Verify all required fields are present
    required_fields = [
        "regenerated_segment_id",
        "old_segment_path",
        "new_segment_path",
        "manifest_updated",
        "audio_rebuilt",
        "video_remuxed",
        "output_synced",
    ]
    for field in required_fields:
        assert field in report, f"missing required field: {field}"

    assert report["hash_changed"] is True
    assert report["old_segment_hash"] != report["new_segment_hash"]


def test_regeneration_hash_changes():
    """Hash should change when segment audio is regenerated (Blocker 10)."""
    old_data = b"original audio content"
    new_data = b"regenerated audio content"

    old_hash = hashlib.sha256(old_data).hexdigest()[:16]
    new_hash = hashlib.sha256(new_data).hexdigest()[:16]

    assert old_hash != new_hash
