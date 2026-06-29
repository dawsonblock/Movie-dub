"""Tests for review_segments.json status propagation (Blocker 6)."""

import importlib.util
import json
import sys
from pathlib import Path


HELPERS_PATH = Path(__file__).resolve().parents[2] / "scripts" / "dub_job_helpers.py"
SPEC = importlib.util.spec_from_file_location("dub_job_helpers", HELPERS_PATH)
helpers = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = helpers
SPEC.loader.exec_module(helpers)


def _make_manifest(results: list[dict]) -> dict:
    return {"ok": sum(1 for r in results if r.get("status", "ok") == "ok"),
            "error": sum(1 for r in results if r.get("status") == "error"),
            "results": results}


def test_review_propagates_error_status(tmp_path):
    """A manifest item with status=error should become status=error in review (Blocker 6)."""
    manifest = _make_manifest([
        {"id": 1, "status": "ok", "timing_status": "accept", "output_audio": ""},
        {"id": 2, "status": "error", "timing_status": "accept", "output_audio": ""},
    ])
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    queue = [
        {"id": 1, "line": 1, "text": "hello", "start": 0.0, "end": 1.0},
        {"id": 2, "line": 2, "text": "world", "start": 1.0, "end": 2.0},
    ]
    queue_file = tmp_path / "queue.json"
    queue_file.write_text(json.dumps(queue), encoding="utf-8")

    review_path = tmp_path / "review.json"
    gen_dir = tmp_path / "generated"
    gen_dir.mkdir()
    helpers.write_review_file(
        review_path, queue_file, manifest_file, None, tmp_path, gen_dir
    )
    review = json.loads(review_path.read_text(encoding="utf-8"))
    assert review[0]["status"] == "ok"
    assert review[1]["status"] == "error"


def test_review_propagates_skipped_status(tmp_path):
    """A manifest item with status=skipped should become status=skipped in review (Blocker 6)."""
    manifest = _make_manifest([
        {"id": 1, "status": "skipped_empty_text", "timing_status": "accept"},
    ])
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    queue = [{"id": 1, "line": 1, "text": "", "start": 0.0, "end": 1.0}]
    queue_file = tmp_path / "queue.json"
    queue_file.write_text(json.dumps(queue), encoding="utf-8")

    review_path = tmp_path / "review.json"
    gen_dir = tmp_path / "generated"
    gen_dir.mkdir()
    helpers.write_review_file(
        review_path, queue_file, manifest_file, None, tmp_path, gen_dir
    )
    review = json.loads(review_path.read_text(encoding="utf-8"))
    assert review[0]["status"] == "skipped_empty_text"


def test_review_timing_status_still_works(tmp_path):
    """Timing-based needs_review should still work when manifest status is ok (Blocker 6)."""
    manifest = _make_manifest([
        {"id": 1, "status": "ok", "timing_status": "rewrite_shorter"},
    ])
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    queue = [{"id": 1, "line": 1, "text": "hello", "start": 0.0, "end": 1.0}]
    queue_file = tmp_path / "queue.json"
    queue_file.write_text(json.dumps(queue), encoding="utf-8")

    review_path = tmp_path / "review.json"
    gen_dir = tmp_path / "generated"
    gen_dir.mkdir()
    helpers.write_review_file(
        review_path, queue_file, manifest_file, None, tmp_path, gen_dir
    )
    review = json.loads(review_path.read_text(encoding="utf-8"))
    assert review[0]["status"] == "needs_review"


def test_review_manifest_status_overrides_timing(tmp_path):
    """Manifest status should take priority over timing_status (Blocker 6)."""
    manifest = _make_manifest([
        {"id": 1, "status": "error", "timing_status": "rewrite_shorter"},
    ])
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    queue = [{"id": 1, "line": 1, "text": "hello", "start": 0.0, "end": 1.0}]
    queue_file = tmp_path / "queue.json"
    queue_file.write_text(json.dumps(queue), encoding="utf-8")

    review_path = tmp_path / "review.json"
    gen_dir = tmp_path / "generated"
    gen_dir.mkdir()
    helpers.write_review_file(
        review_path, queue_file, manifest_file, None, tmp_path, gen_dir
    )
    review = json.loads(review_path.read_text(encoding="utf-8"))
    assert review[0]["status"] == "error"
