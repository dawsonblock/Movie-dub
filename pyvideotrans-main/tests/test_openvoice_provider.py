from videotrans.tts._openvoice import _partial_failure_message


def test_partial_failure_message_lists_failed_segment_ids():
    manifest = {
        "ok": 1,
        "results": [
            {"id": 1, "status": "ok"},
            {"id": 2, "status": "error", "error": "boom"},
        ],
    }

    message = _partial_failure_message(manifest, "/tmp/openvoice-manifest.json")

    assert "OpenVoice partially succeeded: 1/2 segments generated" in message
    assert "Failed/skipped segment IDs: 2" in message
    assert "Errors: 0, skipped: 0" in message
    assert "Manifest: /tmp/openvoice-manifest.json" in message


def test_partial_failure_message_lists_skipped_segment_ids():
    manifest = {
        "ok": 1,
        "skipped": 1,
        "results": [
            {"id": 1, "status": "ok"},
            {"id": 2, "status": "skipped_empty_text"},
        ],
    }

    message = _partial_failure_message(manifest, "/tmp/openvoice-manifest.json")

    assert "OpenVoice partially succeeded: 1/2 segments generated" in message
    assert "Failed/skipped segment IDs: 2" in message
    assert "skipped: 1" in message
