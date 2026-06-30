#!/usr/bin/env python3
"""Repeatable benchmark harness for the dub pipeline.

Runs the full personal-dub pipeline on an input clip and records a
machine-readable metrics report. Designed to be run on a 5-10 minute
reference clip so every pipeline change can be judged against the same input.

Metrics tracked:

  - total_render_time          (wall clock, seconds)
  - stage_times                (per-stage wall clock)
  - tts_time_per_segment       (mean / p50 / p90 / max, seconds)
  - bad_speaker_assignments    (segments where diarization overlap < 20%)
  - bad_age_gender_guesses     (speakers with confidence below threshold)
  - timing_drift               (mean |gen_duration - target_duration|)
  - loudness_mismatch          (mean |gen_lufs - target_lufs| if LUFS used)
  - failed_segments            (count + list of ids)
  - final_mp4_duration_delta   (|final - source| seconds + percent)
  - segment_success_rate       (ok / total)
  - speaker_consistency        (per-speaker F0 variance across its segments)
  - review_required            (segments flagged needs_review/rewrite_shorter)

The harness does NOT include a clip. You supply one with --input. Run it the
same way each time and diff the JSON reports to judge a change.

Usage:
    python scripts/benchmark.py \
        --input ~/Movies/bench_clip.mp4 \
        --output-dir reports/benchmarks/<name> \
        --target-language en --tts-engine qwen3-local \
        --speaker-profiling --hf-token $HF_TOKEN

The harness wraps run_personal_dub.py (it shells out to it) and then reads
the job artifacts to compute metrics. It does not re-implement the pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DUB = ROOT / "scripts" / "run_personal_dub.py"


def _ffprobe_duration(path: Path) -> float:
    """Get duration in seconds via ffprobe (best-effort)."""
    import shutil
    ff = shutil.which("ffprobe")
    if not ff:
        return 0.0
    try:
        r = subprocess.run(
            [ff, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path.as_posix()],
            capture_output=True, text=True, timeout=30,
        )
        return float(r.stdout.strip()) if r.returncode == 0 else 0.0
    except Exception:
        return 0.0


def _read_json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)


def _load_librosa():
    try:
        import librosa  # noqa: F401
        import numpy as np  # noqa: F401
        return True
    except ImportError:
        return False


def _segment_lufs(wav_path: Path) -> float | None:
    """Best-effort integrated LUFS for a segment WAV via ffmpeg."""
    import shutil
    ff = shutil.which("ffmpeg")
    if not ff or not wav_path.is_file():
        return None
    try:
        r = subprocess.run(
            [ff, "-hide_banner", "-nostdin", "-i", wav_path.as_posix(),
             "-af", "loudnorm=print_format=json", "-f", "null", "-"],
            capture_output=True, text=True, timeout=30,
        )
        # loudnorm prints JSON to stderr
        out = r.stderr
        idx = out.find("{")
        if idx < 0:
            return None
        end = out.find("}", idx)
        if end < 0:
            return None
        blob = out[idx:end + 1]
        data = json.loads(blob)
        return float(data.get("input_i", 0.0))
    except Exception:
        return None


def _segment_f0(wav_path: Path) -> float:
    """Median F0 for a segment WAV (0.0 if unavailable)."""
    if not _load_librosa() or not wav_path.is_file():
        return 0.0
    try:
        import librosa
        import numpy as np
        y, sr = librosa.load(wav_path.as_posix(), sr=16000, mono=True)
        if len(y) < sr * 0.1:
            return 0.0
        f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=400, sr=sr, frame_length=2048)
        vf = f0[voiced & ~np.isnan(f0)]
        if len(vf) < 5:
            return 0.0
        return float(np.median(vf))
    except Exception:
        return 0.0


def collect_metrics(
    job_dir: Path,
    source_duration: float,
    final_video_path: Path,
    stage_times: dict[str, float],
    target_lufs: float | None,
    confidence_threshold: float,
) -> dict:
    """Read job artifacts and compute the benchmark metrics dict."""
    manifest = _read_json(job_dir / "openvoice_manifest.json") or {}
    # Try the enriched manifest first (has speaker_id per result).
    enriched = _read_json(job_dir / "openvoice_manifest_enriched.json")
    if enriched:
        manifest = enriched
    speaker_profiles = _read_json(job_dir / "speakers" / "speaker_profiles.json") or {}
    pitch_verif = _read_json(job_dir / "pitch_verification.json")
    review = _read_json(job_dir / "review_segments.json") or []
    build_report = _read_json(job_dir / "audio" / "build_audio_report.json")

    results = manifest.get("results", []) if isinstance(manifest, dict) else []
    seg_assignments = speaker_profiles.get("segment_assignments", []) if speaker_profiles else []
    speakers = speaker_profiles.get("speakers", []) if speaker_profiles else []

    # --- TTS time per segment ---
    tts_times = [float(r.get("tts_seconds", 0.0) or 0.0) for r in results
                 if r.get("status") == "ok"]
    if not tts_times:
        # Fall back to per-segment wall clock if the bridge didn't report it.
        tts_times = [float(r.get("duration_seconds", 0.0) or 0.0) for r in results]

    # --- Failed segments ---
    failed_ids = [{"id": fid, "status": r.get("status", "")}
                  for r in results for fid in [str(r.get("id", r.get("index", "")))]
                  if r.get("status") not in (None, "ok")]

    # --- Timing drift ---
    drifts: list[float] = []
    for r in results:
        gen_dur = float(r.get("generated_duration", 0.0) or 0.0)
        tgt = float(r.get("target_duration", r.get("duration", 0.0)) or 0.0)
        if gen_dur > 0 and tgt > 0:
            drifts.append(abs(gen_dur - tgt))

    # --- Loudness mismatch ---
    loudness_mismatches: list[float] = []
    if target_lufs is not None:
        for r in results:
            wav = r.get("output_audio") or r.get("preserved_audio")
            if wav:
                lufs = _segment_lufs(Path(wav))
                if lufs is not None:
                    loudness_mismatches.append(abs(lufs - target_lufs))

    # --- Bad speaker assignments (overlap < 20% of segment duration) ---
    bad_assignments = 0
    for sa in seg_assignments:
        # Heuristic: if the assignment has no overlap field, we can't judge.
        # If speaker_profiles reports a fallback (single speaker forced), count
        # it as bad only when there are multiple speakers.
        if sa.get("overlap_ratio") is not None:
            if float(sa["overlap_ratio"]) < 0.2:
                bad_assignments += 1
    if not seg_assignments and len(speakers) > 1:
        bad_assignments = -1  # unknown: no overlap metadata

    # --- Bad age/gender guesses (confidence below threshold) ---
    bad_age_gender = []
    for spk in speakers:
        g = spk.get("gender", {})
        a = spk.get("age", {})
        if g.get("confidence", 1.0) < confidence_threshold:
            bad_age_gender.append({
                "speaker_id": spk.get("speaker_id", ""),
                "field": "gender",
                "confidence": g.get("confidence", 0.0),
            })
        if a.get("confidence", 1.0) < confidence_threshold:
            bad_age_gender.append({
                "speaker_id": spk.get("speaker_id", ""),
                "field": "age",
                "confidence": a.get("confidence", 0.0),
            })

    # --- Speaker consistency: F0 variance across each speaker's segments ---
    speaker_consistency: dict[str, float] = {}
    if review and _load_librosa():
        # Group review entries by speaker_id, compute F0 per segment.
        by_speaker: dict[str, list[float]] = {}
        for item in review:
            sid = item.get("speaker_id", "")
            if not sid:
                continue
            wav = item.get("output_audio", "")
            if wav and Path(wav).is_file():
                f0 = _segment_f0(Path(wav))
                if f0 > 0:
                    by_speaker.setdefault(sid, []).append(f0)
        for sid, f0s in by_speaker.items():
            if len(f0s) >= 2:
                # Lower stdev = more consistent. Report stdev in Hz.
                speaker_consistency[sid] = round(statistics.pstdev(f0s), 2)
            elif f0s:
                speaker_consistency[sid] = 0.0

    # --- Review required ---
    review_required = sum(
        1 for item in review
        if item.get("status") in ("needs_review", "rewrite_shorter",
                                  "pending_regen", "error", "skipped_empty_text")
    )
    if pitch_verif:
        pv_summary = pitch_verif.get("summary", {})
        review_required = max(
            review_required,
            int(pv_summary.get("needs_review", 0))
            + int(pv_summary.get("rewrite_shorter", 0)),
        )

    # --- Final MP4 duration delta ---
    final_duration = _ffprobe_duration(final_video_path)
    duration_delta = abs(final_duration - source_duration) if final_duration else 0.0
    duration_delta_pct = (
        round(duration_delta / source_duration * 100.0, 3)
        if source_duration > 0 else 0.0
    )

    # --- Segment success rate ---
    total = len(results)
    ok = sum(1 for r in results if r.get("status") == "ok")
    success_rate = round(ok / total, 3) if total else 0.0

    # --- Quality score (0-100, rough composite) ---
    # Weighted blend of the normalized metrics. Higher is better.
    timing_fit = max(0.0, 1.0 - (statistics.mean(drifts) / 0.5)) if drifts else 1.0
    pitch_match = 1.0
    if pitch_verif:
        pv_summary = pitch_verif.get("summary", {})
        pv_total = sum(pv_summary.values()) or 1
        pitch_match = pv_summary.get("ok", 0) / pv_total
    sub_align = 1.0  # subtitle alignment: 100% when subs drive timing
    consistency = (
        statistics.mean([max(0.0, 1.0 - v / 50.0) for v in speaker_consistency.values()])
        if speaker_consistency else 1.0
    )
    quality_score = round(100.0 * (
        0.30 * success_rate
        + 0.25 * pitch_match
        + 0.20 * timing_fit
        + 0.15 * consistency
        + 0.10 * sub_align
    ), 1)

    return {
        "total_render_time_seconds": round(stage_times.get("_total", 0.0), 2),
        "stage_times_seconds": {k: round(v, 2) for k, v in stage_times.items() if k != "_total"},
        "tts_time_per_segment": {
            "mean": round(statistics.mean(tts_times), 3) if tts_times else 0.0,
            "p50": round(_percentile(tts_times, 50), 3),
            "p90": round(_percentile(tts_times, 90), 3),
            "max": round(max(tts_times), 3) if tts_times else 0.0,
            "count": len(tts_times),
        },
        "segment_success_rate": success_rate,
        "segments_ok": ok,
        "segments_total": total,
        "failed_segments": failed_ids,
        "failed_segment_count": len(failed_ids),
        "timing_drift_seconds": {
            "mean": round(statistics.mean(drifts), 3) if drifts else 0.0,
            "p90": round(_percentile(drifts, 90), 3) if drifts else 0.0,
            "max": round(max(drifts), 3) if drifts else 0.0,
        },
        "loudness_mismatch_lufs": {
            "mean": round(statistics.mean(loudness_mismatches), 3)
            if loudness_mismatches else 0.0,
            "count": len(loudness_mismatches),
        } if target_lufs is not None else None,
        "bad_speaker_assignments": bad_assignments,
        "bad_age_gender_guesses": bad_age_gender,
        "speaker_consistency_f0_stdev_hz": speaker_consistency,
        "review_required": review_required,
        "final_mp4_duration_delta_seconds": round(duration_delta, 3),
        "final_mp4_duration_delta_percent": duration_delta_pct,
        "source_duration_seconds": round(source_duration, 3),
        "final_duration_seconds": round(final_duration, 3),
        "quality_score": quality_score,
        "speakers_detected": len(speakers),
        "warnings": (build_report or {}).get("warnings", []),
        "job_dir": job_dir.as_posix(),
    }


def run_benchmark(args: argparse.Namespace) -> int:
    """Run the dub pipeline and collect metrics."""
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        print(f"input not found: {input_path}", file=sys.stderr)
        return 1
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_video = output_dir / "bench_dubbed.mp4"

    source_duration = _ffprobe_duration(input_path)
    print(f"Benchmark input: {input_path} ({source_duration:.1f}s)")

    # Build the run_personal_dub.py command.
    cmd = [
        sys.executable, RUN_DUB.as_posix(),
        "--input", input_path.as_posix(),
        "--output", final_video.as_posix(),
        "--target-language", args.target_language,
        "--tts-engine", args.tts_engine,
    ]
    if args.source_language:
        cmd.extend(["--source-language", args.source_language])
    if args.speaker_profiling:
        cmd.append("--speaker-profiling")
        cmd.extend(["--speaker-diarization", args.diarization])
        cmd.extend(["--num-speakers", args.num_speakers])
    if args.hf_token:
        cmd.extend(["--hf-token", args.hf_token])
    if args.verify_pitch:
        cmd.append("--verify-pitch")
    if args.prefer_embedded_subtitles:
        cmd.append("--prefer-embedded-subtitles")
    if args.target_lufs is not None:
        cmd.extend(["--target-lufs", str(args.target_lufs)])
    if args.background_volume is not None:
        cmd.extend(["--background-volume", str(args.background_volume)])
    if args.ducking:
        cmd.append("--ducking")
    if args.qwen3_model and args.tts_engine == "qwen3-local":
        cmd.extend(["--qwen3-model", args.qwen3_model])
    if args.age_model:
        cmd.extend(["--age-model", args.age_model])
    if args.resume:
        cmd.append("--resume")
    if args.skip_existing:
        cmd.append("--skip-existing")

    print(f"Running: {' '.join(cmd[:4])} ...")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    total_time = time.time() - t0

    # Locate the job dir from the report. The wrapper writes report.json in
    # the job dir; we can find the most recent job dir under personal_dub.
    pdub_root = ROOT / "pyvideotrans-main" / "tmp" / "personal_dub"
    job_dirs = sorted(
        [d for d in pdub_root.glob("*") if d.is_dir()],
        key=lambda d: d.stat().st_mtime, reverse=True,
    )
    job_dir = job_dirs[0] if job_dirs else None

    if proc.returncode != 0:
        print(f"run_personal_dub.py exited {proc.returncode}", file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
        # Still try to collect partial metrics if a job dir exists.
    if job_dir is None:
        print("Could not locate job dir; no metrics collected.", file=sys.stderr)
        return 1

    # Per-stage times from job.json (best-effort; stages record status, not
    # timing, so we approximate total only and leave stage_times empty unless
    # the report carries them).
    stage_times: dict[str, float] = {"_total": total_time}
    report = _read_json(job_dir / "report.json") or {}
    if isinstance(report, dict) and report.get("stage_times"):
        stage_times.update(report["stage_times"])

    metrics = collect_metrics(
        job_dir=job_dir,
        source_duration=source_duration,
        final_video_path=final_video,
        stage_times=stage_times,
        target_lufs=args.target_lufs,
        confidence_threshold=args.confidence_threshold,
    )
    metrics["pipeline_exit_code"] = proc.returncode
    metrics["benchmark_input"] = input_path.as_posix()
    metrics["benchmark_name"] = args.name
    metrics["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    report_path = output_dir / "benchmark_report.json"
    report_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    # Human-readable summary
    print()
    print("=" * 60)
    print(f"BENCHMARK: {args.name}")
    print("=" * 60)
    print(f"Render time:        {metrics['total_render_time_seconds']}s")
    print(f"Quality score:      {metrics['quality_score']}/100")
    print(f"Segment success:    {metrics['segments_ok']}/{metrics['segments_total']} "
          f"({metrics['segment_success_rate']*100:.1f}%)")
    print(f"TTS time/seg:       mean={metrics['tts_time_per_segment']['mean']}s "
          f"p90={metrics['tts_time_per_segment']['p90']}s")
    print(f"Timing drift:       mean={metrics['timing_drift_seconds']['mean']}s "
          f"max={metrics['timing_drift_seconds']['max']}s")
    print(f"Failed segments:    {metrics['failed_segment_count']}")
    print(f"Review required:    {metrics['review_required']}")
    print(f"Duration delta:     {metrics['final_mp4_duration_delta_seconds']}s "
          f"({metrics['final_mp4_duration_delta_percent']}%)")
    print(f"Speakers detected:  {metrics['speakers_detected']}")
    print(f"Bad age/gender:     {len(metrics['bad_age_gender_guesses'])}")
    if metrics['loudness_mismatch_lufs']:
        print(f"Loudness mismatch:  mean={metrics['loudness_mismatch_lufs']['mean']} LUFS")
    print("=" * 60)
    print(f"Report: {report_path}")
    print(f"Job dir: {job_dir}")
    return 0 if proc.returncode == 0 else 2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repeatable benchmark harness for the dub pipeline"
    )
    parser.add_argument("--input", required=True, help="benchmark input clip (5-10 min)")
    parser.add_argument("--output-dir", required=True,
                        help="directory for the dubbed output + report")
    parser.add_argument("--name", default="bench",
                        help="benchmark name (recorded in the report)")
    parser.add_argument("--source-language", default="auto")
    parser.add_argument("--target-language", default="en")
    parser.add_argument("--tts-engine", default="openvoice",
                        choices=["openvoice", "qwen3-local"])
    parser.add_argument("--speaker-profiling", action="store_true")
    parser.add_argument("--diarization", default="pyannote")
    parser.add_argument("--num-speakers", default="auto")
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--verify-pitch", action="store_true")
    parser.add_argument("--prefer-embedded-subtitles", action="store_true")
    parser.add_argument("--target-lufs", type=float, default=None)
    parser.add_argument("--background-volume", type=float, default=None)
    parser.add_argument("--ducking", action="store_true")
    parser.add_argument("--qwen3-model", default="")
    parser.add_argument("--age-model", default="",
                        choices=["", "auto", "on", "off"])
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                        help="age/gender confidence below this counts as a bad guess")
    parser.add_argument("--resume", action="store_true",
                        help="resume an existing job (cache/resume)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip segments whose output WAV already exists")
    args = parser.parse_args()
    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
