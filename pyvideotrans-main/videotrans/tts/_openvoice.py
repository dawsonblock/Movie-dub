import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from videotrans.configure import config
from videotrans.configure.config import ROOT_DIR, logger, settings
from videotrans.configure.excepts import DubbingSrtError, StopTask
from videotrans.tts._base import BaseTTS
from videotrans.util import tools


def _default_workspace_root() -> Path:
    return Path(ROOT_DIR).resolve().parent


def _setting_path(name: str, default: Path) -> str:
    value = str(settings.get(name, "") or "").strip()
    return Path(value).expanduser().as_posix() if value else default.as_posix()


def _normalize_openvoice_language(language: str | None) -> str:
    if not language:
        return "EN"
    value = language.strip().upper().replace("-", "_")
    language_map = {
        "EN_US": "EN",
        "EN_GB": "EN",
        "EN_AU": "EN",
        "EN_IN": "EN",
        "ZH_CN": "ZH",
        "ZH_TW": "ZH",
        "JA": "JP",
        "KO": "KR",
        "KO_KR": "KR",
    }
    return language_map.get(value, value.split("_")[0])


def _read_manifest(path: str) -> dict:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception(f"Failed to read OpenVoice manifest: {path}")
        return {}


def _partial_failure_message(manifest: dict, manifest_file: str) -> str:
    results = manifest.get("results") if isinstance(manifest, dict) else []
    failed = [item for item in results if item.get("status") == "error"] if isinstance(results, list) else []
    failed_ids = [str(item.get("id", item.get("index", "?"))) for item in failed[:20]]
    suffix = ""
    if len(failed) > len(failed_ids):
        suffix = f", +{len(failed) - len(failed_ids)} more"
    failed_text = ", ".join(failed_ids) + suffix if failed_ids else "unknown"
    return (
        f"OpenVoice partially succeeded: {manifest.get('ok', 0)}/{len(results) or '?'} segments generated. "
        f"Failed segment IDs: {failed_text}. Manifest: {manifest_file}"
    )


@dataclass
class OpenVoiceTTS(BaseTTS):
    def __post_init__(self):
        super().__post_init__()
        workspace = _default_workspace_root()
        self.openvoice_repo = _setting_path("openvoice_repo_path", workspace / "OpenVoice-main")
        self.openvoice_env_python = _setting_path(
            "openvoice_env_python", Path(self.openvoice_repo) / ".venv" / "bin" / "python"
        )
        self.openvoice_checkpoint_dir = _setting_path(
            "openvoice_checkpoint_dir", Path(self.openvoice_repo) / "checkpoints_v2"
        )
        self.default_reference = str(settings.get("openvoice_default_reference", "") or "").strip()
        self.openvoice_language = str(settings.get("openvoice_language", "") or "").strip()
        self.openvoice_base_speaker = str(settings.get("openvoice_base_speaker", "") or "").strip()
        self.bridge_script = _setting_path("openvoice_bridge_script", workspace / "bridge" / "openvoice_segment_tts.py")

    def _validate_paths(self):
        missing = []
        for label, path in [
            ("OpenVoice env python", self.openvoice_env_python),
            ("OpenVoice repo", self.openvoice_repo),
            ("OpenVoice bridge script", self.bridge_script),
            ("OpenVoice checkpoint dir", self.openvoice_checkpoint_dir),
        ]:
            if not Path(path).exists():
                missing.append(f"{label}: {path}")
        converter = Path(self.openvoice_checkpoint_dir) / "converter"
        if not (converter / "config.json").is_file() or not (converter / "checkpoint.pth").is_file():
            missing.append(f"OpenVoice V2 converter checkpoint: {converter}")
        if missing:
            raise StopTask(
                "OpenVoice is not ready. Configure/install the missing paths:\n" + "\n".join(missing)
            )

    def _prepare_queue(self):
        bridge_queue = []
        for index, item in enumerate(self.queue_tts):
            raw_output = f"{item['filename']}-openvoice.wav"
            new_item = dict(item)
            new_item["openvoice_output"] = raw_output
            if str(new_item.get("role", "")).strip().lower() != "clone" and not new_item.get("voice_reference"):
                new_item["voice_reference"] = self.default_reference
            bridge_queue.append(new_item)
            self.queue_tts[index]["openvoice_output"] = raw_output
        return bridge_queue

    def _exec(self):
        self._validate_paths()
        queue_tts_file = f"{config.TEMP_DIR}/{self.uuid}/openvoice-queue-{time.time()}.json"
        manifest_file = f"{config.TEMP_DIR}/{self.uuid}/openvoice-manifest-{time.time()}.json"
        logs_file = f"{config.TEMP_DIR}/{self.uuid}/openvoice-{time.time()}.log"
        work_dir = f"{config.TEMP_DIR}/{self.uuid}/openvoice-work"

        bridge_queue = self._prepare_queue()
        Path(queue_tts_file).write_text(json.dumps(bridge_queue, ensure_ascii=False), encoding="utf-8")
        language = self.openvoice_language or _normalize_openvoice_language(self.language)

        cmd = [
            self.openvoice_env_python,
            self.bridge_script,
            "--queue-tts-file",
            queue_tts_file,
            "--manifest-file",
            manifest_file,
            "--work-dir",
            work_dir,
            "--openvoice-repo",
            self.openvoice_repo,
            "--checkpoint-dir",
            self.openvoice_checkpoint_dir,
            "--language",
            language,
            "--base-speaker",
            self.openvoice_base_speaker,
            "--speed",
            str(self.get_speed()),
            "--logs-file",
            logs_file,
        ]
        if self.default_reference:
            cmd.extend(["--default-reference", self.default_reference])
        if self.is_cuda:
            cmd.extend(["--device", "cuda:0"])

        self.signal(text="OpenVoice subprocess starting")
        logger.debug(f"OpenVoice command: {cmd}")
        result = subprocess.run(
            cmd,
            cwd=self.openvoice_repo,
            text=True,
            capture_output=True,
            timeout=None,
            env={**os.environ, "PYTHONPATH": self.openvoice_repo + os.pathsep + os.environ.get("PYTHONPATH", "")},
        )
        if result.stdout:
            logger.debug(result.stdout)
        if result.stderr:
            if result.returncode:
                logger.error(result.stderr)
            else:
                logger.debug(result.stderr)

        manifest = _read_manifest(manifest_file)
        if manifest:
            logger.debug(json.dumps(manifest, ensure_ascii=False, indent=2))
        if result.returncode not in (0, 2):
            raise DubbingSrtError(f"OpenVoice failed with exit code {result.returncode}: {result.stderr[-2000:]}")
        if result.returncode == 2:
            message = _partial_failure_message(manifest, manifest_file)
            logger.error(message)
            self.signal(text=message)

        all_task = []
        with ThreadPoolExecutor(max_workers=min(4, len(self.queue_tts), os.cpu_count() or 1)) as pool:
            for item in self.queue_tts:
                raw_output = item.get("openvoice_output")
                if raw_output and tools.vail_file(raw_output):
                    all_task.append(pool.submit(self.convert_to_wav, raw_output, item["filename"]))
            if all_task:
                _ = [task.result() for task in all_task]

        succeed = len([item for item in self.queue_tts if tools.vail_file(item.get("filename"))])
        if succeed < 1:
            raise DubbingSrtError("OpenVoice generated no usable dubbing audio")
        if result.returncode == 2:
            message = _partial_failure_message(manifest, manifest_file)
            raise DubbingSrtError(message)
        self.signal(text=f"OpenVoice dubbing ended: {succeed}/{len(self.queue_tts)}")
