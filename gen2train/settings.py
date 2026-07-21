"""Gen2Train 앱 전역 경로/설정.

kohya_ss의 학습 엔진(sd-scripts)과 이미 세팅된 venv를 재사용하는 것을 전제로 한다.
"""
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SD_SCRIPTS_DIR = BASE_DIR / "sd-scripts"
DATASETS_DIR = BASE_DIR / "datasets"
OUTPUTS_DIR = BASE_DIR / "outputs"
PRESETS_DIR = BASE_DIR / "presets"
SETTINGS_FILE = BASE_DIR / "gen2train" / "settings.local.json"

DEFAULT_SETTINGS = {
    # run.bat이 공유 venv의 python.exe로 앱을 띄우므로 기본값은 sys.executable을 그대로 쓰면 된다.
    "python_path": sys.executable,
    "accelerate_path": None,  # None이면 python_path와 같은 Scripts 폴더의 accelerate(.exe)를 사용
    "num_cpu_threads_per_process": 2,
}


def _default_accelerate_path(python_path: str) -> str:
    p = Path(python_path)
    exe_name = "accelerate.exe" if p.suffix.lower() == ".exe" else "accelerate"
    return str(p.parent / exe_name)


def load_settings() -> dict:
    settings = dict(DEFAULT_SETTINGS)
    if SETTINGS_FILE.exists():
        try:
            saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            settings.update(saved)
        except (json.JSONDecodeError, OSError):
            pass
    if not settings.get("accelerate_path"):
        settings["accelerate_path"] = _default_accelerate_path(settings["python_path"])
    return settings


def save_settings(settings: dict) -> None:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_dirs() -> None:
    for d in (DATASETS_DIR, OUTPUTS_DIR, PRESETS_DIR):
        d.mkdir(parents=True, exist_ok=True)
