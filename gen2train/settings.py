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
    # run.bat/run.sh가 적절한 venv의 python으로 앱을 띄우므로 기본값은 sys.executable을 그대로
    # 쓰면 된다(Windows는 python.exe, Linux/WSL2는 python).
    "python_path": sys.executable,
    "accelerate_path": None,  # None이면 python_path와 같은 폴더(Windows: Scripts, Linux: bin)의 accelerate를 사용
    "num_cpu_threads_per_process": 2,
    # 멀티 GPU: "auto"면 nvidia-smi로 감지된 GPU가 2개 이상일 때 전부 자동으로 사용한다
    # (trainer.py의 _resolve_multi_gpu_args 참고). GPU 개수를 여기 하드코딩하지 않는다.
    "multi_gpu": "auto",  # "auto" | true | false
    "gpu_ids": "",  # 비워두면 auto 감지 결과를 그대로 쓴다. 예: "0,1" -> 0,1번 GPU만 사용
    "num_machines": 1,
    # WSL2에서 멀티 GPU 학습 시 NCCL P2P/SHM을 꺼야 CUDA error 999를 피할 수 있다(trainer.py의
    # system_info.is_wsl() 감지 시에만 적용, 순수 Windows에는 영향 없음). SHM을 끈 채로도
    # 문제가 재발하면 이 값을 true로 바꿔 NCCL_SHM_DISABLE=1까지 추가로 켠다(속도는 손해).
    "nccl_shm_disable": False,
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
