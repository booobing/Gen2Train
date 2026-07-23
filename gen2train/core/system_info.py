"""GPU/VRAM/CPU/RAM 시스템 정보를 주기적으로 수집하는 하트비트.

AI 도우미(챗봇)가 "지금 VRAM이 얼마나 남았는지" 같은 실제 하드웨어 상황을 근거로 추천할 수
있도록, chat_context.py가 매 요청마다 최신 스냅샷을 컨텍스트에 끼워 넣는다.

메인 GUI 프로세스에 torch/CUDA를 직접 로드하지 않기 위해(불필요한 VRAM 점유, 시작 지연을
피하려고) GPU 정보는 nvidia-smi를 서브프로세스로 호출해서 얻는다. 수집 자체(서브프로세스
호출 + psutil의 cpu_percent 블로킹 호출)는 GUI를 멈추지 않도록 백그라운드 스레드에서 돈다.
"""
import os
import shutil
import subprocess
import threading

import psutil
from PySide6.QtCore import QObject, Signal

_NO_WINDOW_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _query_gpu() -> list:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=_NO_WINDOW_FLAGS,
        )
    except Exception:  # noqa: BLE001 - GPU 조회 실패는 치명적이지 않으므로 조용히 빈 목록 반환
        return []
    if result.returncode != 0:
        return []

    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        name, total, used, free, util, driver, cap = parts
        try:
            gpus.append(
                {
                    "name": name,
                    "vram_total_mb": int(float(total)),
                    "vram_used_mb": int(float(used)),
                    "vram_free_mb": int(float(free)),
                    "utilization_pct": int(float(util)),
                    "driver_version": driver,
                    "compute_capability": cap,
                }
            )
        except ValueError:
            continue
    return gpus


def _query_cpu(cpu_percent: float) -> dict:
    return {
        "physical_cores": psutil.cpu_count(logical=False) or 0,
        "logical_cores": psutil.cpu_count(logical=True) or 0,
        "usage_pct": cpu_percent,
    }


def _query_ram() -> dict:
    vm = psutil.virtual_memory()
    return {
        "total_gb": round(vm.total / (1024**3), 1),
        "available_gb": round(vm.available / (1024**3), 1),
        "used_pct": vm.percent,
    }


def is_wsl() -> bool:
    """지금 실행 중인 프로세스가 WSL(WSL1/2) 안인지 감지한다.

    /proc/version에 "microsoft"가 들어있는지로 판단한다 - WSL 커널이 자기 버전 문자열에
    항상 이 표시를 남기는, 여러 배포판/버전에서 두루 통하는 표준적인 감지 방법이다.
    trainer.py가 WSL2 특유의 NCCL 문제(cuMem API 미지원 등)를 우회하는 환경변수를 넣을지
    결정할 때 쓴다.
    """
    if os.name != "posix":
        return False
    try:
        with open("/proc/version", "r", encoding="utf-8", errors="ignore") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def detect_gpu_count() -> int:
    """nvidia-smi로 현재 이 PC에 실제로 붙어 있는 GPU 개수를 감지한다.

    trainer.py가 멀티 GPU 학습 시 몇 개의 프로세스를 띄울지 정할 때 쓴다 - 특정 GPU 개수나
    번호를 하드코딩하지 않고 매번 다시 감지해서, 다른 PC로 옮기거나 GPU 구성이 바뀌어도
    그대로 동작하도록 한다.
    """
    return len(_query_gpu())


def collect_snapshot() -> dict:
    # psutil.cpu_percent(interval=0.2)가 200ms 블로킹하는 것까지 포함해서, 이 함수 전체를
    # 백그라운드 스레드에서만 호출해야 한다(SystemHeartbeat가 그렇게 한다).
    cpu_percent = psutil.cpu_percent(interval=0.2)
    return {
        "gpus": _query_gpu(),
        "cpu": _query_cpu(cpu_percent),
        "ram": _query_ram(),
    }


def format_snapshot(snapshot: dict) -> str:
    """챗봇 프롬프트 컨텍스트에 넣을 한국어 텍스트로 포맷한다."""
    if not snapshot:
        return ""

    lines = ["[시스템 정보 (실시간)]"]

    gpus = snapshot.get("gpus") or []
    if gpus:
        for i, gpu in enumerate(gpus):
            lines.append(
                f"- GPU {i}: {gpu['name']} (드라이버 {gpu['driver_version']}, "
                f"compute capability {gpu['compute_capability']})"
            )
            lines.append(
                f"  VRAM: {gpu['vram_used_mb']:,}MB 사용 중 / {gpu['vram_total_mb']:,}MB 전체 "
                f"(여유 {gpu['vram_free_mb']:,}MB), GPU 사용률 {gpu['utilization_pct']}%"
            )
    else:
        lines.append("- GPU 정보를 가져올 수 없습니다 (nvidia-smi 없음).")

    cpu = snapshot.get("cpu") or {}
    if cpu:
        lines.append(
            f"- CPU: 물리 코어 {cpu.get('physical_cores', '?')}개 / "
            f"논리 코어 {cpu.get('logical_cores', '?')}개, 현재 사용률 {cpu.get('usage_pct', '?')}%"
        )

    ram = snapshot.get("ram") or {}
    if ram:
        lines.append(
            f"- RAM: {ram.get('available_gb', '?')}GB 사용 가능 / {ram.get('total_gb', '?')}GB 전체 "
            f"(사용률 {ram.get('used_pct', '?')}%)"
        )

    return "\n".join(lines)


class SystemHeartbeat(QObject):
    """일정 간격으로 시스템 정보를 백그라운드에서 갱신하는 하트비트."""

    updated = Signal(dict)

    def __init__(self, interval_seconds: float = 5.0, parent=None):
        super().__init__(parent)
        self._interval = interval_seconds
        self._snapshot: dict = {}
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop_event.is_set():
            try:
                snapshot = collect_snapshot()
            except Exception:  # noqa: BLE001 - 하트비트는 절대 죽으면 안 된다
                snapshot = {}
            if snapshot:
                self._snapshot = snapshot
                self.updated.emit(snapshot)
            self._stop_event.wait(self._interval)

    def latest(self) -> dict:
        return self._snapshot

    def stop(self):
        self._stop_event.set()
