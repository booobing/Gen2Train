"""Gen2Train 메인 앱 설치 스크립트.

torch/accelerate/transformers/diffusers 등 학습에 필요한 패키지와 PySide6/psutil(UI/하트비트용)을
설치한다. kohya_ss의 공유 venv가 실제로 동작하면 그걸 재사용하고(이미 torch 등이 깔려 있을
가능성이 높아 다운로드를 아낀다), 없거나 망가져 있으면(예: 다른 PC에서 복사된 venv라
pyvenv.cfg가 그 PC의 경로를 가리키는 경우) 이 프로젝트 전용 venv(Gen2Train\\venv)를 새로
만든다. 어떤 PC에서 실행해도 동작하도록 Python/CUDA 위치를 하드코딩하지 않고 그때그때 탐색한다.

사용법: Windows는 setup.bat, Linux/WSL2는 setup.sh를 실행하거나, 이 파일을 아무
python으로나 직접 실행. run.bat/run.sh는 필요한 패키지가 없으면 이 스크립트를 자동으로
실행한다.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

IS_WINDOWS = os.name == "nt"

BASE_DIR = Path(__file__).resolve().parent
VENV_DIR = BASE_DIR / "venv"
# kohya_ss 공유 venv는 Windows용 venv라 Linux/WSL2에서는 재사용할 수 없다 - 그쪽은 항상 이
# 프로젝트 전용 venv(VENV_DIR)를 새로 만든다 (ensure_venv 참고).
SHARED_VENV_DIR = BASE_DIR.parent / "kohya_ss" / "kohya_ss" / "venv"
SD_SCRIPTS_REQUIREMENTS = BASE_DIR / "sd-scripts" / "requirements.txt"
EXTRA_REQUIREMENTS = BASE_DIR / "requirements-extra.txt"

# chat_backend/cuda_discovery.py의 CUDA 탐색 로직을 그대로 재사용한다 - 같은 로직을
# 여기에 또 하드코딩하지 않는다.
sys.path.insert(0, str(BASE_DIR / "chat_backend"))
from cuda_discovery import find_cuda_path  # noqa: E402

# PyTorch 공식 wheel 인덱스가 실제로 게시하는 CUDA 태그(2026-07 확인: cu118/121/124/126/128
# 전부 존재). 정확한 마이너 버전이 없어도 같은 메이저 버전 안에서는 최신 CUDA 드라이버가
# 이전 마이너 버전용으로 빌드된 바이너리를 대체로 문제없이 실행한다(CUDA의 마이너 버전
# 순방향 호환성).
KNOWN_TORCH_CUDA_TAGS = {
    12: ["cu128", "cu126", "cu124", "cu121"],
    11: ["cu118"],
}


# sd-scripts(kohya_ss)의 requirements.txt는 pytorch-lightning==1.9.0/bitsandbytes==0.44.0처럼
# 몇 년 된 버전을 그대로 고정하고 있어, 너무 최신인 Python(3.13+)에서는 그 버전들의 prebuilt
# wheel이 아예 없어 소스 빌드로 새다가 실패하기 쉽다(numpy<=2.0도 마찬가지). Windows 쪽
# 기본값도 3.11이므로 이 범위를 "검증된" 버전으로 취급한다.
SUPPORTED_PY_VERSIONS = {(3, 10), (3, 11)}


def log(msg: str) -> None:
    print(f"[setup] {msg}", flush=True)


def run(cmd, **kwargs) -> subprocess.CompletedProcess:
    log("실행: " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, **kwargs)


def _python_version(path: str) -> tuple | None:
    try:
        result = subprocess.run(
            [path, "-c", "import sys; print(sys.version_info[0], sys.version_info[1])"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        major, minor = result.stdout.split()
        return (int(major), int(minor))
    except ValueError:
        return None


def find_system_python() -> str:
    """venv를 새로 만들 때 쓸 시스템 Python을 찾는다. 특정 사용자/설치 경로를 가정하지 않는다."""
    if IS_WINDOWS:
        for py_args in (["py", "-3.11"], ["py", "-3"]):
            try:
                result = subprocess.run(
                    py_args + ["-c", "import sys; print(sys.executable)"],
                    capture_output=True, text=True, timeout=10,
                )
            except (OSError, subprocess.SubprocessError):
                continue
            if result.returncode == 0:
                path = result.stdout.strip()
                if path and Path(path).exists():
                    return path
    else:
        for candidate in ("python3.11", "python3.10"):
            found = shutil.which(candidate)
            if found:
                return found

    # 여기까지 왔으면 정확히 3.10/3.11로 이름 붙은 인터프리터를 못 찾은 것이다. PATH의
    # python3/python으로 그럭저럭 넘어갈 수는 있지만(예: conda base 환경의 python3가 3.14인
    # 경우), sd-scripts의 오래된 고정 버전 패키지들이 빌드 실패할 가능성이 높으므로 그냥
    # 조용히 쓰지 않고 미리 경고한다.
    found = shutil.which("python") or shutil.which("python3")
    if found:
        version = _python_version(found)
        if version is not None and version not in SUPPORTED_PY_VERSIONS:
            log(f"경고: 시스템에서 찾은 Python이 {version[0]}.{version[1]}입니다 (검증된 버전: 3.10/3.11).")
            log(
                "sd-scripts가 고정한 오래된 패키지 버전들(numpy<=2.0, pytorch-lightning==1.9.0 등)은 "
                "이보다 새 Python에서 미리 빌드된 wheel이 없어 소스 빌드가 필요할 수 있고, 컴파일러가 "
                "없으면 그 자리에서 실패합니다."
            )
            if not IS_WINDOWS:
                log(
                    "Ubuntu/WSL2라면 Python 3.11을 설치한 뒤 다시 실행해주세요: "
                    "sudo apt update && sudo apt install -y python3.11 python3.11-venv "
                    "(패키지가 없으면: sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt update "
                    "&& sudo apt install -y python3.11 python3.11-venv)"
                )
            log("일단 이 Python으로 계속 진행합니다...")
        return found
    return sys.executable


def shared_venv_python() -> str:
    """kohya_ss 공유 venv의 python.exe 경로. 파일 존재만이 아니라 실제로 동작하는지까지
    확인한다 - 다른 PC에서 복사된 venv는 pyvenv.cfg가 원래 PC의 경로를 가리켜 깨져 있을
    수 있다(run.bat과 동일한 이유). 이 공유 venv 자체가 Windows용이라 Linux/WSL2에서는
    애초에 시도하지 않는다."""
    if not IS_WINDOWS:
        return ""
    candidate = SHARED_VENV_DIR / "Scripts" / "python.exe"
    if not candidate.exists():
        return ""
    try:
        result = subprocess.run(
            [str(candidate), "-c", "print(1)"], capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return str(candidate) if result.stdout.strip() == "1" else ""


def venv_python() -> Path:
    if IS_WINDOWS:
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def ensure_venv() -> str:
    shared = shared_venv_python()
    if shared:
        log(f"kohya_ss 공유 venv를 재사용합니다: {shared}")
        return shared

    if venv_python().exists():
        version = _python_version(str(venv_python()))
        if version is not None and version not in SUPPORTED_PY_VERSIONS:
            # 예전에 이 venv를 만들 때 검증되지 않은 Python(예: conda base의 3.14)이 잡혀서
            # 만들어진 경우. 그대로 재사용하면 numpy 등 오래된 고정 버전 패키지가 계속
            # 소스 빌드로 새다 실패하므로, 조용히 재사용하지 않고 지우고 새로 만든다.
            log(
                f"기존 venv({VENV_DIR})가 검증되지 않은 Python {version[0]}.{version[1]}로 "
                "만들어져 있습니다. 삭제하고 다시 만듭니다..."
            )
            shutil.rmtree(VENV_DIR)
        else:
            log(f"기존 venv를 재사용합니다: {VENV_DIR}")
            return str(venv_python())

    log("kohya_ss 공유 venv가 없거나 동작하지 않아 이 프로젝트 전용 venv를 새로 만듭니다...")
    py = find_system_python()
    log(f"시스템 Python 사용: {py}")
    run([py, "-m", "venv", str(VENV_DIR)], check=True)
    return str(venv_python())


def package_importable(py: str, module: str) -> bool:
    result = subprocess.run([py, "-c", f"import {module}"], capture_output=True, text=True)
    return result.returncode == 0


def install_torch(py: str) -> None:
    # torch뿐 아니라 torchvision도 확인한다 - sd-scripts/library/utils.py와 train_util.py가
    # torchvision을 직접 임포트하는데, torch만 있고 torchvision이 없는 상태(예: 이전 실행이
    # 중간에 실패한 경우)에서도 "이미 설치됨"으로 잘못 판단해 건너뛰지 않도록 한다.
    if package_importable(py, "torch") and package_importable(py, "torchvision"):
        log("torch/torchvision이 이미 설치되어 있습니다.")
        return

    run([py, "-m", "pip", "install", "-q", "-U", "pip"], check=False)

    # torch와 torchvision은 반드시 같은 pip 호출로 함께 설치해야 서로 호환되는 버전 조합을
    # pip이 알아서 골라준다 (공식 PyTorch 설치 안내와 동일한 방식) - 따로따로 설치하면
    # 버전이 어긋나 임포트 시점에야 문제가 드러날 수 있다.
    cuda_path = find_cuda_path()
    if cuda_path:
        match = re.search(r"v(\d+)\.(\d+)", Path(cuda_path).name)
        major = int(match.group(1)) if match else None
        tags = KNOWN_TORCH_CUDA_TAGS.get(major, [])
        for tag in tags:
            log(f"CUDA 툴킷 발견({cuda_path}). PyTorch/torchvision({tag}) 설치를 시도합니다...")
            result = run(
                [py, "-m", "pip", "install", "torch", "torchvision",
                 "--index-url", f"https://download.pytorch.org/whl/{tag}"],
                check=False,
            )
            if result.returncode == 0:
                return
        log("모든 CUDA 태그로 PyTorch GPU 설치를 시도했지만 실패했습니다. CPU 전용으로 설치합니다...")
    else:
        log("CUDA 툴킷을 찾지 못했습니다. CPU 전용 PyTorch를 설치합니다 (느리지만 항상 동작함)...")

    run([py, "-m", "pip", "install", "torch", "torchvision"], check=True)


def install_requirements(py: str) -> None:
    if not IS_WINDOWS and not (shutil.which("cc") or shutil.which("gcc") or shutil.which("g++")):
        # sd-scripts requirements의 일부(numpy, safetensors, schedulefree 등)는 prebuilt
        # wheel이 없는 Python 버전/플랫폼 조합이면 소스를 직접 컴파일하려 든다. 컴파일러가
        # 없으면 몇 GB를 내려받은 뒤에야(torch 설치 이후) 실패하게 되므로, 여기서 미리
        # 확인해 바로 안내한다.
        log("경고: C 컴파일러(gcc/g++)를 찾지 못했습니다. 일부 패키지는 소스 빌드가 필요할 수 있습니다.")
        log("Ubuntu/WSL2라면 다음을 먼저 실행해주세요: sudo apt update && sudo apt install -y build-essential")

    if SD_SCRIPTS_REQUIREMENTS.exists():
        log("학습 관련 패키지(accelerate, transformers, diffusers 등)를 설치합니다...")
        # sd-scripts/requirements.txt의 마지막 줄("-e .")은 sd-scripts 자신을 editable
        # 패키지로 설치하라는 kohya_ss의 관례다. pip은 "."을 requirements.txt 파일의
        # 위치가 아니라 "이 pip 프로세스를 호출한 cwd"를 기준으로 해석하므로, cwd를
        # 명시적으로 sd-scripts 폴더로 지정하지 않으면 setup.py를 어느 위치에서
        # 실행했는지에 따라 엉뚱한 곳을 가리켜 실패할 수 있다.
        run(
            [py, "-m", "pip", "install", "-r", str(SD_SCRIPTS_REQUIREMENTS)],
            cwd=str(SD_SCRIPTS_REQUIREMENTS.parent),
            check=True,
        )
    if EXTRA_REQUIREMENTS.exists():
        log("UI/하트비트 관련 패키지(PySide6, psutil)를 설치합니다...")
        run([py, "-m", "pip", "install", "-r", str(EXTRA_REQUIREMENTS)], check=True)


def main() -> None:
    py = ensure_venv()
    install_torch(py)
    install_requirements(py)
    log("설치 완료! run.bat으로 Gen2Train을 실행하세요.")


if __name__ == "__main__":
    main()
