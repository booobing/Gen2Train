"""Gen2Train 챗봇(AI 도우미) 백엔드 설치 스크립트.

venv_chat을 만들고 llama-cpp-python + GGUF로 EXAONE-Deep-2.4B를 설치한다.
어떤 PC에서 실행해도 동작하도록, Python/CUDA 위치를 하드코딩하지 않고 그때그때 탐색한다.

원래 계획은 AutoAWQ를 먼저 시도하고 실패하면 GGUF로 자동 전환하는 것이었는데, 개발 중
RTX 5070(Blackwell, compute capability 12.0)에서 시도해보니 AutoAWQ의 사전빌드
커널(autoawq-kernels 0.0.7)이 이 세대를 전혀 지원하지 않아 torch를 Blackwell 미지원
버전(2.3.1)으로 강제 다운그레이드하며 100% 실패했다. 반면 llama-cpp-python은
CMAKE_CUDA_ARCHITECTURES=native로 소스에서 직접 빌드하면 그 GPU에 맞는 네이티브 가속으로
정상 동작한다. 그래서 이 스크립트는 처음부터 GGUF 경로만 시도한다 (chat_server.py도 GGUF
백엔드만 구현되어 있다). CUDA 빌드가 안 되는 환경(CUDA 미설치 등)에서는 자동으로 CPU 전용
설치로 넘어간다 - 느리지만 어떤 PC에서도 최소한 동작은 한다.

만약 다른(구형) GPU에서 AutoAWQ가 실제로 동작한다면, 그건 별도로 chat_server.py에
AWQ 로딩 경로를 추가해야 쓸 수 있다 - 이 스크립트가 자동으로 그렇게 해주지는 않는다.

사용법: Gen2Train\\setup_chat.bat 을 더블클릭하거나, 이 파일을 아무 python으로나 실행.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
VENV_DIR = BASE_DIR / "venv_chat"
CHAT_BACKEND_DIR = BASE_DIR / "chat_backend"
MODELS_DIR = CHAT_BACKEND_DIR / "models"
BACKEND_CONFIG_PATH = CHAT_BACKEND_DIR / "backend_config.json"

GGUF_REPO = "LGAI-EXAONE/EXAONE-Deep-2.4B-GGUF"
GGUF_FILENAME = "EXAONE-Deep-2.4B-Q4_K_M.gguf"


def log(msg: str) -> None:
    print(f"[setup_chat] {msg}", flush=True)


def venv_python() -> Path:
    return VENV_DIR / "Scripts" / "python.exe"


def run(cmd, **kwargs) -> subprocess.CompletedProcess:
    log("실행: " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, **kwargs)


def find_system_python() -> str:
    """venv_chat을 새로 만들 때 쓸 시스템 Python을 찾는다. 특정 사용자/설치 경로를 가정하지 않는다."""
    # 1) Windows Python Launcher로 3.11을 우선 찾고, 없으면 아무 3.x나 찾는다.
    for py_args in (["py", "-3.11"], ["py", "-3"]):
        try:
            result = subprocess.run(
                py_args + ["-c", "import sys; print(sys.executable)"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode == 0:
            path = result.stdout.strip()
            if path and Path(path).exists():
                return path
    # 2) PATH에 등록된 python
    found = shutil.which("python") or shutil.which("python3")
    if found:
        return found
    # 3) 마지막 수단: 지금 이 스크립트를 실행 중인 인터프리터
    return sys.executable


def find_cuda_path() -> str:
    """표준 설치 경로에 있는 CUDA 툴킷 중 가장 최신 버전을 찾는다.

    CUDA_PATH 환경변수를 무조건 믿지 않는다 - 여러 버전이 함께 설치된 PC에서는 이 값이
    오래된 버전을 가리키고 있을 수 있다(실제로 이 개발 PC도 v11.8/v12.8이 같이 설치돼
    있었는데 CUDA_PATH는 v11.8을 가리켰다). 최신 GPU(Blackwell 등)는 최신 CUDA가 필요하므로
    디렉터리를 스캔해 실제로 존재하는 버전 중 최신을 우선 고르고, 표준 경로 자체가 없을 때만
    CUDA_PATH를 마지막 수단으로 쓴다.
    """
    def version_key(path: Path) -> tuple:
        nums = []
        for part in path.name.lstrip("vV").split("."):
            try:
                nums.append(int(part))
            except ValueError:
                break
        return tuple(nums)

    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    cuda_root = Path(program_files) / "NVIDIA GPU Computing Toolkit" / "CUDA"
    if cuda_root.exists():
        versions = [p for p in cuda_root.iterdir() if p.is_dir() and p.name.lower().startswith("v")]
        if versions:
            return str(max(versions, key=version_key))

    env_cuda = os.environ.get("CUDA_PATH")
    if env_cuda and Path(env_cuda).exists():
        return env_cuda

    return ""


def short_build_temp_dir() -> Path:
    # llama.cpp 소스 트리 안에 워낙 깊은 경로가 있어서, Windows의 260자 경로 제한에
    # 걸려 압축 해제가 실패할 수 있다. 어떤 드라이브에 이 프로젝트가 있든(꼭 C:가 아니어도)
    # 그 드라이브 루트에 짧은 임시 폴더를 만들어 우회한다.
    drive = Path(__file__).resolve().drive or "C:"
    return Path(drive + "\\_g2t_build_tmp")


def ensure_venv() -> None:
    if venv_python().exists():
        log(f"venv_chat이 이미 있습니다: {VENV_DIR}")
        return
    log("venv_chat을 새로 만듭니다...")
    py = find_system_python()
    log(f"시스템 Python 사용: {py}")
    run([py, "-m", "venv", str(VENV_DIR)], check=True)


def install_llama_cpp_python() -> str:
    """CUDA 빌드를 시도하고, 실패하면 CPU 전용으로 폴백한다. 실제 설치된 backend 이름을 반환."""
    py = str(venv_python())
    run([py, "-m", "pip", "install", "-q", "-U", "pip"], check=False)

    cuda_path = find_cuda_path()
    if cuda_path:
        log(f"CUDA 툴킷 발견({cuda_path}). GPU 가속 빌드를 시도합니다...")
        env = os.environ.copy()
        env["CUDA_PATH"] = cuda_path
        env["PATH"] = str(Path(cuda_path) / "bin") + os.pathsep + env.get("PATH", "")
        env["CMAKE_ARGS"] = "-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=native"
        short_tmp = short_build_temp_dir()
        short_tmp.mkdir(exist_ok=True)
        env["TEMP"] = str(short_tmp)
        env["TMP"] = str(short_tmp)

        result = run(
            [py, "-m", "pip", "install", "llama-cpp-python", "--no-cache-dir"],
            env=env,
        )
        if result.returncode == 0:
            return "gguf-cuda"
        log("GPU 가속 빌드 실패. CPU 전용으로 다시 시도합니다 (느리지만 항상 동작함)...")
    else:
        log("CUDA 툴킷을 찾지 못했습니다. CPU 전용으로 설치합니다 (느리지만 항상 동작함)...")

    result = run([py, "-m", "pip", "install", "llama-cpp-python", "--no-cache-dir"])
    if result.returncode != 0:
        raise RuntimeError("llama-cpp-python 설치에 실패했습니다. 수동으로 확인이 필요합니다.")
    return "gguf-cpu"


def download_model() -> None:
    py = str(venv_python())
    run([py, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / GGUF_FILENAME
    if model_path.exists():
        log(f"모델이 이미 있습니다: {model_path}")
        return

    log(f"GGUF 모델을 다운로드합니다: {GGUF_REPO}/{GGUF_FILENAME} (약 1.6GB, 몇 분 걸릴 수 있음)")
    download_script = (
        "from huggingface_hub import hf_hub_download; "
        f"hf_hub_download(repo_id='{GGUF_REPO}', filename='{GGUF_FILENAME}', local_dir=r'{MODELS_DIR}')"
    )
    run([py, "-c", download_script], check=True)


def write_backend_config(backend: str) -> None:
    # model_path는 파일명만 저장한다(절대경로 X). chat_server.py가 이 파일과 같은 폴더의
    # models/ 밑에서 찾으므로, Gen2Train 폴더를 통째로 다른 위치/PC로 옮겨도 그대로 동작한다.
    config = {
        "backend": backend,
        "model_path": GGUF_FILENAME,
        "n_gpu_layers": -1 if backend == "gguf-cuda" else 0,
        "n_ctx": 16384,
    }
    BACKEND_CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"backend_config.json 기록 완료: backend={backend}")


def main() -> None:
    ensure_venv()
    backend = install_llama_cpp_python()
    download_model()
    write_backend_config(backend)
    log("설치 완료! Gen2Train 앱을 실행하고 'AI 도우미' 탭을 열어보세요.")


if __name__ == "__main__":
    main()
