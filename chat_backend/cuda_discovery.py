# cuda_discovery.py
"""설치된 CUDA 툴킷 중 가장 최신 버전을 찾는다.

setup_chat_model.py(빌드 시점)와 chat_server.py(실행 시점)가 똑같이 이 함수를 쓴다 -
특정 CUDA 버전이나 경로를 두 곳에 따로 하드코딩하지 않고, 항상 이 한 곳에서만 찾는다.
"""
from pathlib import Path
import os


def find_cuda_path() -> str:
    """표준 설치 경로에 있는 CUDA 툴킷 중 가장 최신 버전을 찾는다.

    CUDA_PATH/CUDA_HOME 환경변수를 무조건 믿지 않는다 - 여러 버전이 함께 설치된 PC에서는 이
    값이 오래된 버전을 가리키고 있을 수 있다(실제로 이 개발 PC도 v11.8/v12.8이 같이 설치돼
    있었는데 CUDA_PATH는 v11.8을 가리켰다). 최신 GPU(Blackwell 등)는 최신 CUDA가 필요하므로
    디렉터리를 스캔해 실제로 존재하는 버전 중 최신을 우선 고르고, 표준 경로 자체가 없을 때만
    환경변수를 마지막 수단으로 쓴다.
    """
    if os.name == "nt":
        return _find_cuda_path_windows()
    return _find_cuda_path_linux()


def _find_cuda_path_windows() -> str:
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


def _find_cuda_path_linux() -> str:
    """Linux(WSL2 포함)의 표준 CUDA 툴킷 설치 경로에서 최신 버전을 찾는다.

    WSL2는 NVIDIA 공식 안내(https://developer.nvidia.com/cuda/wsl)대로 설치해도 네이티브
    Linux와 똑같이 /usr/local/cuda-X.Y에 설치되므로(드라이버는 Windows 쪽에만 있고 WSL2
    안에는 툴킷만 설치) 별도 분기가 필요 없다.
    """
    def version_key(path: Path) -> tuple:
        nums = []
        for part in path.name.split("-")[-1].split("."):
            try:
                nums.append(int(part))
            except ValueError:
                break
        return tuple(nums)

    usr_local = Path("/usr/local")
    if usr_local.exists():
        versions = [p for p in usr_local.iterdir() if p.is_dir() and p.name.startswith("cuda-")]
        if versions:
            return str(max(versions, key=version_key))
        plain = usr_local / "cuda"  # 버전 번호 없는 심볼릭 링크만 있는 경우
        if plain.exists():
            return str(plain)

    for var in ("CUDA_HOME", "CUDA_PATH"):
        env_cuda = os.environ.get(var)
        if env_cuda and Path(env_cuda).exists():
            return env_cuda

    return ""
