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

사용법: Windows는 setup_chat.bat, Linux/WSL2는 setup_chat.sh를 실행하거나, 이 파일을 아무
python으로나 직접 실행.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from cuda_discovery import find_cuda_path

IS_WINDOWS = os.name == "nt"

BASE_DIR = Path(__file__).resolve().parent.parent
VENV_DIR = BASE_DIR / "venv_chat"
CHAT_BACKEND_DIR = BASE_DIR / "chat_backend"
MODELS_DIR = CHAT_BACKEND_DIR / "models"
BACKEND_CONFIG_PATH = CHAT_BACKEND_DIR / "backend_config.json"

GGUF_REPO = "LGAI-EXAONE/EXAONE-Deep-2.4B-GGUF"
GGUF_FILENAME = "EXAONE-Deep-2.4B-Q4_K_M.gguf"

# abetlen.github.io가 실제로 게시하는 llama-cpp-python 사전빌드 CUDA wheel 태그
# (2026-07 확인: cu121/cu122/cu123/cu124/cu118은 존재, cu126/cu128은 404). 정확한 마이너
# 버전이 없어도 같은 메이저 버전 안에서는 최신 CUDA 드라이버가 예전 마이너 버전용으로 빌드된
# 바이너리를 대체로 문제없이 실행한다(CUDA의 마이너 버전 순방향 호환성) - 그래서 소스 빌드가
# 이 PC의 Visual Studio/CUDA 조합과 근본적으로 안 맞아 전부 실패해도(예: 최신 VS가 이
# CUDA의 컴파일 도구와 바이너리 비호환) 이 목록으로 최신 버전을 유지하며 GPU를 쓸 수 있다.
KNOWN_PREBUILT_CUDA_TAGS = {
    12: ["cu124", "cu123", "cu122", "cu121"],
    11: ["cu118"],
}


def log(msg: str) -> None:
    print(f"[setup_chat] {msg}", flush=True)


def venv_python() -> Path:
    if IS_WINDOWS:
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def run(cmd, **kwargs) -> subprocess.CompletedProcess:
    log("실행: " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, **kwargs)


def find_system_python() -> str:
    """venv_chat을 새로 만들 때 쓸 시스템 Python을 찾는다. 특정 사용자/설치 경로를 가정하지 않는다."""
    if IS_WINDOWS:
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
    else:
        # 1) Linux/WSL2: python3.11을 우선 찾고, 없으면 아무 python3나 찾는다.
        for candidate in ("python3.11", "python3"):
            found = shutil.which(candidate)
            if found:
                return found
    # 2) PATH에 등록된 python
    found = shutil.which("python") or shutil.which("python3")
    if found:
        return found
    # 3) 마지막 수단: 지금 이 스크립트를 실행 중인 인터프리터
    return sys.executable


def short_build_temp_dir() -> Path:
    # llama.cpp 소스 트리 안에는 웹 UI(tools/ui/src/lib/components/...)의 깊게 중첩된 경로가
    # 있어서 Windows의 260자 경로 제한에 아슬아슬하게(실측 약 261자) 걸려 압축 해제가 실패할
    # 수 있다. pip이 자기 내부적으로 붙이는 "pip-install-XXXXXXXX\llama-cpp-python_<32자 해시>"
    # 부분(약 70자)은 줄일 수 없으므로, 우리가 통제할 수 있는 접두 폴더 이름을 최대한
    # 짧게(2글자) 잡아 여유를 최대한 확보한다. 어떤 드라이브에 이 프로젝트가 있든(꼭 C:가
    # 아니어도) 그 드라이브 루트에 만든다.
    drive = Path(__file__).resolve().drive or "C:"
    return Path(drive + "\\_g")


def find_vs_installations() -> list:
    """C++ 빌드 도구가 설치된 모든 Visual Studio(Build Tools 포함)를 최신순으로 찾는다.

    llama-cpp-python은 Windows에서 미리 빌드된 wheel이 없어서 항상 소스를 C/C++로 컴파일해야
    하는데, 이때 CMake가 쓰는 nmake/cl.exe는 Visual Studio(또는 Build Tools)를 설치해도 기본
    PATH에는 없다 - "Developer Command Prompt"를 통해서만 잡힌다. vswhere.exe는 VS2017 이후
    아무 Visual Studio 제품(Build Tools만 설치해도 포함)에나 같이 깔리는 표준 검색 도구다.

    최신 VS가 설치된 CUDA 버전이 공식 지원하는 컴파일러 범위보다 너무 새로우면(예: CUDA
    12.6 + VS 2026 preview) nvcc 내부 도구(cudafe++ 등)가 그 MSVC가 만든 코드를 파싱하다
    충돌하는, 플래그로도 우회 불가능한 실제 바이너리 비호환이 날 수 있다. 같은 PC에 여러 VS
    버전이 나란히 설치돼 있으면 전부 후보로 반환해서(최신 우선) 하나씩 재시도할 수 있게 한다.
    반환값: [(버전 문자열, vcvarsall.bat 경로), ...] - 최신 버전이 먼저 온다.
    """
    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.exists():
        return []
    try:
        result = subprocess.run(
            [
                str(vswhere), "-products", "*",
                "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-format", "json",
            ],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    try:
        instances = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []

    def version_key(inst: dict) -> tuple:
        nums = []
        for part in str(inst.get("installationVersion", "")).split("."):
            try:
                nums.append(int(part))
            except ValueError:
                break
        return tuple(nums)

    instances.sort(key=version_key, reverse=True)

    found = []
    for inst in instances:
        install_path = inst.get("installationPath")
        if not install_path:
            continue
        vcvarsall = Path(install_path) / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
        if vcvarsall.exists():
            found.append((str(inst.get("installationVersion", "?")), str(vcvarsall)))
    return found


def find_vcvarsall() -> str:
    """가장 최신 Visual Studio의 vcvarsall.bat 경로. (CUDA 호환성 재시도용 여러 후보가
    필요하면 find_vs_installations()를 쓴다.)"""
    installations = find_vs_installations()
    return installations[0][1] if installations else ""


def find_ninja() -> str:
    """Ninja 빌드 도구 경로를 찾는다.

    CMake의 "Visual Studio" MSBuild 생성기는 VS 버전별 CUDA 통합 파일이 필요해 깨지기 쉽고,
    "NMake Makefiles"는 그 문제는 없지만 컴파일을 한 번에 한 파일씩 순차로만 처리해 ggml-cuda처럼
    무거운 .cu 파일이 많은 프로젝트에서는 매우 느리다. Ninja는 두 문제 모두 없다 - CUDA 통합
    파일이 필요 없고, CPU 코어 수만큼 자동으로 병렬 컴파일한다. PATH에 없어도 Visual
    Studio/Build Tools를 설치하면 거의 항상 같이 번들로 들어있다.
    """
    found = shutil.which("ninja")
    if found:
        return found
    vcvarsall = find_vcvarsall()
    if vcvarsall:
        vs_root = Path(vcvarsall).parent.parent.parent.parent
        candidate = vs_root / "Common7" / "IDE" / "CommonExtensions" / "Microsoft" / "CMake" / "Ninja" / "ninja.exe"
        if candidate.exists():
            return str(candidate)
    return ""


def msvc_build_env(base_env: dict, vcvarsall: str = None) -> dict | None:
    """vcvarsall.bat x64를 실행해 얻은 PATH/INCLUDE/LIB 등을 base_env에 합쳐 반환한다.

    vcvarsall을 지정하지 않으면: nmake/cl.exe가 이미 PATH에 있으면(개발자 명령 프롬프트에서
    직접 실행한 경우 등) 그대로 base_env를 반환하고, 없으면 가장 최신 VS를 자동으로 찾는다.
    (특정 VS 후보를 강제하고 싶으면 - 예: 최신 VS가 CUDA와 호환되지 않아 다른 버전을 재시도할
    때 - vcvarsall을 명시적으로 넘긴다. 이때는 PATH에 이미 다른 VS의 cl.exe가 잡혀 있을 수
    있으므로 "이미 있으면 그대로 쓰기" 지름길을 타지 않는다.)
    vcvarsall.bat을 못 찾으면 None을 반환한다 - 이 경우 소스 빌드 자체가 불가능하므로
    호출하는 쪽에서 설치 안내 메시지를 띄우고 중단해야 한다.
    """
    if vcvarsall is None:
        if shutil.which("nmake", path=base_env.get("PATH")) and shutil.which("cl", path=base_env.get("PATH")):
            return base_env
        vcvarsall = find_vcvarsall()
        if not vcvarsall:
            return None

    # cmd.exe /c "<string with embedded quotes>"를 argv 리스트로 넘기면 subprocess의 자동
    # 인용부호 처리와 cmd.exe의 파싱 규칙이 서로 안 맞아(중첩된 큰따옴표가 리터럴 백슬래시로
    # 깨짐) vcvarsall.bat이 아예 실행되지 않는 문제가 있었다. 임시 .bat 파일을 만들어
    # 그 안에서 호출하면 이런 중첩 인용 문제 자체가 생기지 않는다.
    marker = "___G2T_VCVARS_DONE___"
    fd, bat_path = tempfile.mkstemp(suffix=".bat")
    os.close(fd)
    try:
        Path(bat_path).write_text(
            f'@echo off\r\ncall "{vcvarsall}" x64\r\nif errorlevel 1 exit /b 1\r\necho {marker}\r\nset\r\n',
            encoding="mbcs",
        )
        try:
            result = subprocess.run([bat_path], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return None
    finally:
        Path(bat_path).unlink(missing_ok=True)

    if marker not in result.stdout:
        return None

    env = base_env.copy()
    for line in result.stdout.split(marker, 1)[1].splitlines():
        key, sep, value = line.partition("=")
        if sep and key:
            env[key] = value
    return env


def long_paths_enabled() -> bool:
    """Windows의 긴 경로(LongPathsEnabled) 지원 여부를 레지스트리에서 확인한다.

    최신 llama-cpp-python 소스에는 웹 UI(vendor/llama.cpp/tools/ui/src/lib/components/...)의
    깊게 중첩된 컴포넌트 트리가 같이 들어있어서, TEMP를 아무리 짧게 잡아도(_g2t_build_tmp)
    전체 경로가 Windows 기본 260자 제한을 넘어 pip 압축 해제가 실패하는 경우가 있다.
    이 레지스트리 값이 꺼져 있으면(기본값) 짧은 TEMP로도 해결이 안 되므로 빌드 전에 미리
    확인해서, 매번 똑같이 실패할 다운로드+빌드를 두 번(GPU/CPU) 반복하지 않는다.
    """
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as key:
            value, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
            return bool(value)
    except OSError:
        return False


def ensure_venv() -> None:
    if venv_python().exists():
        log(f"venv_chat이 이미 있습니다: {VENV_DIR}")
        return
    log("venv_chat을 새로 만듭니다...")
    py = find_system_python()
    log(f"시스템 Python 사용: {py}")
    run([py, "-m", "venv", str(VENV_DIR)], check=True)


def _apply_build_tools(env: dict, ninja_path: str) -> dict:
    """generator(Ninja/NMake) 설정을 env에 적용한다. CMake의 기본 생성기 선택 로직이
    환경에 따라(특히 최신/프리뷰 Visual Studio가 설치된 경우) "Visual Studio" MSBuild
    생성기를 고를 수 있는데, 이건 CUDA Toolkit 설치 시 그 정확한 VS 버전용 통합
    파일(BuildCustomizations)이 같이 등록돼 있어야만 동작한다. CUDA를 나중에 설치했거나
    VS를 나중에 새로 깔면 이 매칭이 깨져 "No CUDA toolset found"로 실패한다. Ninja나
    NMake Makefiles로 고정하면 nvcc를 직접 호출해 이 문제 자체가 생기지 않는다. 둘 다 이
    문제를 피하지만 NMake는 파일을 한 번에 하나씩만 순차 컴파일해 ggml-cuda처럼 무거운
    .cu 파일이 많은 프로젝트에서는 몹시 느리므로, 있으면 Ninja를 우선한다."""
    if ninja_path:
        env["CMAKE_GENERATOR"] = "Ninja"
        env["CMAKE_MAKE_PROGRAM"] = ninja_path
        env["PATH"] = str(Path(ninja_path).parent) + os.pathsep + env.get("PATH", "")
    else:
        env["CMAKE_GENERATOR"] = "NMake Makefiles"
    return env


def _try_gpu_build(py: str, build_env: dict, cuda_path: str, extra_cuda_flags: str = None) -> bool:
    env = build_env.copy()
    env["CUDA_PATH"] = cuda_path
    env["PATH"] = str(Path(cuda_path) / "bin") + os.pathsep + env.get("PATH", "")
    cmake_args = "-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=native"
    if extra_cuda_flags:
        cmake_args += f" -DCMAKE_CUDA_FLAGS={extra_cuda_flags}"
    env["CMAKE_ARGS"] = cmake_args
    result = run([py, "-m", "pip", "install", "llama-cpp-python", "--no-cache-dir"], env=env)
    return result.returncode == 0


def _try_prebuilt_cuda_wheel(py: str, cuda_path: str) -> bool:
    """이 PC의 Visual Studio/CUDA 조합으로는 소스 빌드가 근본적으로 안 될 때(예: 최신 VS의
    컴파일러가 이 CUDA 버전 내부 도구와 바이너리 비호환) 마지막으로 커뮤니티 사전빌드
    wheel을 시도한다. CUDA 메이저 버전만 보고 그 안에서 실제 게시된 태그를 최신순으로
    시도하며, --only-binary로 wheel이 없으면 소스 빌드로 새지 않고 바로 실패하게 한다."""
    match = re.search(r"v(\d+)\.(\d+)", Path(cuda_path).name)
    if not match:
        return False
    major = int(match.group(1))
    tags = KNOWN_PREBUILT_CUDA_TAGS.get(major, [])
    if not tags:
        return False

    for tag in tags:
        log(f"커뮤니티 사전빌드 CUDA wheel({tag})을 시도합니다...")
        result = run(
            [
                py, "-m", "pip", "install", "llama-cpp-python",
                "--extra-index-url", f"https://abetlen.github.io/llama-cpp-python/whl/{tag}",
                "--prefer-binary", "--only-binary=llama-cpp-python",
                "--force-reinstall", "--no-cache-dir",
            ],
            check=False,
        )
        if result.returncode == 0:
            log(f"사전빌드 wheel({tag}) 설치 성공.")
            return True
    return False


def install_llama_cpp_python() -> str:
    """CUDA 빌드를 시도하고, 실패하면 CPU 전용으로 폴백한다. 실제 설치된 backend 이름을 반환."""
    py = str(venv_python())
    run([py, "-m", "pip", "install", "-q", "-U", "pip"], check=False)

    if IS_WINDOWS:
        return _install_llama_cpp_python_windows(py)
    return _install_llama_cpp_python_linux(py)


def _install_llama_cpp_python_linux(py: str) -> str:
    """Linux/WSL2에서는 Windows처럼 MSVC 툴체인을 찾아 헤맬 필요가 없다 - cmake와 gcc/g++만
    있으면 CMake가 알아서 기본 생성기(대개 Unix Makefiles, ninja가 PATH에 있으면 그쪽)로
    빌드한다. Windows 쪽 코드에 있는 MSBuild-CUDA 통합 파일 문제나 260자 경로 제한도 Linux에는
    없다. CUDA 지원 여부는 nvcc/CUDA 툴킷 유무로만 판단해 CMAKE_ARGS를 켜고 끈다.
    """
    have_cmake = bool(shutil.which("cmake"))
    have_compiler = bool(shutil.which("cc") or shutil.which("gcc") or shutil.which("g++"))
    if not have_cmake or not have_compiler:
        missing = [name for name, ok in (("cmake", have_cmake), ("gcc/g++", have_compiler)) if not ok]
        log(f"빌드 도구가 없습니다: {', '.join(missing)}.")
        log("Ubuntu/WSL2라면 다음을 실행한 뒤 다시 시도해주세요:")
        log("  sudo apt update && sudo apt install -y build-essential cmake")
        raise RuntimeError("빌드 도구(cmake/gcc)가 설치되어 있지 않습니다.")

    env = os.environ.copy()
    cuda_path = find_cuda_path()
    if cuda_path:
        log(f"CUDA 툴킷 발견({cuda_path}). GPU 가속 빌드를 시도합니다...")
        env["CUDA_PATH"] = cuda_path
        env["PATH"] = str(Path(cuda_path) / "bin") + os.pathsep + env.get("PATH", "")
        env["CMAKE_ARGS"] = "-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=native"
        result = run([py, "-m", "pip", "install", "llama-cpp-python", "--no-cache-dir"], env=env, check=False)
        if result.returncode == 0:
            return "gguf-cuda"
        log("GPU 가속 빌드 실패. CPU 전용으로 재시도합니다...")
    else:
        log(
            "CUDA 툴킷을 찾지 못했습니다. WSL2라면 Windows 쪽에는 NVIDIA 드라이버만 설치하고, "
            "WSL2 안에서는 CUDA 툴킷을 따로 설치해야 합니다 "
            "(https://developer.nvidia.com/cuda/wsl 안내 참고, 드라이버는 설치하지 말 것). "
            "CPU 전용으로 설치합니다..."
        )

    env.pop("CMAKE_ARGS", None)
    result = run([py, "-m", "pip", "install", "llama-cpp-python", "--no-cache-dir"], env=env)
    if result.returncode != 0:
        raise RuntimeError("llama-cpp-python 설치에 실패했습니다. 수동으로 확인이 필요합니다.")
    return "gguf-cpu"


def _install_llama_cpp_python_windows(py: str) -> str:
    # llama-cpp-python은 Windows용 사전빌드 wheel이 없어 pip이 항상 소스를 C/C++로 컴파일한다.
    vs_installations = find_vs_installations()
    if not vs_installations:
        log("MSVC C++ 빌드 도구(nmake/cl.exe)를 찾지 못했습니다.")
        log("llama-cpp-python은 Windows용 사전빌드 wheel이 없어 반드시 로컬에서 컴파일해야 합니다.")
        log(
            "Visual Studio Build Tools를 설치한 뒤 다시 실행해주세요 "
            "(https://visualstudio.microsoft.com/downloads/ 에서 'Build Tools for Visual Studio' 다운로드, "
            "설치 화면에서 'C++를 사용한 데스크톱 개발' 워크로드를 선택)."
        )
        raise RuntimeError("MSVC 빌드 도구(Visual Studio Build Tools)가 설치되어 있지 않습니다.")

    if not long_paths_enabled():
        # 짧은 TEMP 접두(short_build_temp_dir)로 대부분의 경우는 260자 제한 안에 들어오지만,
        # 그래도 딱 걸리는 경우를 대비해 미리 안내만 해둔다 - 여기서 바로 중단하지는 않는다.
        log("참고: Windows 긴 경로(Long Path) 지원이 꺼져 있습니다 (기본값).")
        log(
            "빌드 중 파일 경로가 너무 길다는 오류(No such file or directory 등)가 나오면, "
            "관리자 권한 PowerShell에서 아래 명령을 한 번 실행한 뒤 setup_chat.bat을 다시 실행해주세요:"
        )
        log(
            '  New-ItemProperty -Path "HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem" '
            '-Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force'
        )

    short_tmp = short_build_temp_dir()
    short_tmp.mkdir(exist_ok=True)
    ninja_path = find_ninja()
    if ninja_path:
        log(f"Ninja 빌드 도구 발견({ninja_path}). 병렬 컴파일로 빌드합니다.")
    else:
        log("Ninja를 찾지 못해 NMake Makefiles로 빌드합니다 (더 느림, 순차 컴파일).")

    cuda_path = find_cuda_path()
    if cuda_path:
        log(f"CUDA 툴킷 발견({cuda_path}).")

        # 설치된 Visual Studio가 이 CUDA 버전이 공식 지원하는 컴파일러 범위보다 새로우면
        # ("unsupported Microsoft Visual Studio version!" 또는 cudafe++ 등 내부 도구가
        # 실제로 충돌하는 바이너리 비호환) 빌드가 실패할 수 있다. 최신 VS를 먼저 시도하고,
        # 실패하면 -allow-unsupported-compiler로 한 번 더(단순 버전 체크였다면 이걸로
        # 풀린다), 그래도 안 되면(진짜 바이너리 비호환) 이 PC에 나란히 설치된 더 오래된
        # VS 후보로 넘어가며 같은 두 단계를 반복한다.
        for i, (vs_version, vcvarsall) in enumerate(vs_installations):
            build_env = msvc_build_env(os.environ.copy(), vcvarsall=vcvarsall)
            if build_env is None:
                continue
            build_env["TEMP"] = str(short_tmp)
            build_env["TMP"] = str(short_tmp)
            build_env = _apply_build_tools(build_env, ninja_path)

            tag = f"Visual Studio {vs_version} ({i + 1}/{len(vs_installations)})"
            log(f"GPU 가속 빌드 시도 - {tag}...")
            if _try_gpu_build(py, build_env, cuda_path):
                return "gguf-cuda"

            log(f"{tag}: 기본 빌드 실패. -allow-unsupported-compiler 옵션으로 재시도...")
            if _try_gpu_build(py, build_env, cuda_path, extra_cuda_flags="-allow-unsupported-compiler"):
                return "gguf-cuda"

            if i + 1 < len(vs_installations):
                log(f"{tag}: 계속 실패. 이 PC의 다른 Visual Studio 후보로 재시도합니다...")
            else:
                log(f"{tag}: 마지막 후보도 실패했습니다.")

        log("설치된 모든 Visual Studio 조합으로 GPU 가속 빌드를 시도했지만 전부 실패했습니다.")

        # 이 PC의 Visual Studio/CUDA 조합으로는 소스 빌드가 근본적으로 안 되는 경우(진짜
        # 바이너리 비호환)에도, GPU를 포기하지 않고 커뮤니티가 미리 빌드해둔 wheel로 마지막
        # 시도를 한다. 최신 버전(0.3.34 등)을 유지하지 못할 수 있지만(그 태그에 그 버전
        # wheel이 없으면 pip이 자동으로 그 태그에서 구할 수 있는 최신 버전을 고른다), 소스
        # 컴파일 없이 GPU 가속을 쓸 수 있다.
        if _try_prebuilt_cuda_wheel(py, cuda_path):
            return "gguf-cuda"

        log("커뮤니티 사전빌드 wheel도 실패했습니다. CPU 전용으로 안전하게 설치를 진행합니다 "
            "(느리지만 항상 동작함).")
    else:
        log("CUDA 툴킷을 찾지 못했습니다. CPU 전용으로 설치합니다 (느리지만 항상 동작함)...")

    # CPU 폴백: 순수 C 컴파일에도 MSVC 환경은 필요하므로 가장 최신 VS 후보를 그대로 쓴다.
    build_env = msvc_build_env(os.environ.copy(), vcvarsall=vs_installations[0][1])
    if build_env is None:
        raise RuntimeError("MSVC 빌드 환경을 구성하지 못했습니다.")
    build_env["TEMP"] = str(short_tmp)
    build_env["TMP"] = str(short_tmp)
    build_env = _apply_build_tools(build_env, ninja_path)

    result = run([py, "-m", "pip", "install", "llama-cpp-python", "--no-cache-dir"], env=build_env)
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
    # huggingface_hub는 hf_xet(네이티브 Rust 바이너리)이 설치돼 있으면 자동으로 그걸 써서
    # 다운로드를 가속하려 하는데, 회사 PC 등 애플리케이션 제어 정책(WDAC/AppLocker 등)이 걸린
    # 환경에서는 서명되지 않은 이 DLL 로딩 자체가 차단될 수 있다. 파일이 1.6GB 정도로 크지
    # 않으니 굳이 가속을 쓰지 않고 항상 기본 HTTPS 다운로드 경로로 통일해 이 문제를 피한다.
    env = os.environ.copy()
    env["HF_HUB_DISABLE_XET"] = "1"
    run([py, "-c", download_script], check=True, env=env)


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
