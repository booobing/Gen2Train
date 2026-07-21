"""파라미터 dict를 accelerate launch 커맨드로 조립하고, 학습 프로세스를 실행/모니터링한다.

kohya_ss의 kohya_gui/class_command_executor.py와 kohya_gui/class_accelerate_launch.py가
subprocess를 다루던 방식(별도 토큰으로 인자 구성, psutil로 프로세스 트리 종료)을 그대로 계승한다.
"""
import re
from pathlib import Path
from typing import Optional

import psutil
from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from .. import settings
from . import arg_introspect, dataset_prep, log_translate

# train_network.py의 tqdm(desc="steps") 출력: "steps:  34%|███▍ | 170/500 [02:15<04:23, 1.25it/s, avr_loss=0.08]"
STEP_LINE_RE = re.compile(r"steps:\s*\d+%\|.*?\|\s*(\d+)/(\d+)\s*\[")
# accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}\n") 출력
EPOCH_LINE_RE = re.compile(r"^epoch\s+(\d+)/(\d+)")


class TrainingError(Exception):
    pass


def _format_arg(spec: arg_introspect.ArgSpec, value) -> list[str]:
    if spec.kind == "bool":
        return [spec.flag] if bool(value) else []
    if value is None or value == "":
        return []
    if spec.nargs in ("*", "+"):
        parts = str(value).split()
        return [spec.flag, *parts] if parts else []
    return [spec.flag, str(value)]


def build_command(
    model_type: str,
    pretrained_model_path: str,
    train_data_dir: Path,
    output_dir: Path,
    output_name: str,
    params: dict,
    reg_data_dir: Optional[Path] = None,
) -> tuple[str, list[str]]:
    """(program, args) 튜플을 반환한다. QProcess.start(program, args)에 그대로 넘기면 된다."""
    app_settings = settings.load_settings()
    specs = arg_introspect.get_arg_specs(model_type)
    script_path = settings.SD_SCRIPTS_DIR / f"{arg_introspect.SCRIPT_MODULES[model_type]}.py"

    mixed_precision = params.get("mixed_precision") or "fp16"
    num_cpu_threads = app_settings.get("num_cpu_threads_per_process", 2)

    args = [
        "launch",
        "--num_cpu_threads_per_process",
        str(num_cpu_threads),
        "--mixed_precision",
        str(mixed_precision),
        str(script_path),
        "--pretrained_model_name_or_path",
        str(pretrained_model_path),
        "--train_data_dir",
        str(train_data_dir),
        "--output_dir",
        str(output_dir),
        "--output_name",
        str(output_name),
        "--network_module",
        "networks.lora",
        "--caption_extension",
        dataset_prep.CAPTION_EXTENSION,
    ]
    if reg_data_dir is not None:
        args += ["--reg_data_dir", str(reg_data_dir)]

    for spec in specs:
        if spec.dest not in params:
            continue
        args.extend(_format_arg(spec, params[spec.dest]))

    return app_settings["accelerate_path"], args


def _quote_for_log(token: str) -> str:
    return f'"{token}"' if " " in token else token


class TrainerProcess(QObject):
    """QProcess로 accelerate launch를 실행하고 로그/진행률을 시그널로 내보낸다."""

    log_line = Signal(str)
    progress = Signal(int, int, int, int)  # step, total_steps, epoch, total_epochs
    finished = Signal(bool, str)  # success, message

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process: Optional[QProcess] = None
        self._current_epoch = 0
        self._total_epochs = 0

    def is_running(self) -> bool:
        return self._process is not None and self._process.state() != QProcess.NotRunning

    def start(self, program: str, args: list[str], cwd: Optional[Path] = None) -> None:
        if self.is_running():
            raise TrainingError("이미 학습이 진행 중입니다.")

        self._current_epoch = 0
        self._total_epochs = 0

        process = QProcess(self)
        process.setProgram(program)
        process.setArguments(args)
        process.setProcessChannelMode(QProcess.MergedChannels)

        # sd-scripts 소스에는 영/일 이중언어 로그 메시지가 섞여 있는데, 한국어 Windows의 기본
        # 콘솔 코드페이지(cp949)로는 일부 한자를 표현할 수 없어 UnicodeEncodeError로 죽는다.
        # PYTHONIOENCODING은 sys.stdout/stderr의 인코딩만 UTF-8로 바꿔서 이 문제를 해결한다.
        #
        # 주의: PYTHONUTF8=1은 여기 쓰면 안 된다. 이건 locale.getpreferredencoding()까지
        # 전역으로 UTF-8로 바꿔버려서, git.exe/nvidia-smi처럼 실제로는 cp949로 출력하는
        # 외부 네이티브 도구의 출력을 읽는 다른 subprocess 호출(예: DataLoader 워커 초기화,
        # sd-scripts가 커밋 해시를 얻으려는 git 호출 등)까지 UTF-8로 잘못 디코딩하게 만들어
        # UnicodeDecodeError를 유발한다. 실제로 이 때문에 DataLoader 워커가 계속 죽었다 다시
        # 뜨면서 avr_loss=nan, 스텝당 수십 초라는 심각한 성능 저하로 이어진 적이 있다.
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        # sd-scripts는 rich 콘솔 핸들러로 로그를 찍는데, 실제 터미널이 아니면 기본 80컬럼으로
        # 줄바꿈해서 메시지 하나가 여러 줄로 쪼개진다. 로그 번역/파싱이 한 줄 단위로 매칭하므로
        # 폭을 넉넉히 줘서 각 메시지가 한 줄에 온전히 나오게 한다.
        env.insert("COLUMNS", "300")
        env.insert("LINES", "50")
        process.setProcessEnvironment(env)

        if cwd is not None:
            process.setWorkingDirectory(str(cwd))
        process.readyReadStandardOutput.connect(self._on_output)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_error)

        self._process = process
        self.log_line.emit("$ " + " ".join(_quote_for_log(t) for t in [program, *args]))
        process.start()

    def stop(self) -> None:
        if not self.is_running():
            return
        pid = self._process.processId()
        try:
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
        except psutil.NoSuchProcess:
            pass
        self.log_line.emit("[Gen2Train] 학습 중지를 요청했습니다.")

    def _on_output(self) -> None:
        if not self._process:
            return
        data = bytes(self._process.readAllStandardOutput())
        text = data.decode("utf-8", errors="replace")
        for raw_line in text.splitlines():
            line = raw_line.rstrip("\r")
            if not line:
                continue
            # 진행률 파싱은 원문 기준(정규식이 원문 형태를 가정), 화면 표시만 한국어로 번역한다.
            self.log_line.emit(log_translate.translate_line(line))
            self._parse_progress(line)

    def _parse_progress(self, line: str) -> None:
        m = EPOCH_LINE_RE.search(line.strip())
        if m:
            self._current_epoch = int(m.group(1))
            self._total_epochs = int(m.group(2))
            self.progress.emit(0, 0, self._current_epoch, self._total_epochs)
            return
        m = STEP_LINE_RE.search(line)
        if m:
            step, total = int(m.group(1)), int(m.group(2))
            self.progress.emit(step, total, self._current_epoch, self._total_epochs)

    def _on_error(self, _error) -> None:
        if self._process is None:
            return
        self.log_line.emit(f"[Gen2Train] 프로세스 오류: {self._process.errorString()}")

    def _on_finished(self, exit_code: int, exit_status) -> None:
        success = exit_code == 0 and exit_status == QProcess.NormalExit
        message = "학습이 완료되었습니다." if success else f"학습이 종료되었습니다 (exit code {exit_code})."
        self.log_line.emit(f"[Gen2Train] {message}")
        self.finished.emit(success, message)
        self._process = None
