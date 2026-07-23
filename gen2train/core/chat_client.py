"""venv_chat의 chat_server.py를 상주 프로세스로 관리하는 Qt 클라이언트.

trainer.py의 TrainerProcess와 같은 QProcess 패턴을 쓰지만, 1회성 학습 실행이 아니라
모델을 한 번 로드해두고 여러 질문을 계속 처리하는 상주 프로세스라는 점이 다르다.
stdin/stdout으로 JSON Lines 프로토콜을 주고받는다 (chat_backend/chat_server.py 참고).
"""
import json
import os
from pathlib import Path
from typing import Optional

import psutil
from PySide6.QtCore import QObject, QProcess, Signal

from .. import settings

CHAT_BACKEND_DIR = settings.BASE_DIR / "chat_backend"
_CHAT_VENV_DIR = settings.BASE_DIR / "venv_chat"
# venv_chat은 chat_backend/setup_chat_model.py가 만드는데, 그 스크립트도 OS에 따라
# Scripts/python.exe(Windows) 또는 bin/python(Linux/WSL2)에 만든다 - 여기서 같은 규칙을
# 따라야 한다.
CHAT_VENV_PYTHON = (
    _CHAT_VENV_DIR / "Scripts" / "python.exe" if os.name == "nt" else _CHAT_VENV_DIR / "bin" / "python"
)
CHAT_SERVER_SCRIPT = CHAT_BACKEND_DIR / "chat_server.py"
BACKEND_CONFIG_PATH = CHAT_BACKEND_DIR / "backend_config.json"


class ChatClient(QObject):
    ready = Signal(str)  # backend 이름 (예: "gguf-cuda")
    thought_received = Signal(str, str)  # request_id, 생각 과정 텍스트 조각
    token_received = Signal(str, str)  # request_id, 최종 답변 텍스트 조각
    response_done = Signal(str)  # request_id
    error = Signal(str, str)  # request_id(치명적 오류면 빈 문자열), message

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process: Optional[QProcess] = None
        self._buffer = ""
        self._ready = False

    def is_installed(self) -> bool:
        return CHAT_VENV_PYTHON.exists() and CHAT_SERVER_SCRIPT.exists() and BACKEND_CONFIG_PATH.exists()

    def is_running(self) -> bool:
        return self._process is not None and self._process.state() != QProcess.NotRunning

    def is_ready(self) -> bool:
        return self._ready

    def start(self) -> None:
        if self.is_running():
            return
        if not self.is_installed():
            self.error.emit(
                "",
                "챗봇 모델이 설치되어 있지 않습니다. Gen2Train\\setup_chat.bat을 먼저 실행하세요.",
            )
            return

        self._ready = False
        self._buffer = ""

        process = QProcess(self)
        process.setProgram(str(CHAT_VENV_PYTHON))
        process.setArguments([str(CHAT_SERVER_SCRIPT)])
        process.setWorkingDirectory(str(CHAT_BACKEND_DIR))
        # stderr(모델 로딩 시 llama.cpp가 찍는 진단 로그)와 stdout(JSON 프로토콜)을 섞으면
        # JSON 파싱이 깨지므로 분리한다.
        process.setProcessChannelMode(QProcess.SeparateChannels)
        process.readyReadStandardOutput.connect(self._on_stdout)
        process.errorOccurred.connect(self._on_process_error)
        process.finished.connect(self._on_finished)

        self._process = process
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
        self._ready = False

    def send(self, request_id: str, context: str, question: str) -> None:
        if not self.is_running() or not self._ready:
            self.error.emit(request_id, "모델이 아직 준비되지 않았습니다.")
            return
        payload = json.dumps(
            {"id": request_id, "context": context, "question": question}, ensure_ascii=False
        )
        self._process.write((payload + "\n").encode("utf-8"))

    # ------------------------------------------------------------- internal

    def _on_stdout(self) -> None:
        if not self._process:
            return
        data = bytes(self._process.readAllStandardOutput())
        self._buffer += data.decode("utf-8", errors="replace")
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            self._dispatch(obj)

    def _dispatch(self, obj: dict) -> None:
        msg_type = obj.get("type")
        if msg_type == "ready":
            self._ready = True
            self.ready.emit(obj.get("backend", ""))
        elif msg_type == "thought":
            self.thought_received.emit(obj.get("id", ""), obj.get("text", ""))
        elif msg_type == "thought_done":
            pass  # token_received가 오기 시작하면 답변 단계로 넘어간 것이므로 별도 처리 불필요
        elif msg_type == "token":
            self.token_received.emit(obj.get("id", ""), obj.get("text", ""))
        elif msg_type == "done":
            self.response_done.emit(obj.get("id", ""))
        elif msg_type == "error":
            self.error.emit(obj.get("id", ""), obj.get("message", ""))
        elif msg_type == "fatal_error":
            self._ready = False
            self.error.emit("", obj.get("message", ""))

    def _on_process_error(self, _err) -> None:
        if self._process is None:
            return
        self.error.emit("", f"챗봇 프로세스 오류: {self._process.errorString()}")

    def _on_finished(self, _exit_code, _exit_status) -> None:
        self._ready = False
        self._process = None
