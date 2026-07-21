"""로컬 LLM(EXAONE-Deep, chat_backend/) 기반 파라미터 추천/QA 채팅 패널."""
import uuid

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..core.chat_client import ChatClient
from ..core.chat_context import RECOMMEND_QUESTION, build_context
from ..core.param_store import ParamStore


def _blockquote(text: str) -> str:
    # 마크다운 인용문(>) 형태로 만들어 생각 과정을 본문과 시각적으로 구분한다.
    lines = text.splitlines() or [""]
    return "\n".join(f"> {line}" for line in lines)


class ChatPanel(QWidget):
    def __init__(
        self,
        store: ParamStore,
        get_model_type,
        get_top_state,
        is_training_running,
        get_system_snapshot=None,
        parent=None,
    ):
        """
        get_model_type: () -> str          현재 모델 타입("sd"/"sdxl")을 얻는 콜백
        get_top_state: () -> dict          main_window의 top 필드 dict를 얻는 콜백
        is_training_running: () -> bool    학습 중 여부 (VRAM 경합 경고용)
        get_system_snapshot: () -> dict    SystemHeartbeat의 최신 GPU/VRAM/CPU/RAM 스냅샷을 얻는 콜백
        """
        super().__init__(parent)
        self._store = store
        self._get_model_type = get_model_type
        self._get_top_state = get_top_state
        self._is_training_running = is_training_running
        self._get_system_snapshot = get_system_snapshot or (lambda: {})

        self._client = ChatClient(self)
        self._client.ready.connect(self._on_ready)
        self._client.token_received.connect(self._on_token)
        self._client.response_done.connect(self._on_done)
        self._client.error.connect(self._on_error)

        self._current_request_id = None
        self._current_answer_raw = None
        self._history_md: list = []
        self._pending_request = None  # (request_id, context, question) - ready를 기다리는 중인 요청

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._status_label = QLabel("AI 도우미 - 아직 모델을 불러오지 않았습니다.")
        self._status_label.setStyleSheet("color: gray;")
        layout.addWidget(self._status_label)

        self._system_status_label = QLabel("")
        self._system_status_label.setStyleSheet("color: gray; font-size: 11px;")
        self._system_status_label.setToolTip("AI 도우미가 답변할 때 참고하는 실시간 시스템 정보(하트비트)")
        layout.addWidget(self._system_status_label)
        self.update_system_status(self._get_system_snapshot())

        self._log = QTextBrowser()
        self._log.setOpenExternalLinks(False)
        layout.addWidget(self._log, 1)

        recommend_row = QHBoxLayout()
        self._recommend_btn = QPushButton("현재 설정 기준으로 추천받기")
        self._recommend_btn.setToolTip("지금 입력해둔 파라미터 값과 도움말 설명을 근거로 개선점을 물어본다.")
        self._recommend_btn.clicked.connect(self._on_recommend_clicked)
        recommend_row.addWidget(self._recommend_btn)
        layout.addLayout(recommend_row)

        input_row = QHBoxLayout()
        self._input = QLineEdit()
        self._input.setPlaceholderText("궁금한 점을 물어보세요 (예: network_dim을 64로 올리면 어떻게 될까?)")
        self._input.returnPressed.connect(self._on_send_clicked)
        self._send_btn = QPushButton("전송")
        self._send_btn.clicked.connect(self._on_send_clicked)
        input_row.addWidget(self._input, 1)
        input_row.addWidget(self._send_btn)
        layout.addLayout(input_row)

    # ------------------------------------------------------------- 모델 시작

    def _ensure_started(self) -> bool:
        if self._client.is_ready():
            return True
        if not self._client.is_installed():
            self._append_system(
                "챗봇 모델이 설치되어 있지 않습니다. Gen2Train\\setup_chat.bat을 먼저 실행한 뒤 앱을 다시 시작하세요."
            )
            return False
        if self._is_training_running():
            reply = QMessageBox.question(
                self,
                "AI 도우미",
                "학습이 진행 중입니다. 챗봇 모델을 로드하면 VRAM을 추가로 사용해 학습 속도가\n"
                "느려지거나 실패할 수 있습니다. 그래도 계속할까요?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return False
        if not self._client.is_running():
            self._status_label.setText("모델 로딩 중입니다... (수 초 정도 걸립니다)")
            self._client.start()
        return True

    def _on_ready(self, backend: str):
        self._status_label.setText(f"준비됨 (백엔드: {backend})")
        if self._pending_request is not None:
            req_id, context, question = self._pending_request
            self._pending_request = None
            self._client.send(req_id, context, question)

    # --------------------------------------------------------------- 전송

    def _send_question(self, question: str, context: str):
        if not question.strip():
            return
        if not self._ensure_started():
            return
        if self._current_request_id is not None:
            self._append_system("이전 답변이 아직 진행 중입니다. 완료된 후 다시 시도하세요.")
            return

        self._current_request_id = str(uuid.uuid4())
        self._current_answer_raw = ""
        self._append_user(question)
        self._refresh_log()
        self._send_btn.setEnabled(False)

        if self._client.is_ready():
            self._client.send(self._current_request_id, context, question)
        else:
            # 모델이 아직 로딩 중(또는 막 start()를 호출한 직후)이면, ready 시그널이 올 때
            # 자동으로 보내도록 큐에 담아둔다. 여기서 그냥 보내면 로딩 중엔 조용히 씹혀서
            # _current_request_id가 영원히 안 풀리는 문제가 있었다.
            self._pending_request = (self._current_request_id, context, question)

    def _build_context(self) -> str:
        return build_context(
            self._store,
            self._get_model_type(),
            top_state=self._get_top_state(),
            system_snapshot=self._get_system_snapshot(),
        )

    def _on_recommend_clicked(self):
        self._send_question(RECOMMEND_QUESTION, self._build_context())

    def _on_send_clicked(self):
        question = self._input.text().strip()
        if not question:
            return
        self._input.clear()
        self._send_question(question, self._build_context())

    def update_system_status(self, snapshot: dict):
        if not snapshot:
            self._system_status_label.setText("")
            return
        gpus = snapshot.get("gpus") or []
        parts = []
        if gpus:
            gpu = gpus[0]
            used_gb = gpu["vram_used_mb"] / 1024
            total_gb = gpu["vram_total_mb"] / 1024
            parts.append(f"GPU: {gpu['name']} · VRAM {used_gb:.1f}/{total_gb:.1f}GB")
        ram = snapshot.get("ram") or {}
        if ram:
            parts.append(f"RAM {ram.get('used_pct', '?')}%")
        cpu = snapshot.get("cpu") or {}
        if cpu:
            parts.append(f"CPU {cpu.get('usage_pct', '?')}%")
        self._system_status_label.setText(" · ".join(parts))

    # ------------------------------------------------------------- 응답 처리

    def _on_token(self, req_id: str, text: str):
        if req_id != self._current_request_id or self._current_answer_raw is None:
            return
        self._current_answer_raw += text
        self._refresh_log()

    def _on_done(self, req_id: str):
        if req_id != self._current_request_id:
            return
        if self._current_answer_raw is not None:
            self._history_md.append(self._render_assistant_markdown(self._current_answer_raw))
        self._current_answer_raw = None
        self._current_request_id = None
        self._send_btn.setEnabled(True)
        self._refresh_log()

    def _on_error(self, req_id: str, message: str):
        if req_id and req_id != self._current_request_id:
            return
        self._append_system(f"오류: {message}")
        self._current_answer_raw = None
        self._current_request_id = None
        self._send_btn.setEnabled(True)
        self._refresh_log()

    # ------------------------------------------------------------- 화면 표시
    #
    # 모델 답변이 마크다운(###, **굵게**, 목록 등)으로 나오므로, HTML을 직접 조립하는 대신
    # 전체 대화를 마크다운 텍스트로 쌓아두고 QTextBrowser.setMarkdown()으로 그때그때
    # 다시 렌더링한다.

    def _append_user(self, text: str):
        self._history_md.append(f"**나:** {text}")

    def _append_system(self, text: str):
        self._history_md.append(f"*[Gen2Train] {text}*")
        self._refresh_log()

    def _render_assistant_markdown(self, raw_text: str, in_progress: bool = False) -> str:
        label = "**AI 도우미 (입력 중...)**" if in_progress else "**AI 도우미**"
        if "</thought>" in raw_text:
            thought, final = raw_text.split("</thought>", 1)
            thought = thought.replace("<thought>", "").strip()
            final = final.strip()
            return f"{label}\n\n*생각 과정*\n\n{_blockquote(thought)}\n\n{final if final else '...'}"
        if "<thought>" in raw_text:
            thought = raw_text.replace("<thought>", "").strip()
            return f"{label}\n\n*(생각 중)*\n\n{_blockquote(thought)}"
        return f"{label}\n\n{raw_text if raw_text else '...'}"

    def _refresh_log(self):
        parts = list(self._history_md)
        if self._current_answer_raw is not None:
            parts.append(self._render_assistant_markdown(self._current_answer_raw, in_progress=True))
        self._log.setMarkdown("\n\n---\n\n".join(parts))
        scrollbar = self._log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def shutdown(self):
        self._client.stop()
