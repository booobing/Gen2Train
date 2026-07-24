"""Gen2Train 메인 윈도우: 폴더/모델 지정, 파라미터 탭, 로그, 학습 시작/중지."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import settings
from ..core import dataset_prep, docs, presets, trainer
from ..core.param_store import ParamStore
from ..core.system_info import SystemHeartbeat
from .advanced_panel import AdvancedPanel
from .basic_panel import BasicPanel
from .chat_panel import ChatPanel
from .help_panel import HelpPanel
from .widgets import FilePickerRow, FolderPickerRow

MODEL_TYPES = [("SD 1.5 / SD 2.x", "sd"), ("SDXL", "sdxl")]

# Prodigy/DAdapt 계열은 옵티마이저 자체가 학습률을 자동 조절하기 때문에 alpha=1을 의도적으로 쓴다.
_EXOTIC_ALPHA_OPTIMIZERS = {"prodigy", "dadaptadam", "dadaptation", "dadaptlion", "dadaptsgd", "dadaptadan"}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gen2Train - 결함 이미지 생성용 LoRA 학습")

        settings.ensure_dirs()
        self._store = ParamStore()
        self._trainer = trainer.TrainerProcess(self)
        self._trainer.log_line.connect(self._append_log)
        self._trainer.progress.connect(self._on_progress)
        self._trainer.finished.connect(self._on_finished)
        self._epoch_text = ""
        self._restoring_state = False
        # AI 도우미가 답변할 때 참고할 GPU/VRAM/CPU/RAM 정보를 백그라운드에서 주기적으로 갱신.
        self._heartbeat = SystemHeartbeat(interval_seconds=5.0, parent=self)

        self._build_ui()
        self._load_last_used(silent_error=True)
        QApplication.instance().focusChanged.connect(self._on_focus_changed)

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        main_content = QWidget()
        root = QVBoxLayout(main_content)

        root.addWidget(self._build_model_data_group())

        tabs = QTabWidget()
        basic_scroll = QScrollArea()
        basic_scroll.setWidgetResizable(True)
        self._basic_panel = BasicPanel(self._store)
        basic_scroll.setWidget(self._basic_panel)
        tabs.addTab(basic_scroll, "기본 설정")

        self._advanced_panel = AdvancedPanel(self._store, model_type=self._current_model_type())
        tabs.addTab(self._advanced_panel, "고급 설정 (전체 옵션)")
        root.addWidget(tabs, 1)

        root.addWidget(self._build_run_controls())
        root.addWidget(self._build_log_console(), 1)

        self._help_panel = HelpPanel()
        self._help_panel.set_content(docs.build_html(self._current_model_type()))

        self._chat_panel = ChatPanel(
            self._store,
            get_model_type=self._current_model_type,
            get_top_state=lambda: self._collect_state()["top"],
            is_training_running=self._trainer.is_running,
            get_system_snapshot=self._heartbeat.latest,
        )
        self._heartbeat.updated.connect(self._chat_panel.update_system_status)

        right_tabs = QTabWidget()
        right_tabs.addTab(self._help_panel, "도움말")
        right_tabs.addTab(self._chat_panel, "AI 도우미")

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(main_content)
        splitter.addWidget(right_tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.setCentralWidget(splitter)

    def _on_focus_changed(self, _old, new):
        if new is None:
            return
        dest = new.property("g2t_dest")
        if dest:
            self._help_panel.show_dest(dest)

    def _build_model_data_group(self) -> QGroupBox:
        box = QGroupBox("모델 && 데이터")
        layout = QVBoxLayout(box)

        model_row = QHBoxLayout()
        self._model_type_combo = QComboBox()
        for label, value in MODEL_TYPES:
            self._model_type_combo.addItem(label, value)
        self._model_type_combo.currentIndexChanged.connect(self._on_model_type_changed)
        model_row.addWidget(QLabel("모델 타입"))
        model_row.addWidget(self._model_type_combo)
        model_row.addStretch(1)
        layout.addLayout(model_row)

        self._model_path = FilePickerRow(
            "베이스 모델", name_filter="Model files (*.safetensors *.ckpt *.pt)", placeholder="학습에 사용할 기반 SD 모델 파일"
        )
        layout.addWidget(self._model_path)

        self._defect_dir = FolderPickerRow("결함 이미지 폴더", placeholder="결함 이미지가 들어있는 폴더 (필수)")
        layout.addWidget(self._defect_dir)

        self._good_dir = FolderPickerRow("양품 이미지 폴더", placeholder="정상 제품 이미지 폴더 (선택, 정규화 이미지로 사용)")
        layout.addWidget(self._good_dir)

        trigger_row = QHBoxLayout()
        self._trigger_word = QLineEdit()
        self._trigger_word.setPlaceholderText("예: scratch_defect (결함 이미지 전체에 붙는 캡션 단어)")
        trigger_row.addWidget(QLabel("트리거 단어"))
        trigger_row.addWidget(self._trigger_word, 1)
        layout.addLayout(trigger_row)

        extra_row = QHBoxLayout()
        self._extra_tags = QLineEdit()
        self._extra_tags.setPlaceholderText("선택: 트리거 단어 뒤에 붙일 추가 태그 (쉼표로 구분)")
        extra_row.addWidget(QLabel("추가 태그"))
        extra_row.addWidget(self._extra_tags, 1)
        layout.addLayout(extra_row)

        existing_caption_row = QHBoxLayout()
        self._use_existing_captions = QCheckBox("이미지 폴더에 이미 있는 캡션(.txt) 사용")
        self._use_existing_captions.setToolTip(
            "체크하면 결함 이미지 폴더에서 같은 파일명의 .txt 캡션을 찾아 트리거 단어 뒤에 이어붙인다.\n"
            "이미지마다 결함 종류(스크래치/덴트, 세로/가로 등)가 다르다면, WD14 태거나 수동 태깅으로 만든 캡션을\n"
            "그대로 활용하는 이 방식이 트리거 단어 하나로 뭉뚱그리는 것보다 훨씬 정확하게 학습된다.\n"
            "캡션 파일이 없는 이미지는 트리거 단어(+추가 태그)만 사용된다."
        )
        self._use_existing_captions.toggled.connect(self._on_use_existing_captions_toggled)
        existing_caption_row.addWidget(self._use_existing_captions)
        layout.addLayout(existing_caption_row)

        reg_row = QHBoxLayout()
        self._use_reg = QCheckBox("양품 이미지를 정규화 이미지로 사용")
        self._use_reg.setChecked(True)
        reg_row.addWidget(self._use_reg)
        reg_row.addWidget(QLabel("클래스 단어"))
        self._class_word = QLineEdit("normal product")
        reg_row.addWidget(self._class_word)
        layout.addLayout(reg_row)

        repeat_row = QHBoxLayout()
        repeat_row.addWidget(QLabel("결함 이미지 반복 횟수"))
        self._repeat = QSpinBox()
        self._repeat.setRange(1, 1000)
        self._repeat.setValue(10)
        self._repeat.setToolTip("결함 이미지가 적을수록 반복 횟수를 늘려 epoch당 노출 횟수를 맞춘다.")
        repeat_row.addWidget(self._repeat)
        repeat_row.addWidget(QLabel("정규화 이미지 반복 횟수"))
        self._reg_repeat = QSpinBox()
        self._reg_repeat.setRange(1, 1000)
        self._reg_repeat.setValue(1)
        repeat_row.addWidget(self._reg_repeat)
        repeat_row.addStretch(1)
        layout.addLayout(repeat_row)

        output_row = QHBoxLayout()
        self._output_name = QLineEdit()
        self._output_name.setPlaceholderText("결과물 이름 (예: part_scratch_lora)")
        self._output_name.textChanged.connect(self._update_output_hint)
        output_row.addWidget(QLabel("결과물 이름"))
        output_row.addWidget(self._output_name, 1)
        layout.addLayout(output_row)

        self._output_hint = QLabel()
        self._output_hint.setStyleSheet("color: gray;")
        layout.addWidget(self._output_hint)
        self._update_output_hint()

        resume_row = QHBoxLayout()
        self._resume_checkbox = QCheckBox("기존 LoRA에 이어서 학습")
        self._resume_checkbox.setToolTip(
            "체크하면 아래에서 고른 기존 LoRA 파일에서 이어서 학습한다.\n"
            "dim_from_weights(랭크 자동 인식)가 함께 켜지며, network_weights 경로 없이 이 옵션만 켜면 학습이 바로 실패하므로 반드시 파일을 지정해야 한다."
        )
        self._resume_checkbox.toggled.connect(self._on_resume_toggled)
        resume_row.addWidget(self._resume_checkbox)
        resume_row.addStretch(1)
        layout.addLayout(resume_row)

        self._resume_lora_path = FilePickerRow(
            "기존 LoRA 파일", name_filter="LoRA weights (*.safetensors)", placeholder="이어서 학습할 기존 LoRA .safetensors 파일"
        )
        self._resume_lora_path.setEnabled(False)
        self._resume_lora_path.pathChanged.connect(self._on_resume_path_changed)
        layout.addWidget(self._resume_lora_path)

        return box

    def _build_run_controls(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        preset_row = QHBoxLayout()
        self._preset_combo = QComboBox()
        self._refresh_preset_list()
        preset_row.addWidget(QLabel("프리셋"))
        preset_row.addWidget(self._preset_combo, 1)
        load_btn = QPushButton("불러오기")
        load_btn.clicked.connect(self._on_load_preset)
        save_btn = QPushButton("현재 설정 저장")
        save_btn.clicked.connect(self._on_save_preset)
        preset_row.addWidget(load_btn)
        preset_row.addWidget(save_btn)
        layout.addLayout(preset_row)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel("대기 중")
        layout.addWidget(self._status_label)

        btn_row = QHBoxLayout()
        self._start_btn = QPushButton("학습 시작")
        self._start_btn.clicked.connect(self._on_start)
        self._stop_btn = QPushButton("중지")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        self._open_output_btn = QPushButton("출력 폴더 열기")
        self._open_output_btn.clicked.connect(self._on_open_output)
        btn_row.addWidget(self._start_btn)
        btn_row.addWidget(self._stop_btn)
        btn_row.addWidget(self._open_output_btn)
        layout.addLayout(btn_row)

        return widget

    def _build_log_console(self) -> QWidget:
        box = QGroupBox("학습 로그")
        layout = QVBoxLayout(box)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(5000)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        self._log.setFont(mono)
        layout.addWidget(self._log)
        return box

    # ------------------------------------------------------------- helpers

    def _current_model_type(self) -> str:
        return self._model_type_combo.currentData() if hasattr(self, "_model_type_combo") else "sd"

    def _on_model_type_changed(self, _index):
        if self._trainer.is_running():
            QMessageBox.warning(self, "Gen2Train", "학습 중에는 모델 타입을 변경할 수 없습니다.")
            return
        self._advanced_panel.set_model_type(self._current_model_type())
        self._help_panel.set_content(docs.build_html(self._current_model_type()))

    def _on_resume_path_changed(self, path: str):
        if self._restoring_state:
            return
        self._store.set("network_weights", path or None)

    def _on_resume_toggled(self, checked: bool):
        self._resume_lora_path.setEnabled(checked)
        if self._restoring_state:
            return

        self._store.set("dim_from_weights", checked)
        if not checked:
            self._store.set("network_weights", None)
            self._resume_lora_path.set_path("")
            return

        self._store.set("network_weights", self._resume_lora_path.path() or None)
        reply = QMessageBox.question(
            self,
            "이어서 학습 추천 설정",
            "이어서 학습 시 흔히 권장되는 설정을 같이 적용할까요?\n\n"
            "- Shuffle caption 켬 (캡션 태그 순서/구성이 조금 달라져도 안정적으로 학습)\n"
            "- Learning rate를 현재 값의 절반으로 낮춤 (기존 지식 보존, 과적합 방지)\n\n"
            "나중에 고급 탭에서 언제든 다시 바꿀 수 있다.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._store.set("shuffle_caption", True)
            current_lr = self._store.get("learning_rate") or 0.0001
            self._store.set("learning_rate", round(current_lr / 2, 8))

    def _on_use_existing_captions_toggled(self, checked: bool):
        if self._restoring_state or not checked:
            return
        reply = QMessageBox.question(
            self,
            "캡션 혼합 권장 설정",
            "이미지별로 캡션 내용이 달라지면 다음을 함께 켜는 것이 좋습니다:\n\n"
            "- Shuffle caption 켬 (여러 태그의 순서에 편향되지 않도록)\n"
            "- Keep tokens = 1 (트리거 단어는 항상 맨 앞에 고정)\n\n"
            "지금 적용할까요?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._store.set("shuffle_caption", True)
            self._store.set("keep_tokens", 1)

    def _run_output_dir(self) -> Path:
        name = dataset_prep.sanitize_run_name(self._output_name.text() or "run")
        return settings.OUTPUTS_DIR / name

    def _update_output_hint(self):
        self._output_hint.setText(f"저장 위치: {self._run_output_dir()}")

    def _append_log(self, line: str):
        self._log.appendPlainText(line)

    # ---------------------------------------------------------- run/stop

    def _on_start(self):
        if self._trainer.is_running():
            return
        try:
            model_path = self._model_path.path()
            if not model_path or not Path(model_path).exists():
                raise ValueError("베이스 모델 파일을 선택하세요.")
            if not self._trigger_word.text().strip():
                raise ValueError("트리거 단어를 입력하세요.")
            if not self._output_name.text().strip():
                raise ValueError("결과물 이름을 입력하세요.")

            if self._store.get("dim_from_weights") and not self._store.get("network_weights"):
                raise ValueError(
                    "'DIM from weights'가 켜져 있는데 기존 LoRA 파일 경로가 없습니다.\n"
                    "상단의 '기존 LoRA에 이어서 학습'에서 파일을 선택하거나, 고급 탭에서 해당 옵션을 꺼주세요."
                )
            if self._resume_checkbox.isChecked():
                resume_path = self._store.get("network_weights")
                if not resume_path or not Path(resume_path).exists():
                    raise ValueError("이어서 학습할 기존 LoRA 파일 경로를 확인하세요.")
        except ValueError as exc:
            QMessageBox.critical(self, "Gen2Train", str(exc))
            return

        # 이미지 복사 등 실제 작업을 시작하기 전에 확인받는다 (취소 시 불필요한 파일 복사를 피한다).
        if not self._confirm_network_alpha_sane():
            return

        try:
            good_dir = self._good_dir.path() or None
            result = dataset_prep.prepare_dataset(
                run_name=self._output_name.text(),
                defect_dir=Path(self._defect_dir.path()),
                trigger_word=self._trigger_word.text(),
                repeat=self._repeat.value(),
                good_dir=Path(good_dir) if good_dir else None,
                use_good_as_reg=self._use_reg.isChecked(),
                class_word=self._class_word.text() or "normal product",
                reg_repeat=self._reg_repeat.value(),
                extra_tags=self._extra_tags.text(),
                use_existing_captions=self._use_existing_captions.isChecked(),
            )
        except (dataset_prep.DatasetPrepError, OSError) as exc:
            QMessageBox.critical(self, "Gen2Train", str(exc))
            return

        self._append_log(
            f"[Gen2Train] 데이터셋 준비 완료: 결함 {result.num_defect_images}장"
            + (f", 정규화(양품) {result.num_good_images}장" if result.reg_data_dir else "")
        )

        output_dir = self._run_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        program, args = trainer.build_command(
            model_type=self._current_model_type(),
            pretrained_model_path=model_path,
            train_data_dir=result.train_data_dir,
            output_dir=output_dir,
            output_name=dataset_prep.sanitize_run_name(self._output_name.text()),
            params=self._store.as_dict(),
            reg_data_dir=result.reg_data_dir,
        )

        # 멀티 GPU면 학습 프로세스 여러 개가 거의 동시에 뜨는데, CLIP 토크나이저를 처음 쓰는
        # PC에서는 그 프로세스들이 전부 동시에 같은 파일을 HuggingFace 캐시로 받으려다 경쟁
        # 상태로 캐시가 깨질 수 있다(실제로 재현됨). 띄우기 전에 미리 하나씩 받아둔다 - 이미
        # 캐시돼 있으면 순식간에 끝난다.
        self._append_log("[Gen2Train] 토크나이저 캐시를 확인합니다...")
        trainer.prewarm_tokenizer_cache(self._current_model_type(), settings.load_settings()["python_path"])

        try:
            self._trainer.start(program, args, cwd=settings.SD_SCRIPTS_DIR)
        except trainer.TrainingError as exc:
            QMessageBox.critical(self, "Gen2Train", str(exc))
            return

        self._progress_bar.setValue(0)
        self._epoch_text = ""
        self._status_label.setText("학습 시작 중...")
        self._set_running_state(True)
        self._save_last_used()

    def _confirm_network_alpha_sane(self) -> bool:
        network_dim = self._store.get("network_dim")
        network_alpha = self._store.get("network_alpha")
        optimizer_type = str(self._store.get("optimizer_type") or "").lower()
        if not network_dim or not network_alpha or optimizer_type in _EXOTIC_ALPHA_OPTIMIZERS:
            return True
        if network_alpha >= network_dim / 8:
            return True
        reply = QMessageBox.question(
            self,
            "설정 확인",
            f"Network Alpha({network_alpha})가 Network Dim({network_dim})에 비해 많이 낮습니다.\n"
            "일반적인 옵티마이저(AdamW 등)에서는 보통 Dim의 절반 정도(예: Dim 32 -> Alpha 16)를 권장합니다.\n"
            "Prodigy/DAdapt 계열처럼 의도적으로 Alpha=1을 쓰는 경우가 아니라면 값을 조정하는 게 좋습니다.\n\n"
            "그대로 진행할까요?",
            QMessageBox.Yes | QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    def _on_stop(self):
        reply = QMessageBox.question(
            self,
            "학습 중지",
            "학습을 중지할까요? 지금까지의 진행 상황은 마지막 저장 시점까지만 남습니다.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._trainer.stop()

    def _on_progress(self, step, total_steps, epoch, total_epochs):
        if total_epochs:
            self._epoch_text = f"Epoch {epoch}/{total_epochs}"
        if total_steps:
            pct = int(step / total_steps * 100)
            self._progress_bar.setValue(pct)
            self._status_label.setText(f"{self._epoch_text} · Step {step}/{total_steps} ({pct}%)".strip(" ·"))
        elif self._epoch_text:
            self._status_label.setText(self._epoch_text)

    def _on_finished(self, success: bool, message: str):
        self._set_running_state(False)
        self._status_label.setText(message)
        if success:
            self._progress_bar.setValue(100)
            QMessageBox.information(self, "Gen2Train", message)
        else:
            QMessageBox.warning(self, "Gen2Train", message)

    def _set_running_state(self, running: bool):
        self._start_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        self._model_type_combo.setEnabled(not running)

    def _on_open_output(self):
        path = self._run_output_dir()
        path.mkdir(parents=True, exist_ok=True)
        # os.startfile은 Windows 전용이라 Linux/WSL2에서는 AttributeError로 죽는다(실제로
        # 재현됨). macOS는 open, Linux는 xdg-open(데스크톱 환경/파일 관리자가 있을 때만
        # 동작 - WSL2는 최소 설치라 없을 수 있다)을 쓰고, 그마저 없으면 폴더 경로를 메시지로
        # 보여줘서 최소한 죽지는 않게 한다.
        if os.name == "nt":
            os.startfile(str(path))  # noqa: S606 - Windows 전용 API
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif shutil.which("xdg-open"):
            subprocess.run(["xdg-open", str(path)], check=False)
        else:
            QMessageBox.information(self, "Gen2Train", f"출력 폴더: {path}")

    # ----------------------------------------------------------- presets

    def _collect_state(self) -> dict:
        return {
            "top": {
                "model_path": self._model_path.path(),
                "model_type": self._current_model_type(),
                "defect_dir": self._defect_dir.path(),
                "good_dir": self._good_dir.path(),
                "trigger_word": self._trigger_word.text(),
                "extra_tags": self._extra_tags.text(),
                "use_reg": self._use_reg.isChecked(),
                "class_word": self._class_word.text(),
                "repeat": self._repeat.value(),
                "reg_repeat": self._reg_repeat.value(),
                "output_name": self._output_name.text(),
                "resume_enabled": self._resume_checkbox.isChecked(),
                "use_existing_captions": self._use_existing_captions.isChecked(),
            },
            "params": self._store.as_dict(),
        }

    def _apply_state(self, state: dict):
        # 복원 중에는 체크박스 toggled 핸들러가 확인 다이얼로그를 띄우거나 LR을 다시 절반으로
        # 깎는 등의 부작용을 일으키면 안 되므로, 메서드 전체를 이 플래그로 감싼다.
        self._restoring_state = True
        try:
            # "top"(모델/폴더/트리거 단어 등)이 아예 없는 프리셋은 학습 파라미터만 담은 프리셋이므로,
            # 이미 입력해둔 폴더/트리거 단어 등을 지우지 않도록 top 항목은 건드리지 않는다.
            top = state.get("top")
            if top:
                self._model_path.set_path(top.get("model_path", ""))
                model_type = top.get("model_type", "sd")
                idx = self._model_type_combo.findData(model_type)
                if idx >= 0:
                    self._model_type_combo.setCurrentIndex(idx)
                self._defect_dir.set_path(top.get("defect_dir", ""))
                self._good_dir.set_path(top.get("good_dir", ""))
                self._trigger_word.setText(top.get("trigger_word", ""))
                self._extra_tags.setText(top.get("extra_tags", ""))
                self._use_reg.setChecked(top.get("use_reg", True))
                self._class_word.setText(top.get("class_word", "normal product"))
                self._repeat.setValue(top.get("repeat", 10))
                self._reg_repeat.setValue(top.get("reg_repeat", 1))
                self._output_name.setText(top.get("output_name", ""))
                self._use_existing_captions.setChecked(top.get("use_existing_captions", False))

            params = state.get("params", {})
            if params:
                self._store.set_many(params, silent=False)

            # 이어서 학습 체크박스/파일 표시는 store(dim_from_weights, network_weights)를 그대로 반영한다.
            resume_active = bool(self._store.get("dim_from_weights"))
            self._resume_checkbox.setChecked(resume_active)
            self._resume_lora_path.setEnabled(resume_active)
            self._resume_lora_path.set_path(self._store.get("network_weights") or "")
        finally:
            self._restoring_state = False

    def _refresh_preset_list(self):
        self._preset_combo.clear()
        self._preset_combo.addItems(presets.list_presets())

    def _on_save_preset(self):
        name, ok = QInputDialog.getText(self, "프리셋 저장", "프리셋 이름")
        if not ok or not name.strip():
            return
        presets.save_preset(name.strip(), self._collect_state())
        self._refresh_preset_list()
        idx = self._preset_combo.findText(name.strip())
        if idx >= 0:
            self._preset_combo.setCurrentIndex(idx)

    def _on_load_preset(self):
        name = self._preset_combo.currentText()
        if not name:
            return
        state = presets.load_preset(name)
        if state:
            self._apply_state(state)

    def _save_last_used(self):
        presets.save_last_used(self._collect_state())

    def _load_last_used(self, silent_error: bool = False):
        try:
            state = presets.load_last_used()
            if state:
                self._apply_state(state)
        except Exception as exc:  # noqa: BLE001 - 시작 시점 복원 실패는 앱을 막지 않는다
            if not silent_error:
                raise
            self._append_log(f"[Gen2Train] 이전 설정 복원 실패: {exc}")

    # -------------------------------------------------------------- close

    def closeEvent(self, event):
        if self._trainer.is_running():
            reply = QMessageBox.question(
                self,
                "종료",
                "학습이 진행 중입니다. 종료하면 학습이 중단됩니다. 종료할까요?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return
            self._trainer.stop()
        self._chat_panel.shutdown()
        self._heartbeat.stop()
        self._save_last_used()
        event.accept()
