"""실사용에 필요한 핵심 파라미터만 모은 기본 탭."""
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QFormLayout, QLabel, QLineEdit, QSpinBox, QWidget

from ..core.help_texts import get_help
from ..core.param_store import ParamStore

RESOLUTION_PRESETS = ["512,512", "640,640", "768,768", "1024,1024"]
MIXED_PRECISION_CHOICES = ["no", "fp16", "bf16"]
OPTIMIZER_CHOICES = ["AdamW8bit", "AdamW", "Lion8bit", "Lion", "Prodigy", "DAdaptAdam", "SGDNesterov8bit"]


def _tip(dest: str, fallback: str) -> str:
    return get_help(dest, fallback)


# 사용자가 아무 값도 건드리지 않았을 때 채워질 실사용 기본값.
BASIC_DEFAULTS = {
    "resolution": "512,512",
    "train_batch_size": 2,
    "gradient_accumulation_steps": 1,
    "max_train_epochs": 10,
    "learning_rate": 0.0001,
    "network_dim": 32,
    "network_alpha": 16,
    "save_every_n_epochs": 1,
    "seed": 42,
    "mixed_precision": "fp16",
    "optimizer_type": "AdamW8bit",
}


class BasicPanel(QWidget):
    def __init__(self, store: ParamStore, parent=None):
        super().__init__(parent)
        self._store = store
        self._bindings: dict[str, tuple] = {}  # dest -> (get_fn, set_fn)
        self._syncing = False

        layout = QFormLayout(self)
        layout.setSpacing(10)

        self._add_combo(
            layout,
            dest="resolution",
            label="해상도",
            choices=RESOLUTION_PRESETS,
            editable=True,
            tooltip=_tip("resolution", "학습 이미지 해상도 (가로,세로). 결함이 작다면 해상도를 높이는 게 유리하다."),
        )
        self._add_spin(
            layout,
            dest="train_batch_size",
            label="배치 크기",
            minimum=1,
            maximum=64,
            tooltip=_tip(
                "train_batch_size",
                "한 번에 학습할 이미지 수. VRAM이 부족하면 줄인다. "
                "현재 VRAM 사용량은 'AI 도우미' 탭 상단에서 실시간으로 확인할 수 있다.",
            ),
        )
        self._add_spin(
            layout,
            dest="gradient_accumulation_steps",
            label="그래디언트 누적 스텝",
            minimum=1,
            maximum=64,
            tooltip=_tip("gradient_accumulation_steps", "배치 사이즈를 낮추는 대신 이 값을 올리면 VRAM은 적게 쓰면서 실질적으로 큰 배치처럼 학습할 수 있다."),
        )
        self._effective_batch_label = QLabel()
        self._effective_batch_label.setStyleSheet("color: gray;")
        self._effective_batch_label.setToolTip("배치 크기 × 그래디언트 누적 스텝. VRAM 설정을 바꿀 때 이 값을 일정하게 유지하면 학습 결과가 비슷하게 나온다.")
        layout.addRow("", self._effective_batch_label)
        self._add_spin(
            layout,
            dest="max_train_epochs",
            label="Epoch 수",
            minimum=1,
            maximum=1000,
            tooltip=_tip("max_train_epochs", "전체 데이터셋을 몇 번 반복 학습할지."),
        )
        self._add_double_spin(
            layout,
            dest="learning_rate",
            label="Learning rate",
            minimum=0.0,
            maximum=1.0,
            decimals=6,
            step=0.0001,
            tooltip=_tip("learning_rate", "학습률. 너무 크면 붕괴, 너무 작으면 학습이 느리다. 기본 0.0001 권장."),
        )
        self._add_spin(
            layout,
            dest="network_dim",
            label="Network Dim (rank)",
            minimum=1,
            maximum=256,
            tooltip=_tip("network_dim", "LoRA 랭크. 클수록 표현력은 늘지만 과적합/용량도 커진다."),
        )
        self._add_double_spin(
            layout,
            dest="network_alpha",
            label="Network Alpha",
            minimum=0.1,
            maximum=256.0,
            decimals=1,
            step=1.0,
            tooltip=_tip("network_alpha", "LoRA 가중치 스케일. 보통 network_dim의 절반~같은 값을 사용한다."),
        )
        self._add_spin(
            layout,
            dest="save_every_n_epochs",
            label="저장 주기(epoch)",
            minimum=1,
            maximum=100,
            tooltip=_tip("save_every_n_epochs", "몇 epoch마다 중간 체크포인트를 저장할지."),
        )
        self._add_spin(
            layout,
            dest="seed",
            label="시드",
            minimum=-1,
            maximum=2_147_483_647,
            special_min_text="랜덤",
            tooltip=_tip("seed", "재현 가능한 결과가 필요하면 고정값을, 매번 다르게 하려면 -1(랜덤)을 사용한다."),
        )
        self._add_combo(
            layout,
            dest="mixed_precision",
            label="정밀도",
            choices=MIXED_PRECISION_CHOICES,
            editable=False,
            tooltip=_tip("mixed_precision", "fp16이 가장 무난하다. bf16은 최신 GPU에서 더 안정적일 수 있다."),
        )
        self._add_combo(
            layout,
            dest="optimizer_type",
            label="옵티마이저",
            choices=OPTIMIZER_CHOICES,
            editable=True,
            tooltip=_tip("optimizer_type", "AdamW8bit가 VRAM 절약과 성능의 균형이 좋아 기본값으로 권장된다."),
        )
        self._add_line_edit(
            layout,
            dest="text_encoder_lr",
            label="Text Encoder 학습률",
            placeholder="비워두면 Unet 학습률을 따라감. SDXL에서 VRAM/속도가 부족하면 0 입력 권장",
            tooltip=_tip(
                "text_encoder_lr",
                "비워두면 Unet과 같은 학습률을 쓴다. SDXL처럼 텍스트 인코더가 커서 VRAM/속도 부담이 크면 0을 입력해 "
                "텍스트 인코더 학습을 생략할 수 있다 (캡션 의미 전달 자체는 그대로 유지된다).",
            ),
        )

        self._store.valueChanged.connect(self._on_store_changed)
        self._apply_defaults()

    # -- 위젯 생성 헬퍼 -----------------------------------------------------

    def _add_spin(self, layout, dest, label, minimum, maximum, tooltip="", special_min_text=None):
        box = QSpinBox()
        box.setRange(minimum, maximum)
        if special_min_text:
            box.setSpecialValueText(special_min_text)
        box.setToolTip(tooltip)
        box.setProperty("g2t_dest", dest)  # 포커스 시 도움말 패널이 이 항목으로 스크롤하도록
        box.valueChanged.connect(lambda v: self._on_widget_changed(dest, v))
        layout.addRow(label, box)
        self._bindings[dest] = (box.value, lambda v: box.setValue(int(v) if v is not None else minimum))

    def _add_double_spin(self, layout, dest, label, minimum, maximum, decimals, step, tooltip=""):
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setDecimals(decimals)
        box.setSingleStep(step)
        box.setToolTip(tooltip)
        box.setProperty("g2t_dest", dest)
        box.valueChanged.connect(lambda v: self._on_widget_changed(dest, v))
        layout.addRow(label, box)
        self._bindings[dest] = (box.value, lambda v: box.setValue(float(v) if v is not None else minimum))

    def _add_combo(self, layout, dest, label, choices, editable, tooltip=""):
        box = QComboBox()
        box.setEditable(editable)
        box.addItems(choices)
        box.setToolTip(tooltip)
        box.setProperty("g2t_dest", dest)
        box.currentTextChanged.connect(lambda v: self._on_widget_changed(dest, v))
        layout.addRow(label, box)
        self._bindings[dest] = (box.currentText, lambda v: box.setCurrentText(str(v) if v is not None else ""))

    def _add_line_edit(self, layout, dest, label, placeholder="", tooltip=""):
        # 고급 탭의 ArgFieldWidget과 동일하게 "빈 문자열 = None(스크립트 기본값)"으로 취급한다.
        # nargs='*' 인자(예: text_encoder_lr)는 여러 값을 공백으로 구분한 문자열로 표현되므로
        # 스핀박스가 아닌 일반 텍스트 입력으로 다뤄야 고급 탭과 표현이 어긋나지 않는다.
        box = QLineEdit()
        box.setPlaceholderText(placeholder)
        box.setToolTip(tooltip)
        box.setProperty("g2t_dest", dest)
        box.textChanged.connect(lambda v: self._on_widget_changed(dest, v.strip() or None))
        layout.addRow(label, box)
        self._bindings[dest] = (box.text, lambda v: box.setText("" if v is None else str(v)))

    # -- store 동기화 ---------------------------------------------------------

    def _on_widget_changed(self, dest, value):
        if self._syncing:
            return
        self._store.set(dest, value)

    def _on_store_changed(self, dest, value):
        if dest in ("train_batch_size", "gradient_accumulation_steps"):
            self._update_effective_batch_label()
        if dest not in self._bindings:
            return
        _get, set_fn = self._bindings[dest]
        self._syncing = True
        try:
            set_fn(value)
        finally:
            self._syncing = False

    def _update_effective_batch_label(self):
        batch = self._store.get("train_batch_size") or 1
        accum = self._store.get("gradient_accumulation_steps") or 1
        self._effective_batch_label.setText(f"실질 배치 크기(batch × 누적) = {batch} × {accum} = {batch * accum}")

    def _apply_defaults(self):
        for dest, value in BASIC_DEFAULTS.items():
            if self._store.get(dest) is None:
                self._store.set(dest, value, silent=False)
            else:
                # 이미 프리셋 등으로 값이 있으면 위젯에 반영만 한다.
                self._on_store_changed(dest, self._store.get(dest))
        self._update_effective_batch_label()
