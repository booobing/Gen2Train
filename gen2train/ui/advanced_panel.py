"""sd-scripts의 argparse 전체를 introspection으로 읽어 자동 생성하는 고급 파라미터 탭."""
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..core import arg_introspect
from ..core.help_texts import GROUP_DESCRIPTIONS, get_help
from ..core.param_store import ParamStore
from .widgets import ArgFieldWidget, CollapsibleSection

DEFAULT_EXPANDED_GROUPS = {"LoRA 네트워크", "옵티마이저 / LR 스케줄러"}
HIGHLIGHT_STYLE = "background-color: #fff3b0; font-weight: bold; border-radius: 3px;"


class AdvancedPanel(QWidget):
    """model_type(sd/sdxl)에 맞는 전체 CLI 파라미터를 그룹별 접이식 폼으로 노출한다.

    파라미터가 176개 안팎이라 스크롤로 찾기 번거로우므로, 상단 검색창에 이름(dest/flag)이나
    한국어 설명 일부를 입력하면 일치하는 항목의 그룹을 자동으로 펼치고 그 위치로 스크롤 + 잠깐
    하이라이트한다.
    """

    def __init__(self, store: ParamStore, model_type: str = "sd", parent=None):
        super().__init__(parent)
        self._store = store
        self._model_type = model_type
        self._field_widgets: dict[str, ArgFieldWidget] = {}
        self._label_widgets: dict[str, QLabel] = {}
        self._section_by_dest: dict[str, CollapsibleSection] = {}
        self._specs_by_dest: dict = {}
        self._syncing = False

        self._search_matches: list = []
        self._search_index = -1
        self._highlighted_label = None
        self._highlight_timer = QTimer(self)
        self._highlight_timer.setSingleShot(True)
        self._highlight_timer.timeout.connect(self._clear_highlight)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(self._build_search_row())

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        outer.addWidget(self._scroll)

        self._store.valueChanged.connect(self._on_store_changed)
        self.set_model_type(model_type)

    def _build_search_row(self):
        row = QHBoxLayout()
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("파라미터 이름/설명 검색 (예: network_dim, 학습률)")
        self._search_input.textChanged.connect(self._on_search_text_changed)
        self._search_input.returnPressed.connect(self._go_to_next_match)
        self._search_result_label = QLabel("")
        self._search_result_label.setStyleSheet("color: gray;")
        self._search_result_label.setMinimumWidth(80)
        prev_btn = QPushButton("이전")
        prev_btn.clicked.connect(self._go_to_prev_match)
        next_btn = QPushButton("다음")
        next_btn.clicked.connect(self._go_to_next_match)
        row.addWidget(self._search_input, 1)
        row.addWidget(self._search_result_label)
        row.addWidget(prev_btn)
        row.addWidget(next_btn)
        return row

    def set_model_type(self, model_type: str):
        self._model_type = model_type
        self._field_widgets = {}
        self._label_widgets = {}
        self._section_by_dest = {}
        self._specs_by_dest = {}
        self._search_matches = []
        self._search_index = -1
        self._search_result_label.setText("")

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(6)

        specs = arg_introspect.get_arg_specs(model_type)
        specs_by_group: dict[str, list] = {}
        for spec in specs:
            specs_by_group.setdefault(spec.group, []).append(spec)
            self._specs_by_dest[spec.dest] = spec

        for group in arg_introspect.group_order():
            group_specs = specs_by_group.get(group)
            if not group_specs:
                continue
            section = CollapsibleSection(f"{group} ({len(group_specs)})", expanded=group in DEFAULT_EXPANDED_GROUPS)
            section.set_description(GROUP_DESCRIPTIONS.get(group, ""))
            form = QFormLayout()
            form.setSpacing(6)
            for spec in group_specs:
                tooltip = get_help(spec.dest, spec.help)
                initial = self._store.get(spec.dest, spec.default)
                field = ArgFieldWidget(spec, value=initial)
                field.setToolTip(tooltip)
                field.changed.connect(lambda value, dest=spec.dest: self._on_field_changed(dest, value))
                self._field_widgets[spec.dest] = field

                label = QLabel(spec.flag.lstrip("-"))
                label.setToolTip(tooltip)
                form.addRow(label, field)
                self._label_widgets[spec.dest] = label
                self._section_by_dest[spec.dest] = section

                if self._store.get(spec.dest) is None and spec.default is not None:
                    self._store.set(spec.dest, spec.default, silent=True)

            section_content = QWidget()
            section_content.setLayout(form)
            section.add_widget(section_content)
            content_layout.addWidget(section)

        content_layout.addStretch(1)
        self._scroll.setWidget(content)

    def _on_field_changed(self, dest, value):
        if self._syncing:
            return
        self._store.set(dest, value)

    def _on_store_changed(self, dest, value):
        field = self._field_widgets.get(dest)
        if field is None:
            return
        self._syncing = True
        try:
            field.set_value(value)
        finally:
            self._syncing = False

    # --------------------------------------------------------------- 검색

    def _on_search_text_changed(self, text: str):
        query = text.strip().lower()
        self._clear_highlight()
        if not query:
            self._search_matches = []
            self._search_index = -1
            self._search_result_label.setText("")
            return

        matches = []
        for dest, spec in self._specs_by_dest.items():
            haystack = " ".join([dest, spec.flag, get_help(dest, spec.help)]).lower()
            if query in haystack:
                matches.append(dest)
        self._search_matches = matches
        self._search_index = -1
        if matches:
            self._go_to_next_match()
        else:
            self._search_result_label.setText("일치 항목 없음")

    def _go_to_next_match(self):
        if not self._search_matches:
            return
        self._search_index = (self._search_index + 1) % len(self._search_matches)
        self._show_match()

    def _go_to_prev_match(self):
        if not self._search_matches:
            return
        self._search_index = (self._search_index - 1) % len(self._search_matches)
        self._show_match()

    def _show_match(self):
        dest = self._search_matches[self._search_index]
        self._search_result_label.setText(f"{self._search_index + 1}/{len(self._search_matches)}개 일치")

        section = self._section_by_dest.get(dest)
        if section is not None and not section.is_expanded():
            section.set_expanded(True)

        field = self._field_widgets.get(dest)
        label = self._label_widgets.get(dest)
        target = field or label
        if target is None:
            return

        # 섹션을 막 펼친 직후에는 레이아웃이 아직 갱신되지 않았을 수 있어, 다음 이벤트
        # 루프 틱으로 스크롤을 미룬다.
        QTimer.singleShot(0, lambda: self._scroll.ensureWidgetVisible(target, 50, 50))

        self._clear_highlight()
        if label is not None:
            label.setStyleSheet(HIGHLIGHT_STYLE)
            self._highlighted_label = label
            self._highlight_timer.start(2000)

    def _clear_highlight(self):
        if self._highlighted_label is not None:
            self._highlighted_label.setStyleSheet("")
            self._highlighted_label = None
