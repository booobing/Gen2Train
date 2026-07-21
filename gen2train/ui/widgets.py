"""여러 패널에서 재사용하는 공용 위젯."""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

NONE_DISPLAY = "(기본값 사용)"


class FolderPickerRow(QWidget):
    """라벨 + 경로 입력창 + '찾아보기' 버튼."""

    pathChanged = Signal(str)

    def __init__(self, label: str, placeholder: str = "", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel(label)
        self._label.setMinimumWidth(110)
        self._edit = QLineEdit()
        self._edit.setPlaceholderText(placeholder)
        self._edit.textChanged.connect(self.pathChanged.emit)
        browse_btn = QPushButton("찾아보기...")
        browse_btn.clicked.connect(self._browse)

        layout.addWidget(self._label)
        layout.addWidget(self._edit, 1)
        layout.addWidget(browse_btn)

    def _browse(self):
        path = QFileDialog.getExistingDirectory(self, f"{self._label.text()} 선택")
        if path:
            self._edit.setText(path)

    def path(self) -> str:
        return self._edit.text().strip()

    def set_path(self, path: str):
        self._edit.setText(path or "")


class FilePickerRow(QWidget):
    """라벨 + 경로 입력창 + '찾아보기' 버튼 (파일 선택)."""

    pathChanged = Signal(str)

    def __init__(self, label: str, name_filter: str = "All files (*.*)", placeholder: str = "", parent=None):
        super().__init__(parent)
        self._name_filter = name_filter
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel(label)
        self._label.setMinimumWidth(110)
        self._edit = QLineEdit()
        self._edit.setPlaceholderText(placeholder)
        self._edit.textChanged.connect(self.pathChanged.emit)
        browse_btn = QPushButton("찾아보기...")
        browse_btn.clicked.connect(self._browse)

        layout.addWidget(self._label)
        layout.addWidget(self._edit, 1)
        layout.addWidget(browse_btn)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, f"{self._label.text()} 선택", filter=self._name_filter)
        if path:
            self._edit.setText(path)

    def path(self) -> str:
        return self._edit.text().strip()

    def set_path(self, path: str):
        self._edit.setText(path or "")


class ArgFieldWidget(QWidget):
    """arg_introspect.ArgSpec 하나에 대응하는 입력 위젯.

    bool -> 체크박스, choice -> 콤보박스, 그 외(int/float/str/nargs) -> 텍스트 입력.
    숫자 필드는 sd-scripts 대부분이 None(=스크립트 기본값 사용)을 허용하므로,
    스핀박스 대신 텍스트 입력으로 통일해 '비워두면 기본값' 의미를 그대로 살린다.
    """

    changed = Signal(object)

    def __init__(self, spec, value=None, parent=None):
        super().__init__(parent)
        self.spec = spec
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if spec.kind == "bool":
            self._control = QCheckBox()
            self._control.toggled.connect(lambda v: self.changed.emit(v))
        elif spec.kind == "choice" and spec.nargs not in ("*", "+"):
            self._control = QComboBox()
            for choice in spec.choices:
                self._control.addItem(NONE_DISPLAY if choice is None else str(choice), choice)
            self._control.currentIndexChanged.connect(lambda _i: self.changed.emit(self._control.currentData()))
        else:
            self._control = QLineEdit()
            if spec.nargs in ("*", "+"):
                self._control.setPlaceholderText("공백으로 구분하여 여러 값 입력")
            self._control.textChanged.connect(lambda _t: self.changed.emit(self.get_value()))

        # 이 컨트롤에 포커스가 갈 때 도움말 패널이 어떤 항목인지 알 수 있도록 dest를 심어둔다.
        self._control.setProperty("g2t_dest", spec.dest)

        layout.addWidget(self._control)
        self.set_value(value if value is not None else spec.default)

    def get_value(self):
        if isinstance(self._control, QCheckBox):
            return self._control.isChecked()
        if isinstance(self._control, QComboBox):
            return self._control.currentData()
        text = self._control.text().strip()
        if text == "":
            return None
        if self.spec.nargs in ("*", "+"):
            return text
        if self.spec.kind == "int":
            try:
                return int(text)
            except ValueError:
                return None
        if self.spec.kind == "float":
            try:
                return float(text)
            except ValueError:
                return None
        return text

    def set_value(self, value):
        if isinstance(self._control, QCheckBox):
            self._control.setChecked(bool(value))
        elif isinstance(self._control, QComboBox):
            idx = self._control.findData(value)
            self._control.setCurrentIndex(idx if idx >= 0 else 0)
        else:
            self._control.blockSignals(True)
            self._control.setText("" if value is None else str(value))
            self._control.blockSignals(False)


class CollapsibleSection(QWidget):
    """제목을 누르면 접히고 펼쳐지는 그룹 섹션. 고급 탭의 그룹별 묶음에 사용."""

    def __init__(self, title: str, expanded: bool = False, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._toggle = QToolButton()
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._toggle.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self._toggle.clicked.connect(self._on_toggled)

        self._content = QFrame()
        self._content.setVisible(expanded)
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(20, 4, 4, 8)

        outer.addWidget(self._toggle)
        outer.addWidget(self._content)

    def _on_toggled(self, checked: bool):
        self._content.setVisible(checked)
        self._toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

    def set_expanded(self, expanded: bool):
        # 검색 결과로 이동할 때처럼, 버튼 클릭 없이 프로그램적으로 펼치기/접기 위한 메서드.
        self._toggle.setChecked(expanded)
        self._on_toggled(expanded)

    def is_expanded(self) -> bool:
        return self._toggle.isChecked()

    def add_widget(self, widget: QWidget):
        self._content_layout.addWidget(widget)

    def content_layout(self):
        return self._content_layout

    def set_description(self, text: str):
        # 자식 QToolButton 위에서 마우스를 올렸을 때 뜨도록, 버튼 자체에 툴팁을 건다.
        self._toggle.setToolTip(text)
