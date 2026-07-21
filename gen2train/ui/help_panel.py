"""HTML 형태의 설정 백과사전 사이드 패널."""
from PySide6.QtWidgets import QHBoxLayout, QLineEdit, QPushButton, QTextBrowser, QVBoxLayout, QWidget


class HelpPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        top_row = QHBoxLayout()
        self._search = QLineEdit()
        self._search.setPlaceholderText("설정 이름으로 검색 (예: network_dim)")
        self._search.returnPressed.connect(self._on_search)
        home_btn = QPushButton("목차")
        home_btn.setToolTip("맨 위 목차로 이동")
        home_btn.clicked.connect(self._go_home)
        top_row.addWidget(self._search, 1)
        top_row.addWidget(home_btn)
        layout.addLayout(top_row)

        self._browser = QTextBrowser()
        self._browser.setOpenExternalLinks(False)
        self._browser.setOpenLinks(True)  # 목차의 #anchor 링크 클릭 시 자체적으로 스크롤 이동
        layout.addWidget(self._browser, 1)

    def set_content(self, html: str):
        self._browser.setHtml(html)

    def show_dest(self, dest: str):
        if dest:
            self._browser.scrollToAnchor(dest)

    def _go_home(self):
        self._browser.verticalScrollBar().setValue(0)

    def _on_search(self):
        text = self._search.text().strip()
        if text:
            self._browser.scrollToAnchor(text)
