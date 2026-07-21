"""Gen2Train 진입점. run.bat이 kohya_ss 공유 venv의 python으로 이 파일을 실행한다."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtWidgets import QApplication

from gen2train import settings
from gen2train.ui.main_window import MainWindow


def main():
    settings.ensure_dirs()
    app = QApplication(sys.argv)
    app.setApplicationName("Gen2Train")

    window = MainWindow()
    window.resize(1150, 900)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
