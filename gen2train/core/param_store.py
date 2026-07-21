"""기본 탭 / 고급 탭이 공유하는 단일 파라미터 저장소.

두 패널의 위젯이 같은 dest(예: "resolution")를 다룰 때 값이 어긋나지 않도록,
값 변경은 항상 이 스토어를 거치고 valueChanged 시그널로 서로에게 반영된다.
"""
from PySide6.QtCore import QObject, Signal


class ParamStore(QObject):
    valueChanged = Signal(str, object)

    def __init__(self):
        super().__init__()
        self._values: dict = {}

    def get(self, dest, default=None):
        return self._values.get(dest, default)

    def set(self, dest, value, *, silent=False):
        if dest in self._values and self._values[dest] == value:
            return
        self._values[dest] = value
        if not silent:
            self.valueChanged.emit(dest, value)

    def set_many(self, values: dict, *, silent=True):
        for dest, value in values.items():
            self.set(dest, value, silent=silent)

    def as_dict(self) -> dict:
        return dict(self._values)

    def load(self, values: dict):
        self._values = dict(values)
