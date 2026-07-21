"""파라미터 프리셋 저장/불러오기."""
import json
from pathlib import Path

from .. import settings

LAST_USED_NAME = "_last_used"


def _sanitize(name: str) -> str:
    keep = "-_. "
    cleaned = "".join(c for c in name if c.isalnum() or c in keep).strip()
    return cleaned or "preset"


def _path_for(name: str) -> Path:
    return settings.PRESETS_DIR / f"{_sanitize(name)}.json"


def list_presets() -> list[str]:
    settings.ensure_dirs()
    return sorted(
        p.stem for p in settings.PRESETS_DIR.glob("*.json") if p.stem != LAST_USED_NAME
    )


def save_preset(name: str, params: dict) -> Path:
    settings.ensure_dirs()
    path = _path_for(name)
    path.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_preset(name: str) -> dict:
    path = _path_for(name)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def delete_preset(name: str) -> None:
    path = _path_for(name)
    if path.exists():
        path.unlink()


def save_last_used(params: dict) -> None:
    save_preset(LAST_USED_NAME, params)


def load_last_used() -> dict:
    return load_preset(LAST_USED_NAME)
