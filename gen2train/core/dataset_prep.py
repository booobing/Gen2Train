"""결함/양품 이미지 폴더를 kohya(sd-scripts) DreamBooth 방식 학습 폴더로 변환한다.

sd-scripts는 `train_data_dir` 아래 `{repeat}_{concept}` 형태의 하위 폴더를 두면
각 이미지를 epoch당 repeat번 학습하는 서브셋으로 자동 인식한다
(library/train_util.py의 DreamBoothSubset / generate_dreambooth_subsets_config_by_subdirs 참고).
이 모듈은 그 규칙에 맞춰 폴더를 만들고, 이미지마다 트리거 단어를 담은 캡션(.txt)을 생성한다.
"""
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .. import settings

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

# sd-scripts의 --caption_extension 기본값은 ".caption"이라 여기서 만드는 파일과 어긋난다.
# trainer.build_command()가 항상 이 값으로 --caption_extension을 고정 전달하므로 이 상수가 유일한 출처다.
CAPTION_EXTENSION = ".txt"


class DatasetPrepError(Exception):
    pass


@dataclass
class DatasetPrepResult:
    train_data_dir: Path
    reg_data_dir: Optional[Path]
    num_defect_images: int
    num_good_images: int


def sanitize_token(text: str, fallback: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^0-9A-Za-z_\-가-힣]", "", text)
    return text or fallback


def sanitize_run_name(text: str) -> str:
    return sanitize_token(text, fallback="run")


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _write_subset(
    images: list[Path],
    dest_root: Path,
    repeat: int,
    concept_token: str,
    caption_fn: Callable[[Path], str],
    caption_extension: str,
) -> Path:
    subset_dir = dest_root / f"{repeat}_{concept_token}"
    subset_dir.mkdir(parents=True, exist_ok=True)
    for src in images:
        dst = subset_dir / src.name
        shutil.copy2(src, dst)
        caption_path = subset_dir / f"{src.stem}{caption_extension}"
        caption_path.write_text(caption_fn(src), encoding="utf-8")
    return subset_dir


def _make_defect_caption_fn(
    trigger_word: str,
    extra_tags: str,
    use_existing_captions: bool,
    existing_caption_extension: str,
) -> Callable[[Path], str]:
    """이미지별 캡션 문자열을 만든다.

    use_existing_captions가 켜져 있으면, 원본 이미지 옆의 기존 캡션 파일(예: WD14 태거로
    미리 만들어둔 결함 종류별 태그)을 읽어 트리거 단어 뒤에 이어붙인다. 이미지마다 결함
    종류(스크래치/덴트 등)가 달라 캡션을 따로 관리하고 싶은 경우를 위한 옵션이다.
    파일이 없으면 트리거 단어(+추가 태그)만 사용하는 기존 동작으로 자연스럽게 대체된다.
    """

    def caption_fn(src: Path) -> str:
        parts = [trigger_word]
        if use_existing_captions:
            existing_path = src.with_suffix(existing_caption_extension)
            if existing_path.exists():
                existing_text = existing_path.read_text(encoding="utf-8").strip()
                if existing_text:
                    parts.append(existing_text)
        if extra_tags.strip():
            parts.append(extra_tags.strip())
        return ", ".join(parts)

    return caption_fn


def prepare_dataset(
    run_name: str,
    defect_dir: Path,
    trigger_word: str,
    repeat: int = 10,
    good_dir: Optional[Path] = None,
    use_good_as_reg: bool = True,
    class_word: str = "normal product",
    reg_repeat: int = 1,
    extra_tags: str = "",
    use_existing_captions: bool = False,
    existing_caption_extension: str = ".txt",
    caption_extension: str = CAPTION_EXTENSION,
) -> DatasetPrepResult:
    defect_dir = Path(defect_dir)
    trigger_word = trigger_word.strip()
    if not trigger_word:
        raise DatasetPrepError("트리거 단어를 입력하세요.")

    defect_images = list_images(defect_dir)
    if not defect_images:
        raise DatasetPrepError(f"결함 이미지 폴더에서 이미지를 찾을 수 없습니다: {defect_dir}")

    run_token = sanitize_run_name(run_name)
    run_dir = settings.DATASETS_DIR / run_token
    if run_dir.exists():
        shutil.rmtree(run_dir)  # Gen2Train이 자체 생성/관리하는 파생 데이터라 재실행 시 초기화해도 안전함
    run_dir.mkdir(parents=True, exist_ok=True)

    train_root = run_dir / "img"
    _write_subset(
        images=defect_images,
        dest_root=train_root,
        repeat=repeat,
        concept_token=sanitize_token(trigger_word, fallback="defect"),
        caption_fn=_make_defect_caption_fn(
            trigger_word=trigger_word,
            extra_tags=extra_tags,
            use_existing_captions=use_existing_captions,
            existing_caption_extension=existing_caption_extension,
        ),
        caption_extension=caption_extension,
    )

    reg_root: Optional[Path] = None
    num_good_images = 0
    good_dir_path = Path(good_dir) if good_dir else None
    if use_good_as_reg and good_dir_path is not None:
        good_images = list_images(good_dir_path)
        num_good_images = len(good_images)
        if good_images:
            reg_root = run_dir / "reg"
            reg_caption = class_word.strip() or "normal product"
            _write_subset(
                images=good_images,
                dest_root=reg_root,
                repeat=reg_repeat,
                concept_token=sanitize_token(class_word, fallback="normal"),
                caption_fn=lambda _src, _text=reg_caption: _text,
                caption_extension=caption_extension,
            )

    return DatasetPrepResult(
        train_data_dir=train_root,
        reg_data_dir=reg_root,
        num_defect_images=len(defect_images),
        num_good_images=num_good_images,
    )
