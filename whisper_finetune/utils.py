from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).parent.parent


class ModelSize(Enum):
    tiny = "tiny"
    base = "base"
    small = "small"
    medium = "medium"
    large = "large"


# maybe other works as well, idk
SUPPORTED_AUDIO_FORMATS = ["wav", "mp3", "ogg"]


def files_ending_with(path: Path, suffixes: list[str], deep: bool) -> Iterable[Path]:
    if deep:
        pattern = "**/*"
    else:
        pattern = "*"

    for suffix in suffixes:
        yield from path.glob(f"{pattern}.{suffix}")
