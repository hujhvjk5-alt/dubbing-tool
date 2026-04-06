"""Segment data structure shared across all pipeline stages."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Segment:
    id: int
    start: float          # seconds
    end: float            # seconds
    original_text: str
    translated_text: str = ""
    speaker_id: str = "SPEAKER_00"
    audio_file: str = ""  # path to synthesized WAV for this segment

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        return cls(
            id=d["id"],
            start=d["start"],
            end=d["end"],
            original_text=d["original_text"],
            translated_text=d.get("translated_text", ""),
            speaker_id=d.get("speaker_id", "SPEAKER_00"),
            audio_file=d.get("audio_file", ""),
        )


def save_segments(segments: list[Segment], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in segments], f, ensure_ascii=False, indent=2)


def load_segments(path: str | Path) -> list[Segment]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Segment.from_dict(d) for d in data]
