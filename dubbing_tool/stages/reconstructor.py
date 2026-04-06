"""Stage 7 — Replace original audio track with dubbed audio using FFmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path


def reconstruct(
    video_path: str | Path,
    dubbed_audio_path: str | Path,
    output_path: str | Path,
    video_duration: float,
) -> Path:
    video_path = Path(video_path)
    dubbed_audio_path = Path(dubbed_audio_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(dubbed_audio_path),
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-t", str(video_duration),   # exact video duration, no -shortest
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg reconstruct failed (code {result.returncode}):\n{result.stderr}"
        )

    return output_path
