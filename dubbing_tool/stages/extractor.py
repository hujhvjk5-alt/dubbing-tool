"""Stage 1 — Extract audio from video using FFmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path

from dubbing_tool.utils.audio import WHISPER_SR


def extract_audio(video_path: str | Path, output_path: str | Path) -> Path:
    """
    Extract audio track from video, convert to 16kHz mono WAV.

    Args:
        video_path: Input video file.
        output_path: Where to write the extracted WAV.

    Returns:
        Path to the written WAV file.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",                          # overwrite without asking
        "-i", str(video_path),
        "-vn",                         # no video
        "-ac", "1",                    # mono
        "-ar", str(WHISPER_SR),        # 16 kHz
        "-acodec", "pcm_s16le",        # 16-bit PCM
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed (code {result.returncode}):\n{result.stderr}"
        )

    return output_path


def get_video_duration(video_path: str | Path) -> float:
    """Return video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{result.stderr}")
    return float(result.stdout.strip())
