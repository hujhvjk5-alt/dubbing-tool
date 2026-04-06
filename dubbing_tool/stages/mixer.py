"""Stage 6 — Assemble per-segment WAVs into a single dubbed audio track using FFmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path

from dubbing_tool.utils.segments import load_segments

_DEFAULT_SR = 24_000


def mix(
    input_path: str | Path,
    output_path: str | Path,
    video_duration: float,
) -> Path:
    """
    Place each segment at its original timestamp using FFmpeg adelay + amix.
    This avoids all sample-rate and format issues that pydub can introduce.
    """
    segments = load_segments(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    valid = [s for s in segments if s.audio_file and Path(s.audio_file).exists()]

    if not valid:
        # No segments — write silence
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"anullsrc=r={_DEFAULT_SR}:cl=mono",
                "-t", str(video_duration),
                "-c:a", "pcm_s16le",
                str(output_path),
            ],
            check=True, capture_output=True,
        )
        return output_path

    # Detect sample rate from first segment
    probe = _probe_sr(valid[0].audio_file)
    sr = probe or _DEFAULT_SR

    # Build filter_complex:
    # Each stream gets delayed to its start time, then all are amixed
    cmd = ["ffmpeg", "-y"]
    for seg in valid:
        cmd += ["-i", seg.audio_file]

    filter_parts = []
    for i, seg in enumerate(valid):
        delay_ms = int(seg.start * 1000)
        filter_parts.append(f"[{i}:a]adelay={delay_ms}|{delay_ms}[d{i}]")

    mix_in = "".join(f"[d{i}]" for i in range(len(valid)))
    filter_parts.append(
        f"{mix_in}amix=inputs={len(valid)}:normalize=0:duration=longest[aout]"
    )

    cmd += [
        "-filter_complex", ";".join(filter_parts),
        "-map", "[aout]",
        "-t", str(video_duration + 1.0),
        "-c:a", "pcm_s16le",
        "-ar", str(sr),
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg mix failed:\n{result.stderr}")

    return output_path


def _probe_sr(audio_file: str) -> int | None:
    """Return sample rate of a WAV file using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_file),
        ],
        capture_output=True, text=True,
    )
    try:
        return int(result.stdout.strip())
    except (ValueError, AttributeError):
        return None
