"""Stage 5 — Time-stretch synthesized segments to match original timing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample

from dubbing_tool.utils.segments import Segment, load_segments, save_segments

MAX_STRETCH = 2.5
MIN_STRETCH = 0.4


def synchronize(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> list[Segment]:
    """
    Time-stretch each segment's audio to exactly fill its original time slot.

    If the TTS audio is shorter than the slot → stretch (slow down).
    If the TTS audio is longer than the slot  → compress (speed up).
    """
    segments = load_segments(input_path)
    output_path = output_path or input_path

    for seg in segments:
        if not seg.audio_file or not Path(seg.audio_file).exists():
            continue

        samples, sr = sf.read(seg.audio_file, dtype="float32")
        tts_duration = len(samples) / sr
        target_duration = seg.duration

        if tts_duration == 0 or target_duration == 0:
            continue

        ratio = tts_duration / target_duration  # >1 means TTS is longer → compress
        ratio = max(MIN_STRETCH, min(MAX_STRETCH, ratio))

        if abs(ratio - 1.0) < 0.02:
            continue

        target_length = int(len(samples) / ratio)
        stretched = resample(samples, target_length).astype(np.float32)
        sf.write(seg.audio_file, stretched, sr, subtype="PCM_16")

    save_segments(segments, output_path)
    return segments
