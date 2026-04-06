"""Shared audio helpers used across pipeline stages."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


WHISPER_SR = 16_000   # Hz required by WhisperX
TTS_SR = 24_000       # Hz output by XTTS-v2


def load_audio(path: str | Path, sr: int = WHISPER_SR) -> tuple[np.ndarray, int]:
    """Load WAV file. Assumes FFmpeg already converted to target sample rate."""
    samples, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if file_sr != sr:
        raise ValueError(
            f"Expected sample rate {sr}Hz but got {file_sr}Hz in {path}. "
            "Re-run extractor to resample."
        )
    return samples, sr


def save_audio(samples: np.ndarray, sr: int, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sr)


def trim_silence(
    samples: np.ndarray, sr: int, threshold: float = 0.01
) -> np.ndarray:
    """Remove leading/trailing silence using simple amplitude threshold."""
    mask = np.abs(samples) > threshold
    if not mask.any():
        return samples
    start = int(np.argmax(mask))
    end = int(len(mask) - np.argmax(mask[::-1]))
    return samples[start:end]


def extract_reference_clip(
    samples: np.ndarray, sr: int, duration: float = 10.0
) -> np.ndarray:
    """Return up to `duration` seconds from the start of the audio for voice cloning."""
    max_samples = int(duration * sr)
    return samples[:max_samples]
