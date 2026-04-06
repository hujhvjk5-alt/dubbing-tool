"""Stage 2 — Transcribe audio to word-level timestamped segments using WhisperX."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from dubbing_tool.utils.segments import Segment, save_segments

if TYPE_CHECKING:
    from dubbing_tool.models.manager import ModelManager


def transcribe(
    audio_path: str | Path,
    output_path: str | Path,
    manager: "ModelManager",
    source_lang: str,
    whisper_model: str = "large-v3",
    device: str = "auto",
) -> list[Segment]:
    """
    Transcribe audio with WhisperX, align to word-level timestamps.

    Args:
        audio_path:    16kHz mono WAV from extractor.
        output_path:   Where to write transcript.json.
        manager:       ModelManager for loading/caching models.
        source_lang:   Source language code, e.g. 'en', 'vi', 'zh'.
        whisper_model: Whisper model size.
        device:        'cuda', 'cpu', or 'auto'.

    Returns:
        List of Segment objects with original_text and timestamps.
    """
    import whisperx

    device = manager.resolve_device(device)

    # Step 1: transcribe
    model = manager.load_whisper(whisper_model, device)
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, language=source_lang, batch_size=16)

    # Step 2: forced alignment for accurate word-level timestamps
    align_model, metadata = manager.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    # Step 3: convert to Segment list
    segments = _build_segments(result["segments"])
    save_segments(segments, output_path)

    return segments


def _build_segments(raw_segments: list[dict]) -> list[Segment]:
    segments = []
    for i, seg in enumerate(raw_segments):
        text = seg.get("text", "").strip()
        if not text:
            continue
        segments.append(
            Segment(
                id=i,
                start=seg["start"],
                end=seg["end"],
                original_text=text,
                speaker_id=seg.get("speaker", "SPEAKER_00"),
            )
        )
    return segments
