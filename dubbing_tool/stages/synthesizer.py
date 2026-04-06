"""Stage 4 — TTS with voice cloning using Coqui XTTS-v2."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from dubbing_tool.utils.segments import Segment, load_segments, save_segments

if TYPE_CHECKING:
    from dubbing_tool.models.manager import ModelManager

# XTTS language codes
LANG_TO_XTTS = {
    "en": "en", "vi": "vi", "zh": "zh-cn", "ja": "ja",
    "ko": "ko", "fr": "fr", "de": "de", "es": "es",
    "ru": "ru", "ar": "ar", "th": "th", "id": "id",
    "pt": "pt", "pl": "pl", "it": "it", "nl": "nl",
    "cs": "cs", "tr": "tr", "hu": "hu",
}


def synthesize(
    input_path: str | Path,
    segments_dir: str | Path,
    reference_audio: str | Path,
    manager: "ModelManager",
    target_lang: str,
    device: str = "auto",
) -> list[Segment]:
    """
    Generate TTS audio for each segment, cloning the voice from reference_audio.

    Args:
        input_path:       translated.json from translator.
        segments_dir:     Directory to write per-segment WAV files.
        reference_audio:  Short WAV clip of the original speaker's voice.
        manager:          ModelManager.
        target_lang:      Language code for TTS, e.g. 'vi'.
        device:           'cuda', 'cpu', or 'auto'.

    Returns:
        List of Segment objects with audio_file paths filled.
    """
    segments = load_segments(input_path)
    segments_dir = Path(segments_dir)
    segments_dir.mkdir(parents=True, exist_ok=True)

    xtts_lang = _resolve_xtts_lang(target_lang)
    tts = manager.load_xtts(device)

    for seg in segments:
        if not seg.translated_text.strip():
            seg.audio_file = ""
            continue

        out_path = segments_dir / f"seg_{seg.id:04d}.wav"
        tts.tts_to_file(
            text=seg.translated_text,
            speaker_wav=str(reference_audio),
            language=xtts_lang,
            file_path=str(out_path),
        )
        seg.audio_file = str(out_path)

        # Debug: print sample rate of first generated segment
        if seg.id == 0:
            import soundfile as sf
            info = sf.info(str(out_path))
            print(f"  [debug] seg_0000: sr={info.samplerate}Hz, duration={info.duration:.2f}s, format={info.subtype}")

    save_segments(segments, input_path)  # update in place with audio_file paths
    return segments


def _resolve_xtts_lang(lang: str) -> str:
    if lang in LANG_TO_XTTS:
        return LANG_TO_XTTS[lang]
    # Allow passing XTTS codes directly
    return lang
