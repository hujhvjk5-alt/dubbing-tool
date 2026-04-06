"""Orchestrator — wires all pipeline stages together."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from dubbing_tool.models.manager import ModelManager
from dubbing_tool.stages.extractor import extract_audio, get_video_duration
from dubbing_tool.stages.transcriber import transcribe
from dubbing_tool.stages.translator import translate
from dubbing_tool.stages.synthesizer import synthesize
from dubbing_tool.stages.synchronizer import synchronize
from dubbing_tool.stages.mixer import mix
from dubbing_tool.stages.reconstructor import reconstruct
from dubbing_tool.utils.audio import extract_reference_clip, load_audio, save_audio, WHISPER_SR

console = Console()


@dataclass
class PipelineConfig:
    source_lang: str
    target_lang: str
    whisper_model: str = "large-v3"
    translation_model: str = "nllb-200-distilled-600M"
    device: str = "auto"
    models_cache_dir: str = "./models_cache"
    output_dir: str = "./output"
    reference_audio: str | None = None  # optional external speaker reference


def run(
    video_path: str | Path,
    output_path: str | Path,
    config: PipelineConfig,
) -> Path:
    """
    Run the full dubbing pipeline on a video file.

    Args:
        video_path:  Input video.
        output_path: Output dubbed video.
        config:      Pipeline settings.

    Returns:
        Path to the output video.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    manager = ModelManager(cache_dir=config.models_cache_dir)

    with tempfile.TemporaryDirectory(prefix="dubbing_") as tmp:
        tmp = Path(tmp)

        # ── 1. Extract audio ───────────────────────────────────────────
        _log("1/7", "Extracting audio")
        raw_audio = tmp / "audio_raw.wav"
        extract_audio(video_path, raw_audio)
        video_duration = get_video_duration(video_path)

        # ── 1b. Extract reference clip for voice cloning ───────────────
        if config.reference_audio:
            reference_wav = Path(config.reference_audio)
        else:
            reference_wav = tmp / "reference.wav"
            samples, sr = load_audio(raw_audio, sr=WHISPER_SR)
            ref_clip = extract_reference_clip(samples, sr, duration=10.0)
            save_audio(ref_clip, sr, reference_wav)

        # ── 2. Transcribe ──────────────────────────────────────────────
        _log("2/7", "Transcribing")
        transcript_json = tmp / "transcript.json"
        transcribe(
            audio_path=raw_audio,
            output_path=transcript_json,
            manager=manager,
            source_lang=config.source_lang,
            whisper_model=config.whisper_model,
            device=config.device,
        )

        # ── 3. Translate ───────────────────────────────────────────────
        _log("3/7", "Translating")
        translated_json = tmp / "translated.json"
        translate(
            input_path=transcript_json,
            output_path=translated_json,
            manager=manager,
            source_lang=config.source_lang,
            target_lang=config.target_lang,
            model_name=config.translation_model,
        )

        # ── 4. Synthesize TTS ──────────────────────────────────────────
        _log("4/7", "Synthesizing speech")
        segments_dir = tmp / "segments"
        synthesize(
            input_path=translated_json,
            segments_dir=segments_dir,
            reference_audio=reference_wav,
            manager=manager,
            target_lang=config.target_lang,
            device=config.device,
        )

        # ── 5. Synchronize timing (skipped for now — debug) ───────────
        _log("5/7", "Synchronizing timing (skipped)")

        # ── 6. Mix ─────────────────────────────────────────────────────
        _log("6/7", "Mixing audio")
        dubbed_wav = tmp / "dubbed.wav"
        mix(
            input_path=translated_json,
            output_path=dubbed_wav,
            video_duration=video_duration,
        )

        # ── 7. Reconstruct video ───────────────────────────────────────
        _log("7/7", "Reconstructing video")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        reconstruct(
            video_path=video_path,
            dubbed_audio_path=dubbed_wav,
            output_path=output_path,
            video_duration=video_duration,
        )

    console.print(f"[bold green]Done![/bold green] → {output_path}")
    return output_path


def _log(step: str, message: str) -> None:
    console.print(f"[cyan][{step}][/cyan] {message}...")
