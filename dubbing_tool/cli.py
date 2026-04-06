"""CLI entry point — python -m dubbing_tool.cli <command>"""

from __future__ import annotations

from pathlib import Path

import click
import yaml
from rich.console import Console

console = Console()

# ── Config loading ─────────────────────────────────────────────────────────────

def _load_config_file(path: str = "config.yaml") -> dict:
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return yaml.safe_load(f) or {}
    return {}


# ── Main group ─────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """Offline video dubbing tool — open-source pipeline."""
    pass


# ── dub ────────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--input",  "-i", "input_path",  required=True,  type=click.Path(exists=True), help="Input video file")
@click.option("--output", "-o", "output_path", required=True,  type=click.Path(), help="Output dubbed video file")
@click.option("--source-lang", "-s", required=True,  help="Source language code, e.g. 'en'")
@click.option("--target-lang", "-t", required=True,  help="Target language code, e.g. 'vi'")
@click.option("--reference-audio", "-r", default=None, type=click.Path(exists=True), help="Optional external speaker reference WAV")
@click.option("--whisper-model",   default=None, help="Whisper model size (large-v3, turbo, medium, base)")
@click.option("--device",          default=None, help="cuda | cpu | auto")
@click.option("--config",          default="config.yaml", help="Path to config.yaml")
def dub(input_path, output_path, source_lang, target_lang, reference_audio, whisper_model, device, config):
    """Run the full dubbing pipeline on a video."""
    from dubbing_tool.pipeline import PipelineConfig, run

    cfg = _load_config_file(config)

    pipeline_config = PipelineConfig(
        source_lang=source_lang,
        target_lang=target_lang,
        whisper_model=whisper_model or cfg.get("whisper_model", "large-v3"),
        translation_model=cfg.get("translation_model", "nllb-200-distilled-600M"),
        device=device or cfg.get("device", "auto"),
        models_cache_dir=cfg.get("models_cache_dir", "./models_cache"),
        output_dir=cfg.get("output_dir", "./output"),
        reference_audio=reference_audio,
    )

    run(input_path, output_path, pipeline_config)


# ── transcribe ─────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--input",  "-i", "input_path",  required=True, type=click.Path(exists=True), help="Input video or WAV file")
@click.option("--output", "-o", "output_path", required=True, type=click.Path(), help="Output transcript.json")
@click.option("--source-lang", "-s", required=True, help="Source language code")
@click.option("--whisper-model", default=None)
@click.option("--device", default=None)
@click.option("--config", default="config.yaml")
def transcribe(input_path, output_path, source_lang, whisper_model, device, config):
    """Stage 2: Transcribe audio/video → transcript.json."""
    import tempfile
    from pathlib import Path as P
    from dubbing_tool.models.manager import ModelManager
    from dubbing_tool.stages.extractor import extract_audio
    from dubbing_tool.stages.transcriber import transcribe as _transcribe

    cfg = _load_config_file(config)
    manager = ModelManager(cfg.get("models_cache_dir", "./models_cache"))

    inp = P(input_path)
    if inp.suffix.lower() in (".mp4", ".mkv", ".avi", ".mov", ".webm"):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav = P(f.name)
        extract_audio(inp, wav)
    else:
        wav = inp

    _transcribe(
        audio_path=wav,
        output_path=output_path,
        manager=manager,
        source_lang=source_lang,
        whisper_model=whisper_model or cfg.get("whisper_model", "large-v3"),
        device=device or cfg.get("device", "auto"),
    )
    console.print(f"[green]Transcript saved →[/green] {output_path}")


# ── translate ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--input",  "-i", "input_path",  required=True, type=click.Path(exists=True), help="transcript.json")
@click.option("--output", "-o", "output_path", required=True, type=click.Path(), help="Output translated.json")
@click.option("--source-lang", "-s", required=True)
@click.option("--target-lang", "-t", required=True)
@click.option("--config", default="config.yaml")
def translate(input_path, output_path, source_lang, target_lang, config):
    """Stage 3: Translate transcript.json → translated.json."""
    from dubbing_tool.models.manager import ModelManager
    from dubbing_tool.stages.translator import translate as _translate

    cfg = _load_config_file(config)
    manager = ModelManager(cfg.get("models_cache_dir", "./models_cache"))

    _translate(
        input_path=input_path,
        output_path=output_path,
        manager=manager,
        source_lang=source_lang,
        target_lang=target_lang,
        model_name=cfg.get("translation_model", "nllb-200-distilled-600M"),
    )
    console.print(f"[green]Translation saved →[/green] {output_path}")


# ── synthesize ─────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--input",  "-i", "input_path",  required=True, type=click.Path(exists=True), help="translated.json")
@click.option("--reference-audio", "-r", required=True, type=click.Path(exists=True), help="Speaker reference WAV")
@click.option("--output-dir", "-o", "output_dir", required=True, type=click.Path(), help="Directory for segment WAV files")
@click.option("--target-lang", "-t", required=True)
@click.option("--device", default=None)
@click.option("--config", default="config.yaml")
def synthesize(input_path, reference_audio, output_dir, target_lang, device, config):
    """Stage 4: Synthesize TTS audio per segment with voice cloning."""
    from dubbing_tool.models.manager import ModelManager
    from dubbing_tool.stages.synthesizer import synthesize as _synthesize

    cfg = _load_config_file(config)
    manager = ModelManager(cfg.get("models_cache_dir", "./models_cache"))

    _synthesize(
        input_path=input_path,
        segments_dir=output_dir,
        reference_audio=reference_audio,
        manager=manager,
        target_lang=target_lang,
        device=device or cfg.get("device", "auto"),
    )
    console.print(f"[green]Segments saved →[/green] {output_dir}")


# ── batch ──────────────────────────────────────────────────────────────────────

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".ts"}

@cli.command()
@click.option("--input-dir",  "-i", "input_dir",  required=True, type=click.Path(exists=True, file_okay=False), help="Directory containing input videos")
@click.option("--output-dir", "-o", "output_dir", required=True, type=click.Path(file_okay=False), help="Directory for dubbed output videos")
@click.option("--source-lang", "-s", required=True)
@click.option("--target-lang", "-t", required=True)
@click.option("--whisper-model", default=None)
@click.option("--device",        default=None)
@click.option("--config",        default="config.yaml")
def batch(input_dir, output_dir, source_lang, target_lang, whisper_model, device, config):
    """Run dubbing pipeline on all videos in a directory, then write a summary log."""
    import time
    import csv
    from datetime import datetime
    from dubbing_tool.pipeline import PipelineConfig, run
    from dubbing_tool.stages.extractor import get_video_duration

    cfg = _load_config_file(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_config = PipelineConfig(
        source_lang=source_lang,
        target_lang=target_lang,
        whisper_model=whisper_model or cfg.get("whisper_model", "large-v3"),
        translation_model=cfg.get("translation_model", "nllb-200-distilled-600M"),
        device=device or cfg.get("device", "auto"),
        models_cache_dir=cfg.get("models_cache_dir", "./models_cache"),
        output_dir=str(output_dir),
    )

    videos = sorted(
        p for p in Path(input_dir).iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not videos:
        console.print(f"[yellow]No video files found in {input_dir}[/yellow]")
        return

    console.print(f"Found [bold]{len(videos)}[/bold] video(s) to process.\n")

    results = []

    for idx, video in enumerate(videos, 1):
        console.rule(f"[{idx}/{len(videos)}] {video.name}")
        out_path = output_dir / video.name

        try:
            video_duration = get_video_duration(video)
        except Exception:
            video_duration = 0.0

        start_time = time.time()
        status = "ok"
        error_msg = ""

        try:
            run(str(video), str(out_path), pipeline_config)
        except Exception as e:
            status = "error"
            error_msg = str(e)
            console.print(f"[red]Error:[/red] {e}")

        processing_time = time.time() - start_time

        results.append({
            "file": video.name,
            "video_duration_s": round(video_duration, 2),
            "processing_time_s": round(processing_time, 2),
            "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
            "error": error_msg,
        })

        console.print(
            f"  [green]✓[/green] {video.name} | "
            f"video={video_duration:.1f}s | "
            f"processed in {processing_time:.1f}s | "
            f"status={status}"
        )

    # Write summary log
    log_path = output_dir / f"batch_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Print summary table
    console.rule("Summary")
    total = len(results)
    ok = sum(1 for r in results if r["status"] == "ok")
    failed = total - ok
    total_processing = sum(r["processing_time_s"] for r in results)

    console.print(f"  Total videos   : {total}")
    console.print(f"  Succeeded      : [green]{ok}[/green]")
    console.print(f"  Failed         : [red]{failed}[/red]")
    console.print(f"  Total time     : {total_processing:.1f}s")
    console.print(f"  Log saved →    : {log_path}")


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    cli()


if __name__ == "__main__":
    main()
