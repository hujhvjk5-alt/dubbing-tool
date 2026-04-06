# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Offline video dubbing CLI tool — takes a video, transcribes speech, translates it, synthesizes dubbed audio with voice cloning, and outputs a new video. No web UI, no cloud API calls, fully self-contained.

## Commands

```bash
# Install dependencies (requires Python 3.12, conda recommended)
pip install -r requirements.txt

# Single video
python -m dubbing_tool.cli dub -i video.mp4 -s vi -t en -o output/dubbed.mp4

# Batch — process entire directory, writes CSV log to output-dir
python -m dubbing_tool.cli batch -i videos/ -o output/ -s vi -t en

# Run individual stages for debugging
python -m dubbing_tool.cli transcribe -i video.mp4 -s vi -o transcript.json
python -m dubbing_tool.cli translate -i transcript.json -s vi -t en -o translated.json
python -m dubbing_tool.cli synthesize -i translated.json -r reference.wav -o segments/ -t en

# Run tests
pytest tests/
pytest tests/test_pipeline.py::test_transcribe

# Lint
ruff check .
ruff format .
```

## Architecture

The pipeline is a linear chain of stages. Each stage reads/writes structured JSON or audio/video files so stages can be run and debugged independently.

```
Input Video
    │
    ▼
[1] Extractor      – FFmpeg: strips audio track → WAV 16kHz mono
    │
    ▼
[2] Transcriber    – WhisperX: produces word-level timestamped segments → transcript.json
    │
    ▼
[3] Translator     – NLLB-200 via CTranslate2 (offline): translates each segment → translated.json
    │
    ▼
[4] Synthesizer    – Coqui XTTS-v2: TTS with voice cloning from extracted reference audio
    │                produces per-segment WAV files (PCM_16 at 24kHz)
    ▼
[5] Synchronizer   – scipy.signal.resample: time-stretches each segment to fit original timing
    │                writes back as PCM_16 (important: avoids float32 format confusion downstream)
    ▼
[6] Mixer          – FFmpeg adelay+amix: places each segment at its original timestamp
    │                uses FFmpeg directly to avoid pydub sample-rate issues
    ▼
[7] Reconstructor  – FFmpeg: replaces original audio track → output video
```

### Module layout

```
dubbing_tool/
  cli.py            – Click CLI entry point (commands: dub, batch, transcribe, translate, synthesize)
  pipeline.py       – Orchestrates the full pipeline, manages temp dir, passes context between stages
  stages/
    extractor.py    – FFmpeg audio extraction
    transcriber.py  – WhisperX transcription
    translator.py   – NLLB / CTranslate2 translation
    synthesizer.py  – XTTS TTS + voice cloning
    synchronizer.py – Audio time-stretching via scipy (writes PCM_16)
    mixer.py        – FFmpeg-based segment assembly (adelay + amix)
    reconstructor.py– Final video assembly (uses -t video_duration, not -shortest)
  models/
    manager.py      – Model download/caching; NLLB converted via ct2-transformers-converter
  utils/
    audio.py        – Audio helpers using soundfile only (no librosa — lzma DLL issues on Windows)
    segments.py     – Segment dataclass and JSON load/save
```

### Key data structure

Segments JSON passed between stages:

```json
[
  {
    "id": 0,
    "start": 1.24,
    "end": 3.80,
    "original_text": "Xin chào, bạn khỏe không?",
    "translated_text": "Hello, how are you?",
    "speaker_id": "SPEAKER_00",
    "audio_file": "segments/seg_0000.wav"
  }
]
```

## Tech Stack

| Purpose | Library |
|---|---|
| Video/audio I/O | `ffmpeg` (subprocess) |
| Transcription + word timestamps | `whisperx` |
| Translation (offline) | `ctranslate2` + NLLB-200 model |
| TTS + voice cloning | `coqui-tts` (installed from GitHub, not PyPI) |
| Time-stretching | `scipy.signal.resample` |
| Audio file I/O | `soundfile` |
| CLI | `click` |
| Config | `pyyaml` + `pydantic-settings` |

## Windows / Environment Notes

- **Python 3.12** required — Python 3.14 is not supported by most ML packages (torch, coqui-tts, whisperx)
- **conda environment** recommended: `conda create -n dubbing python=3.12`
- **FFmpeg** must be installed system-wide (not via conda — causes DLL conflicts with gdk_pixbuf). Use `winget install ffmpeg`
- **torchvision** must be uninstalled — it pulls in `lzma` which has DLL issues on Windows: `pip uninstall torchvision -y`
- **lzma DLL issue**: if `ImportError: DLL load failed while importing _lzma`, run `conda install -c conda-forge xz` or add `%CONDA_PREFIX%\Library\bin` to PATH
- **coqui-tts** is not on PyPI — installed from GitHub: `coqui-tts @ git+https://github.com/idiap/coqui-ai-TTS.git`
- **Mixer uses FFmpeg, not pydub** — pydub caused sample-rate mismatches (11025Hz vs 24kHz) that made audio play at wrong speed
- **Synchronizer writes PCM_16** explicitly — XTTS outputs int16, soundfile reads as float32; must write back as PCM_16 or downstream tools misinterpret the format

## Model Management

Models downloaded on first use, cached in `./models_cache/`. NLLB is converted to CTranslate2 format via `ct2-transformers-converter` on first load (check is `models_cache/nllb/.../ct2/model.bin` exists, not just the directory).

GPU auto-detected; falls back to CPU with a warning.

## Configuration

`config.yaml` at repo root:

```yaml
whisper_model: large-v3        # large-v3 | turbo | medium | base | small
translation_model: nllb-200-distilled-600M
tts_model: tts_models/multilingual/multi-dataset/xtts_v2
device: auto                   # cuda | cpu | auto
output_dir: ./output
models_cache_dir: ./models_cache
```

CLI flags always override config file values.
