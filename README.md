# dubbing-tool

Offline video dubbing pipeline — transcribe, translate, and re-voice any video using open-source AI models. No cloud APIs, no internet required after the first model download.

Inspired by services like ElevenLabs Dubbing, built entirely with open-source components.

## Features

- **Fully offline** after initial model download (~7 GB total)
- **Voice cloning** — dubbed audio uses the original speaker's voice (Coqui XTTS-v2)
- **12 source languages** — any language Whisper supports
- **12 target languages** — EN, VI, ZH, JA, KO, FR, DE, ES, RU, AR, TH, ID
- **Batch mode** — process entire directories with a single command, outputs a CSV summary log
- **Modular pipeline** — each stage can be run independently for debugging

## Pipeline

```
Video → [Extract audio] → [Transcribe] → [Translate] → [Synthesize TTS] → [Sync timing] → [Mix] → [Rebuild video]
         FFmpeg            WhisperX        NLLB-200       Coqui XTTS-v2     scipy           FFmpeg   FFmpeg
```

## Requirements

- Python **3.12** (3.13+ not supported by current ML dependencies)
- [FFmpeg](https://ffmpeg.org/download.html) installed system-wide
- 8 GB RAM minimum (16 GB recommended)
- GPU optional but strongly recommended for TTS (XTTS-v2 is slow on CPU)

## Installation

```bash
# 1. Create environment (conda recommended)
conda create -n dubbing python=3.12
conda activate dubbing

# 2. Install FFmpeg system-wide (not via conda — causes DLL conflicts on Windows)
# macOS:
brew install ffmpeg
# Windows:
winget install ffmpeg
# Ubuntu:
sudo apt install ffmpeg

# 3. Install Python dependencies
pip install -r requirements.txt
```

> **Windows users:** after installing, run `pip uninstall torchvision -y` to avoid a DLL conflict that breaks the pipeline.

## Usage

### Single video

```bash
python -m dubbing_tool.cli dub -i video.mp4 -s vi -t en -o output/dubbed.mp4
```

### Batch processing

```bash
python -m dubbing_tool.cli batch -i videos/ -o output/ -s vi -t en
```

Processes all video files in `videos/` and writes a `batch_log_YYYYMMDD_HHMMSS.csv` to `output/` with per-video duration, processing time, and status.

### Run individual stages

```bash
# Stage 2: transcribe
python -m dubbing_tool.cli transcribe -i video.mp4 -s vi -o transcript.json

# Stage 3: translate
python -m dubbing_tool.cli translate -i transcript.json -s vi -t en -o translated.json

# Stage 4: synthesize TTS with voice cloning
python -m dubbing_tool.cli synthesize -i translated.json -r reference.wav -o segments/ -t en
```

### Options

| Flag | Description |
|---|---|
| `-s`, `--source-lang` | Source language code (e.g. `vi`, `en`) |
| `-t`, `--target-lang` | Target language code |
| `-r`, `--reference-audio` | Custom speaker reference WAV (optional, auto-extracted if omitted) |
| `--whisper-model` | Whisper model size: `large-v3`, `turbo`, `medium`, `base` |
| `--device` | `cuda`, `cpu`, or `auto` (default) |

## Supported Languages

| Code | Language | Source | Target |
|---|---|---|---|
| `en` | English | ✓ | ✓ |
| `vi` | Vietnamese | ✓ | ✓ |
| `zh` | Chinese | ✓ | ✓ |
| `ja` | Japanese | ✓ | ✓ |
| `ko` | Korean | ✓ | ✓ |
| `fr` | French | ✓ | ✓ |
| `de` | German | ✓ | ✓ |
| `es` | Spanish | ✓ | ✓ |
| `ru` | Russian | ✓ | ✓ |
| `ar` | Arabic | ✓ | ✓ |
| `th` | Thai | ✓ | ✓ |
| `id` | Indonesian | ✓ | ✓ |

## Configuration

Edit `config.yaml` to set defaults:

```yaml
whisper_model: large-v3           # large-v3 | turbo | medium | base
translation_model: nllb-200-distilled-600M   # or nllb-200-distilled-1.3B for better quality
device: auto                      # cuda | cpu | auto
output_dir: ./output
models_cache_dir: ./models_cache
```

Models are downloaded on first use and cached in `./models_cache/` (~7 GB).

## Models Used

| Stage | Model | Size |
|---|---|---|
| Transcription | [openai/whisper](https://github.com/openai/whisper) large-v3 | ~3 GB |
| Translation | [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) | ~2.4 GB |
| TTS + Voice cloning | [Coqui XTTS-v2](https://github.com/idiap/coqui-ai-TTS) | ~1.8 GB |

## Known Limitations

- TTS audio timing may not perfectly match original lip movements (no lip-sync)
- Voice cloning quality depends on reference audio clarity (use at least 6 seconds of clean speech)
- CPU-only inference is very slow for XTTS (~10x realtime on CPU vs ~1x on GPU)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License — see [LICENSE](LICENSE).

## Author

[Hùng Nguyễn](https://github.com/hungnm)
