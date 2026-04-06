# Contributing

## Getting started

```bash
git clone https://github.com/hungnm/dubbing-tool.git
cd dubbing-tool
conda create -n dubbing python=3.12
conda activate dubbing
pip install -r requirements.txt
pip uninstall torchvision -y  # Windows only
```

## Project structure

```
dubbing_tool/
  cli.py            – CLI commands (dub, batch, transcribe, translate, synthesize)
  pipeline.py       – Full pipeline orchestrator
  stages/           – One file per pipeline stage
  models/manager.py – Model download and caching
  utils/            – Shared helpers (audio, segments)
```

## Adding a new language

1. Add the BCP-47 code → NLLB code mapping in `stages/translator.py` (`LANG_TO_NLLB`)
2. Add the BCP-47 code → XTTS code mapping in `stages/synthesizer.py` (`LANG_TO_XTTS`)
3. Update the language table in `README.md`

## Adding a new pipeline stage

Each stage follows the same pattern:
- Takes `input_path` (JSON or audio file) and returns a list of `Segment` objects
- Writes output via `save_segments()` or `soundfile.write()`
- Is independently runnable for debugging

## Reporting bugs

Use the GitHub issue tracker. Include:
- OS and Python version
- Full traceback
- The command you ran
- Whether you have a GPU

## Pull requests

- Keep PRs focused on a single change
- Test with at least one real video before submitting
- Update `CLAUDE.md` if you change the architecture or dependencies
