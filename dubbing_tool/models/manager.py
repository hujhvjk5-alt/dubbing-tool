"""Model cache manager — resolves paths and detects compute device."""

from __future__ import annotations

from pathlib import Path


class ModelManager:
    def __init__(self, cache_dir: str | Path = "./models_cache"):
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_device(device: str = "auto") -> str:
        """Return 'cuda' or 'cpu'. 'auto' picks cuda if available."""
        if device != "auto":
            return device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def whisper_dir(self, model_name: str) -> Path:
        return self.cache_dir / "whisperx" / model_name

    def nllb_dir(self, model_name: str) -> Path:
        return self.cache_dir / "nllb" / model_name

    def xtts_dir(self) -> Path:
        return self.cache_dir / "xtts_v2"

    # ------------------------------------------------------------------
    # Model loaders
    # ------------------------------------------------------------------

    def load_whisper(self, model_name: str = "large-v3", device: str = "auto"):
        """Load WhisperX model, downloading to cache_dir on first use."""
        import whisperx

        device = self.resolve_device(device)
        compute_type = "float16" if device == "cuda" else "int8"
        model_dir = str(self.whisper_dir(model_name))

        return whisperx.load_model(
            model_name,
            device=device,
            compute_type=compute_type,
            download_root=model_dir,
        )

    def load_align_model(self, language_code: str, device: str = "auto"):
        """Load WhisperX alignment model for forced word-level alignment."""
        import whisperx

        device = self.resolve_device(device)
        model, metadata = whisperx.load_align_model(
            language_code=language_code,
            device=device,
        )
        return model, metadata

    def load_nllb(self, model_name: str = "nllb-200-distilled-600M"):
        """Load NLLB tokenizer and CTranslate2 model, downloading on first use."""
        from transformers import AutoTokenizer
        import ctranslate2

        model_id = f"facebook/{model_name}"
        nllb_dir = self.nllb_dir(model_name)

        # Download tokenizer via transformers (cached in nllb_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=str(nllb_dir)
        )

        # Download and convert model to CTranslate2 format if needed
        ct2_path = nllb_dir / "ct2"
        if not (ct2_path / "model.bin").exists():
            _convert_nllb_to_ct2(model_id, ct2_path, str(nllb_dir))

        translator = ctranslate2.Translator(str(ct2_path), device="cpu")
        return tokenizer, translator

    def load_xtts(self, device: str = "auto"):
        """Load Coqui XTTS-v2 model."""
        from TTS.api import TTS  # package: coqui-tts

        device = self.resolve_device(device)
        xtts_dir = self.xtts_dir()

        tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=True,
        )
        tts.to(device)
        return tts


def _convert_nllb_to_ct2(model_id: str, output_dir: Path, cache_dir: str) -> None:
    """Convert HuggingFace NLLB model to CTranslate2 format (runs once).

    Uses ct2-transformers-converter CLI installed with ctranslate2.
    """
    import subprocess
    import sys

    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ct2-transformers-converter",
            "--model", model_id,
            "--output_dir", str(output_dir),
            "--quantization", "int8",
            "--force",
        ],
        check=True,
    )
