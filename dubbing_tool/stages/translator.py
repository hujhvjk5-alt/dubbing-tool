"""Stage 3 — Offline translation using NLLB-200 + CTranslate2."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from dubbing_tool.utils.segments import Segment, load_segments, save_segments

if TYPE_CHECKING:
    from dubbing_tool.models.manager import ModelManager

# NLLB language codes: https://github.com/facebookresearch/flores/blob/main/flores200/README.md
LANG_TO_NLLB = {
    "en": "eng_Latn",
    "vi": "vie_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "ru": "rus_Cyrl",
    "ar": "arb_Arab",
    "th": "tha_Thai",
    "id": "ind_Latn",
}


def translate(
    input_path: str | Path,
    output_path: str | Path,
    manager: "ModelManager",
    source_lang: str,
    target_lang: str,
    model_name: str = "nllb-200-distilled-600M",
    batch_size: int = 8,
) -> list[Segment]:
    """
    Translate segments from source_lang to target_lang offline.

    Args:
        input_path:  transcript.json from transcriber.
        output_path: Where to write translated.json.
        manager:     ModelManager.
        source_lang: BCP-47 code, e.g. 'en'.
        target_lang: BCP-47 code, e.g. 'vi'.
        model_name:  NLLB model variant.
        batch_size:  Number of segments to translate at once.

    Returns:
        List of Segment objects with translated_text filled.
    """
    segments = load_segments(input_path)

    src_nllb = _resolve_nllb_code(source_lang)
    tgt_nllb = _resolve_nllb_code(target_lang)

    tokenizer, translator = manager.load_nllb(model_name)
    tokenizer.src_lang = src_nllb

    texts = [s.original_text for s in segments]
    translations = _translate_batch(texts, tokenizer, translator, src_nllb, tgt_nllb, batch_size)

    for seg, translated in zip(segments, translations):
        seg.translated_text = translated

    save_segments(segments, output_path)
    return segments


def _translate_batch(
    texts: list[str],
    tokenizer,
    translator,
    src_nllb: str,
    tgt_nllb: str,
    batch_size: int,
) -> list[str]:
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        # CTranslate2 expects lists of token strings, not IDs
        encoded = [
            tokenizer.convert_ids_to_tokens(
                tokenizer.encode(text, add_special_tokens=True)
            )
            for text in batch
        ]

        translations = translator.translate_batch(
            encoded,
            target_prefix=[[tgt_nllb]] * len(batch),
            max_decoding_length=512,
        )

        for t in translations:
            out_tokens = t.hypotheses[0][1:]  # skip leading language tag
            out_ids = tokenizer.convert_tokens_to_ids(out_tokens)
            decoded = tokenizer.decode(out_ids, skip_special_tokens=True)
            results.append(decoded)
    return results


def _resolve_nllb_code(lang: str) -> str:
    if lang in LANG_TO_NLLB:
        return LANG_TO_NLLB[lang]
    # Allow passing NLLB codes directly (e.g. "vie_Latn")
    if "_" in lang:
        return lang
    raise ValueError(
        f"Unknown language code '{lang}'. "
        f"Supported: {list(LANG_TO_NLLB.keys())}. "
        "Or pass the NLLB code directly (e.g. 'vie_Latn')."
    )
