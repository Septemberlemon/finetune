"""
Human-readable text per *model token* for Chinese using a byte-level BPE tokenizer.

Dependencies:
  pip install --upgrade transformers>=4.44 tokenizers>=0.15

Notes and docs:
  - Fast tokenizers + offset mapping:
    https://huggingface.co/docs/transformers/main/en/fast_tokenizers#returning-offsets-and-encodings
  - Decoding and relation to convert_* helpers:
    https://huggingface.co/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizer.decode
  - Why mojibake happens (byte-level BPE → bytes↔unicode mapping like GPT-2):
    https://huggingface.co/docs/transformers/main/en/tokenizer_summary#bytelevelbpe

Model tokenizer:
  Default: "unsloth/Qwen3-14B" (tokenizer only, no model load). Change MODEL_ID if needed.
  Other compatible examples: "Qwen/Qwen2-7B", "Qwen/Qwen2.5-7B"
"""

from __future__ import annotations
import os
from typing import List, Tuple
from transformers import AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "unsloth/Qwen3-14B")
TEXT = "今天天气真好"  # target string

def fmt_list(xs: List[str]) -> str:
    # Show Python-list formatting with repr to make mojibake obvious
    return "[" + ", ".join(repr(x) for x in xs) + "]"

def main() -> None:
    # Load fast tokenizer to enable offset mapping
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if not getattr(tok, "is_fast", False):
        raise RuntimeError(
            "This demo requires a *fast* tokenizer (Rust-backed). "
            "Install `tokenizers` and pass use_fast=True, or pick a model with fast tokenizer."
        )

    print(f"Tokenizer: {MODEL_ID}")
    print(f"Input text: {TEXT}\n")

    # ---------- REPRO: shows why tokenize()/convert_ids_to_tokens look garbled ----------
    enc_plain = tok(TEXT, add_special_tokens=False)
    input_ids: List[int] = enc_plain["input_ids"]

    print("REPRO: what you saw")
    print("1) tokenizer.decode(input_ids) -> original text")
    print("   ", tok.decode(input_ids))
    print("2) tokenizer.tokenize(text) -> mojibake-like byte-level symbols")
    print("   ", fmt_list(tok.tokenize(TEXT)))
    print("3) tokenizer.convert_ids_to_tokens(input_ids) -> same mojibake symbols")
    print("   ", fmt_list(tok.convert_ids_to_tokens(input_ids)))
    print()

    # ---------- FIX A: use character offset mapping to slice original text ----------
    # This yields human-readable spans that correspond to each *model token piece*.
    enc_off = tok(TEXT, add_special_tokens=False, return_offsets_mapping=True)
    offsets: List[Tuple[int, int]] = enc_off["offset_mapping"]

    # Slice original TEXT by offsets to get readable per-token text
    pieces_offsets = [TEXT[a:b] for (a, b) in offsets]

    # Sanity: reconstructed string equals original (for add_special_tokens=False)
    reconstructed = "".join(pieces_offsets)
    ok = reconstructed == TEXT

    print("FIX A: return_offsets_mapping=True, then slice TEXT[a:b]")
    print("   human-readable per-token spans:", fmt_list(pieces_offsets))
    print("   reconstructed == original:", ok)
    print()

    # ---------- FIX B: decode per-id to readable text ----------
    # Decoding each token id individually also avoids mojibake.
    pieces_decode = [tok.decode([tid], skip_special_tokens=True) for tid in input_ids]
    print("FIX B: per-token decode([id])")
    print("   human-readable per-token decodes:", fmt_list(pieces_decode))
    print()

    # ---------- Optional: show a simple alignment table ----------
    print("Alignment table (id, token_str, offset_slice, per_id_decode):")
    id_tokens = tok.convert_ids_to_tokens(input_ids)
    for i, (tid, raw_tok, (a, b)) in enumerate(zip(input_ids, id_tokens, offsets)):
        slice_text = TEXT[a:b]
        per_id_text = pieces_decode[i]
        print(
            f"{i:2d}: id={tid:<8d} raw={raw_tok!r:>14}  "
            f"offset=({a:>2},{b:<2}) slice={slice_text!r:<6} "
            f"decode_one={per_id_text!r}"
        )

if __name__ == "__main__":
    main()

"""
...
REPRO: what you saw
1) tokenizer.decode(input_ids) -> original text
    今天天气真好
2) tokenizer.tokenize(text) -> mojibake-like byte-level symbols
    ['ä»Ĭå¤©', 'å¤©æ°Ķ', 'çľŁ', 'å¥½']
3) tokenizer.convert_ids_to_tokens(input_ids) -> same mojibake symbols
    ['ä»Ĭå¤©', 'å¤©æ°Ķ', 'çľŁ', 'å¥½']

FIX A: return_offsets_mapping=True, then slice TEXT[a:b]
   human-readable per-token spans: ['今天', '天气', '真', '好']
   reconstructed == original: True

FIX B: per-token decode([id])
   human-readable per-token decodes: ['今天', '天气', '真', '好']
...
"""
