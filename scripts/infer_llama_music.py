#!/usr/bin/env python3
"""Run inference on the fine-tuned Llama music model."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for fine-tuned Llama music token generator.")
    parser.add_argument("--model-dir", default="Task/output/llama-music-sft-v2")
    parser.add_argument(
        "--prompt",
        default="C major chord starting at C5",
        help='Example: "A minor scale starting at A4"',
    )
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling-based generation.")
    parser.add_argument("--temperature", type=float, default=1, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling probability.")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling cutoff.")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Penalty for repeated tokens.")
    parser.add_argument(
        "--output-midi",
        default=None,
        help="Optional MIDI output path. If set, generated text is detokenized to this MIDI file.",
    )
    parser.add_argument(
        "--detokenizer-path",
        default="detokenize/detokenizer.py",
        help="Path to detokenizer script used when --output-midi is set.",
    )
    parser.add_argument("--use-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)

    quantization_config = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model_dir,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()

    # Required generation pattern:
    # "<prompt> <BOM> " -> model generates token sequence until "<EOM>".
    # Trailing space matters because training rows are tokenized as "<BOM> key_...".
    input_text = f"{args.prompt.strip()} <BOM> "

    eos_id = tokenizer.convert_tokens_to_ids("<EOM>")
    if eos_id is None or eos_id < 0:
        # Fallback for tokenizers where <EOM> is not explicitly registered.
        eos_id = tokenizer.eos_token_id

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "eos_token_id": eos_id,
            "repetition_penalty": args.repetition_penalty,
        }
        if args.do_sample:
            gen_kwargs.update(
                {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                }
            )
        output_ids = model.generate(**inputs, **gen_kwargs)

    # Print only the generated continuation after "<prompt> <BOM>".
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    print(generated_text)

    # Quick schema sanity check: expected first structural token is usually key_*.
    first_token_match = re.search(r"(key_[A-G](?:b|#)?|time_\d+(?:\.\d+)?|note_-?\d+|rel_-?\d+|<EOM>)", generated_text)
    if first_token_match and not first_token_match.group(1).startswith("key_"):
        print(
            f"[warn] First structural token is '{first_token_match.group(1)}', not 'key_*'. "
            "This may reduce relative-note decoding quality."
        )

    if args.output_midi:
        detok_path = Path(args.detokenizer_path)
        if not detok_path.exists():
            raise FileNotFoundError(f"Detokenizer script not found: {detok_path}")

        midi_path = Path(args.output_midi)
        midi_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(detok_path),
            "--text",
            generated_text,
            "--output-midi",
            str(midi_path),
        ]
        subprocess.run(cmd, check=True)
        print(f"Wrote MIDI: {midi_path}")


if __name__ == "__main__":
    main()
