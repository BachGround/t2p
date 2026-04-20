#!/usr/bin/env python3
"""Run Llama base-token generation and complementary enrichment in one step."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def project_root() -> Path:
    return script_dir().parent


def parse_args() -> argparse.Namespace:
    root = project_root()
    detok_default = root / "models" / "complementary_transformer" / "detokenizer.py"
    comp_train_default = root / "models" / "complementary_transformer" / "train.py"
    parser = argparse.ArgumentParser(
        description="Llama prompt -> base tokens -> complementary enrichment -> final MIDI."
    )
    parser.add_argument("--llama-model-dir", required=True)
    parser.add_argument("--duration-model-dir", required=True)
    parser.add_argument("--velocity-model-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional output prefix. Defaults to a timestamp like 2026-04-17_14-30-05.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=450)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--dur-temperature", type=float, default=1.0)
    parser.add_argument("--vel-temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument(
        "--base-detokenizer-path",
        default=str(detok_default),
        help="Detokenizer for raw Llama tokens.",
    )
    parser.add_argument(
        "--enriched-detokenizer-path",
        default=str(detok_default),
        help="Detokenizer for complementary enriched tokens.",
    )
    parser.add_argument(
        "--complementary-train-path",
        default=str(comp_train_default),
        help="Path to the complementary transformer entrypoint script.",
    )
    parser.add_argument(
        "--render-mp3",
        action="store_true",
        help="Also render the final MIDI to MP3 via Task/scripts/midi_to_mp3.py.",
    )
    parser.add_argument(
        "--soundfont",
        default=None,
        help="Optional .sf2 SoundFont path used when --render-mp3 is enabled.",
    )
    parser.add_argument(
        "--mp3-bitrate",
        default="192k",
        help="MP3 bitrate used when --render-mp3 is enabled.",
    )
    return parser.parse_args()


def generate_llama_tokens(args: argparse.Namespace) -> str:
    tokenizer = AutoTokenizer.from_pretrained(args.llama_model_dir, use_fast=False)

    quantization_config = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

    model = AutoPeftModelForCausalLM.from_pretrained(
        args.llama_model_dir,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()

    input_text = f"{args.prompt.strip()} <BOM> "
    eos_id = tokenizer.convert_tokens_to_ids("<EOM>")
    if eos_id is None or eos_id < 0:
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

    generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
    if not generated_text:
        raise RuntimeError("Llama produced empty output.")
    return generated_text


def run_subprocess(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    base_tokens_path = out_dir / f"{run_name}_llama_base_tokens.txt"
    base_midi_path = out_dir / f"{run_name}_llama_base.mid"
    enriched_tokens_path = out_dir / f"{run_name}_complementary_enriched_tokens.txt"
    enriched_midi_path = out_dir / f"{run_name}_complementary_enriched.mid"
    enriched_mp3_path = out_dir / f"{run_name}_complementary_enriched.mp3"

    generated_text = generate_llama_tokens(args)
    base_tokens_path.write_text(generated_text + "\n", encoding="utf-8")
    print(f"Wrote base tokens: {base_tokens_path}")

    run_subprocess(
        [
            sys.executable,
            args.base_detokenizer_path,
            "--text",
            generated_text,
            "--output-midi",
            str(base_midi_path),
        ]
    )

    run_subprocess(
        [
            sys.executable,
            args.complementary_train_path,
            "infer",
            "--duration-model-dir",
            args.duration_model_dir,
            "--velocity-model-dir",
            args.velocity_model_dir,
            "--input-tokens",
            str(base_tokens_path),
            "--output-tokens",
            str(enriched_tokens_path),
            "--dur-temperature",
            str(args.dur_temperature),
            "--vel-temperature",
            str(args.vel_temperature),
            "--seed",
            str(args.seed),
        ]
    )

    run_subprocess(
        [
            sys.executable,
            args.enriched_detokenizer_path,
            "--input-file",
            str(enriched_tokens_path),
            "--output-midi",
            str(enriched_midi_path),
        ]
    )

    if args.render_mp3:
        midi_to_mp3_cmd = [
            sys.executable,
            str(script_dir() / "midi_to_mp3.py"),
            "--input-midi",
            str(enriched_midi_path),
            "--output-mp3",
            str(enriched_mp3_path),
            "--bitrate",
            str(args.mp3_bitrate),
        ]
        if args.soundfont:
            midi_to_mp3_cmd.extend(["--soundfont", str(args.soundfont)])
        run_subprocess(midi_to_mp3_cmd)

    print(f"Wrote base MIDI: {base_midi_path}")
    print(f"Wrote enriched tokens: {enriched_tokens_path}")
    print(f"Wrote final MIDI: {enriched_midi_path}")
    if args.render_mp3:
        print(f"Wrote final MP3: {enriched_mp3_path}")


if __name__ == "__main__":
    main()
