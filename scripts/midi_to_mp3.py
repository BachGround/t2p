#!/usr/bin/env python3
"""Render a MIDI file to MP3 using FluidSynth + FFmpeg."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_SOUNDFONT_CANDIDATES = [
    Path("/usr/share/sounds/sf2/default-GM.sf2"),
    Path("/usr/share/sounds/sf2/FluidR3_GM.sf2"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MIDI to MP3.")
    parser.add_argument("--input-midi", required=True, help="Path to input MIDI file.")
    parser.add_argument("--output-mp3", required=True, help="Path to output MP3 file.")
    parser.add_argument(
        "--soundfont",
        default=None,
        help="Optional .sf2 SoundFont path. If omitted, common system locations are checked.",
    )
    parser.add_argument(
        "--bitrate",
        default="192k",
        help="MP3 bitrate passed to ffmpeg, for example 128k or 192k.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Rendered WAV sample rate before MP3 encoding.",
    )
    return parser.parse_args()


def resolve_soundfont(explicit_path: str | None) -> Path:
    if explicit_path:
        soundfont = Path(explicit_path).expanduser().resolve()
        if not soundfont.is_file():
            raise FileNotFoundError(f"SoundFont not found: {soundfont}")
        return soundfont

    for candidate in DEFAULT_SOUNDFONT_CANDIDATES:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "No SoundFont (.sf2) found. Install one or pass --soundfont.\n"
        "Ubuntu example: sudo apt install fluid-soundfont-gm"
    )


def ensure_command_exists(command: str, install_hint: str) -> None:
    if shutil.which(command):
        return
    raise RuntimeError(
        f"Required command '{command}' was not found.\n"
        f"Install hint: {install_hint}"
    )


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def render_midi_to_wav(input_midi: Path, output_wav: Path, soundfont: Path, sample_rate: int) -> None:
    run(
        [
            "fluidsynth",
            "-ni",
            str(soundfont),
            str(input_midi),
            "-F",
            str(output_wav),
            "-r",
            str(sample_rate),
        ]
    )


def encode_wav_to_mp3(input_wav: Path, output_mp3: Path, bitrate: str) -> None:
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_wav),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            bitrate,
            str(output_mp3),
        ]
    )


def main() -> None:
    args = parse_args()

    input_midi = Path(args.input_midi).expanduser().resolve()
    output_mp3 = Path(args.output_mp3).expanduser().resolve()

    if not input_midi.is_file():
        raise FileNotFoundError(f"Input MIDI not found: {input_midi}")

    ensure_command_exists("fluidsynth", "sudo apt install fluidsynth")
    ensure_command_exists("ffmpeg", "sudo apt install ffmpeg")
    soundfont = resolve_soundfont(args.soundfont)

    output_mp3.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="midi_to_mp3_") as tmp_dir:
        wav_path = Path(tmp_dir) / f"{input_midi.stem}.wav"
        render_midi_to_wav(input_midi, wav_path, soundfont, args.sample_rate)
        encode_wav_to_mp3(wav_path, output_mp3, args.bitrate)

    print(f"Input MIDI : {input_midi}")
    print(f"SoundFont  : {soundfont}")
    print(f"Output MP3 : {output_mp3}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
