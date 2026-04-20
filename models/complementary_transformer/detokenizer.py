#!/usr/bin/env python3
"""Detokenize complementary enriched tokens into MIDI.

Supported token pattern around each note:
... time_* vel_* dur_* rel_* ...
or
... time_* vel_* dur_* note_* ...
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from mido import Message, MetaMessage, MidiFile, MidiTrack, second2tick


TICKS_PER_BEAT = 480
TEMPO = 500000  # 120 BPM
DEFAULT_VELOCITY = 63
DEFAULT_DURATION = 0.5

TOKEN_PATTERN = re.compile(
    r"<BOM>|<EOM>|key_[A-G](?:b|#)?|time_\d+(?:\.\d+)?|"
    r"note_-?\d+|rel_-?\d+|dur_\d+(?:\.\d+)?|vel_\d+"
)

KEY_TO_SEMITONE = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}


@dataclass
class NoteEvent:
    time_sec: float
    note: int
    velocity: int
    duration_sec: float


@dataclass
class MidiEvent:
    time_sec: float
    note: int
    on: bool
    velocity: int


def extract_tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def parse_note_events(tokens: list[str]) -> list[NoteEvent]:
    events: list[NoteEvent] = []
    current_time = 0.0
    pending_delta = 0.0
    current_key_offset = 0
    pending_velocity = DEFAULT_VELOCITY
    pending_duration = DEFAULT_DURATION

    for tok in tokens:
        if tok.startswith("time_"):
            pending_delta += float(tok.split("_", 1)[1])
            continue
        if tok.startswith("key_"):
            key_name = tok.split("_", 1)[1]
            if key_name in KEY_TO_SEMITONE:
                current_key_offset = KEY_TO_SEMITONE[key_name]
            continue
        if tok.startswith("vel_"):
            # vel_0..vel_31 -> MIDI 0..127
            idx = int(tok.split("_", 1)[1])
            idx = max(0, min(31, idx))
            pending_velocity = int(round((idx / 31.0) * 127))
            continue
        if tok.startswith("dur_"):
            pending_duration = float(tok.split("_", 1)[1])
            pending_duration = max(0.05, min(10.0, pending_duration))
            continue

        note_val: int | None = None
        if tok.startswith("note_"):
            note_val = int(tok.split("_", 1)[1])
        elif tok.startswith("rel_"):
            rel = int(tok.split("_", 1)[1])
            note_val = rel + current_key_offset

        if note_val is None:
            continue

        current_time += pending_delta
        pending_delta = 0.0
        if 0 <= note_val <= 127:
            events.append(
                NoteEvent(
                    time_sec=current_time,
                    note=note_val,
                    velocity=max(1, min(127, pending_velocity)),
                    duration_sec=pending_duration,
                )
            )

    return events


def events_to_midi(note_events: list[NoteEvent], out_path: Path) -> None:
    midi_events: list[MidiEvent] = []
    for e in note_events:
        midi_events.append(MidiEvent(time_sec=e.time_sec, note=e.note, on=True, velocity=e.velocity))
        midi_events.append(MidiEvent(time_sec=e.time_sec + e.duration_sec, note=e.note, on=False, velocity=0))

    midi_events.sort(key=lambda x: (x.time_sec, 0 if not x.on else 1, x.note))

    mid = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(MetaMessage("set_tempo", tempo=TEMPO, time=0))
    track.append(Message("program_change", program=0, time=0))

    last_time = 0.0
    for e in midi_events:
        delta_sec = max(0.0, e.time_sec - last_time)
        delta_ticks = int(round(second2tick(delta_sec, TICKS_PER_BEAT, TEMPO)))
        if e.on:
            track.append(Message("note_on", note=e.note, velocity=e.velocity, time=delta_ticks))
        else:
            track.append(Message("note_off", note=e.note, velocity=0, time=delta_ticks))
        last_time = e.time_sec

    track.append(MetaMessage("end_of_track", time=0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(out_path))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detokenize complementary enriched token text to MIDI.")
    p.add_argument("--text", default=None, help="Inline text containing tokens.")
    p.add_argument("--input-file", default=None, help="Path to text file with tokens.")
    p.add_argument("--output-midi", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.text and not args.input_file:
        raise ValueError("Provide either --text or --input-file.")
    if args.text and args.input_file:
        raise ValueError("Provide only one of --text or --input-file.")

    text = args.text or Path(args.input_file).read_text(encoding="utf-8")
    tokens = extract_tokens(text)
    if not tokens:
        raise RuntimeError("No valid tokens found.")

    note_events = parse_note_events(tokens)
    if not note_events:
        raise RuntimeError("No note events parsed.")

    out_path = Path(args.output_midi)
    events_to_midi(note_events, out_path)
    print(f"Parsed tokens: {len(tokens)}")
    print(f"Parsed notes: {len(note_events)}")
    print(f"Wrote MIDI: {out_path}")


if __name__ == "__main__":
    main()
