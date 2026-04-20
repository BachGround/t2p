#!/usr/bin/env python3
"""Complementary transformer pipeline for duration/velocity gap filling.

This module provides three subcommands:
- prepare: build supervised samples from MIDI files.
- train: train a future-context transformer classifier (duration or velocity).
- infer: enrich a base token stream with predicted duration then velocity tokens.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import types
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


TIME_QUANT = 0.05
TIME_MIN = 0.05
TIME_MAX = 10.0
NUM_DUR_BINS = 200
NUM_VEL_BINS = 32
LOOKAHEAD_DEFAULT = 32
NUM_HEADS_DEFAULT = 16
NUM_LAYERS_DEFAULT = 16


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _import_base_tokenizer():
    # src/tokenization/my_tokenizer.py imports Bach which imports sounddevice.
    # Stub sounddevice for non-audio training/inference environments.
    sys.modules["sounddevice"] = types.SimpleNamespace(play=lambda *a, **k: None, wait=lambda *a, **k: None)
    tokenization_dir = _project_root() / "src" / "tokenization"
    if str(tokenization_dir) not in sys.path:
        sys.path.insert(0, str(tokenization_dir))
    from my_tokenizer import midi_to_tokens, relative_tokens

    return midi_to_tokens, relative_tokens


def is_note_token(tok: str) -> bool:
    return tok.startswith("rel_") or tok.startswith("note_")


def duration_to_bin(duration_sec: float) -> int:
    if duration_sec < TIME_MIN:
        quant = TIME_MIN
    else:
        quant = round(duration_sec / TIME_QUANT) * TIME_QUANT
    quant = max(TIME_MIN, min(TIME_MAX, quant))
    bin_idx = int(round(quant / TIME_QUANT)) - 1
    return max(0, min(NUM_DUR_BINS - 1, bin_idx))


def bin_to_duration(bin_idx: int) -> float:
    return (max(0, min(NUM_DUR_BINS - 1, bin_idx)) + 1) * TIME_QUANT


def duration_to_token(duration_sec: float) -> str:
    return f"dur_{bin_to_duration(duration_to_bin(duration_sec)):.2f}"


def velocity_to_bin(velocity: int) -> int:
    velocity = max(0, min(127, int(velocity)))
    return int(round((velocity / 127.0) * (NUM_VEL_BINS - 1)))


def bin_to_velocity_token(bin_idx: int) -> str:
    return f"vel_{max(0, min(NUM_VEL_BINS - 1, bin_idx))}"


def extract_note_on_events(midi_path: Path) -> list[tuple[float, int]]:
    """Return note-on events as (velocity, duration_sec), ordered by note-on time.

    Uses FIFO pairing per pitch for overlapping same-note voices.
    """
    import mido

    mid = mido.MidiFile(str(midi_path))
    current_time = 0.0
    active: dict[int, deque[tuple[float, int]]] = defaultdict(deque)
    ordered_on: list[dict[str, float | int | None]] = []

    for msg in mid:
        current_time += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            active[msg.note].append((current_time, int(msg.velocity)))
            ordered_on.append({"note": msg.note, "start": current_time, "vel": int(msg.velocity), "dur": None})
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            q = active[msg.note]
            if not q:
                continue
            start, vel = q.popleft()
            duration = max(0.0, current_time - start)
            # Fill first unmatched note_on record with same pitch+start.
            for rec in ordered_on:
                if rec["dur"] is None and rec["note"] == msg.note and abs(float(rec["start"]) - start) < 1e-6:
                    rec["dur"] = duration
                    break

    out: list[tuple[float, int]] = []
    for rec in ordered_on:
        dur = float(rec["dur"]) if rec["dur"] is not None else TIME_MIN
        out.append((dur, int(rec["vel"])))
    return out


@dataclass
class MidiTokenBundle:
    base_tokens: list[str]
    durations: list[int]
    velocities: list[int]


def build_midi_bundle(midi_path: Path) -> MidiTokenBundle:
    midi_to_tokens, relative_tokens = _import_base_tokenizer()
    base_tokens = relative_tokens(midi_to_tokens(str(midi_path)))
    note_meta = extract_note_on_events(midi_path)

    note_positions = [i for i, tok in enumerate(base_tokens) if is_note_token(tok)]
    if len(note_positions) != len(note_meta):
        raise ValueError(
            f"Note alignment mismatch in {midi_path.name}: "
            f"{len(note_positions)} note tokens vs {len(note_meta)} note_on events"
        )

    durations = [duration_to_bin(d) for d, _ in note_meta]
    velocities = [velocity_to_bin(v) for _, v in note_meta]
    return MidiTokenBundle(base_tokens=base_tokens, durations=durations, velocities=velocities)


def inject_duration_tokens(base_tokens: list[str], duration_bins: list[int]) -> list[str]:
    out: list[str] = []
    note_idx = 0
    for tok in base_tokens:
        if is_note_token(tok):
            out.append(f"dur_{bin_to_duration(duration_bins[note_idx]):.2f}")
            out.append(tok)
            note_idx += 1
        else:
            out.append(tok)
    return out


def inject_velocity_and_duration(
    base_tokens: list[str],
    duration_bins: list[int],
    velocity_bins: list[int],
) -> list[str]:
    out: list[str] = []
    note_idx = 0
    for tok in base_tokens:
        if is_note_token(tok):
            out.append(bin_to_velocity_token(velocity_bins[note_idx]))
            out.append(f"dur_{bin_to_duration(duration_bins[note_idx]):.2f}")
            out.append(tok)
            note_idx += 1
        else:
            out.append(tok)
    return out


def iter_midis(folder: Path, recursive: bool = True) -> Iterable[Path]:
    pattern = "**/*.mid" if recursive else "*.mid"
    yield from folder.glob(pattern)
    pattern_midi = "**/*.midi" if recursive else "*.midi"
    yield from folder.glob(pattern_midi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Complementary transformer: prepare/train/infer.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prepare = sub.add_parser("prepare", help="Build JSONL samples from a MIDI folder.")
    p_prepare.add_argument("--midi-dir", required=True)
    p_prepare.add_argument("--output-dir", default="complementary_transformer/data")
    p_prepare.add_argument("--lookahead", type=int, default=LOOKAHEAD_DEFAULT)
    p_prepare.add_argument("--seed", type=int, default=42)
    p_prepare.add_argument("--train-ratio", type=float, default=0.95)
    p_prepare.add_argument("--recursive", action="store_true")
    p_prepare.add_argument("--max-files", type=int, default=None, help="Optional cap on number of MIDI files.")

    p_train = sub.add_parser("train", help="Train duration or velocity model.")
    p_train.add_argument("--target", choices=["duration", "velocity"], required=True)
    p_train.add_argument("--train-jsonl", required=True)
    p_train.add_argument("--valid-jsonl", default=None)
    p_train.add_argument("--output-dir", required=True)
    p_train.add_argument("--epochs", type=int, default=5)
    p_train.add_argument("--batch-size", type=int, default=128)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--embed-dim", type=int, default=192)
    p_train.add_argument("--num-heads", type=int, default=NUM_HEADS_DEFAULT)
    p_train.add_argument("--num-layers", type=int, default=NUM_LAYERS_DEFAULT)
    p_train.add_argument("--dropout", type=float, default=0.1)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument(
        "--train-chunk-size",
        type=int,
        default=None,
        help="Optional number of training rows to load per chunk.",
    )
    p_train.add_argument(
        "--resume-from",
        default=None,
        help="Optional checkpoint path to resume chunked training from.",
    )
    p_train.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Optional directory for rolling latest.pt checkpoints during training.",
    )

    p_infer = sub.add_parser("infer", help="Predict durations then velocities, and emit enriched tokens.")
    p_infer.add_argument("--duration-model-dir", required=True)
    p_infer.add_argument("--velocity-model-dir", required=True)
    p_infer.add_argument("--input-midi", default=None, help="Optional MIDI path to tokenize.")
    p_infer.add_argument("--input-tokens", default=None, help="Optional text file with base tokens.")
    p_infer.add_argument("--output-tokens", required=True)
    p_infer.add_argument("--dur-temperature", type=float, default=1.0)
    p_infer.add_argument("--vel-temperature", type=float, default=1.0)
    p_infer.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def prepare_samples_for_bundle(
    bundle: MidiTokenBundle,
    lookahead: int,
) -> tuple[list[dict], list[dict]]:
    base = bundle.base_tokens
    dur_stream = inject_duration_tokens(base, bundle.durations)
    duration_rows: list[dict] = []
    velocity_rows: list[dict] = []

    note_idx = 0
    for pos, tok in enumerate(base):
        if not is_note_token(tok):
            continue
        context = base[pos + 1 : pos + 1 + lookahead]
        duration_rows.append({"context": context, "label": bundle.durations[note_idx]})
        note_idx += 1

    note_idx = 0
    for pos, tok in enumerate(dur_stream):
        if not is_note_token(tok):
            continue
        context = dur_stream[pos + 1 : pos + 1 + lookahead]
        velocity_rows.append({"context": context, "label": bundle.velocities[note_idx]})
        note_idx += 1

    return duration_rows, velocity_rows


def build_vocab(rows: list[dict]) -> dict[str, int]:
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for row in rows:
        for tok in row["context"]:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def encode_rows(rows: list[dict], vocab: dict[str, int], lookahead: int) -> list[dict]:
    out: list[dict] = []
    unk = vocab["<UNK>"]
    for row in rows:
        ids = [vocab.get(tok, unk) for tok in row["context"][:lookahead]]
        if len(ids) < lookahead:
            ids.extend([vocab["<PAD>"]] * (lookahead - len(ids)))
        out.append({"input_ids": ids, "label": int(row["label"])})
    return out


class ContextDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.x = torch.tensor([r["input_ids"] for r in rows], dtype=torch.long)
        self.y = torch.tensor([r["label"] for r in rows], dtype=torch.long)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class FutureContextTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        num_classes: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        pos = torch.arange(seqlen, device=input_ids.device).unsqueeze(0).expand(bsz, seqlen)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        key_padding_mask = input_ids.eq(self.pad_id)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        valid = (~key_padding_mask).unsqueeze(-1).float()
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.head(pooled)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_prepare(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    midi_dir = Path(args.midi_dir)
    out_dir = Path(args.output_dir)
    if not midi_dir.exists():
        raise FileNotFoundError(f"MIDI dir not found: {midi_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    duration_rows: list[dict] = []
    velocity_rows: list[dict] = []
    files = sorted(set(iter_midis(midi_dir, recursive=args.recursive)))
    if args.max_files is not None:
        files = files[: max(0, args.max_files)]
    if not files:
        raise RuntimeError(f"No MIDI files found under: {midi_dir}")

    skipped = 0
    for midi_path in files:
        try:
            bundle = build_midi_bundle(midi_path)
            d_rows, v_rows = prepare_samples_for_bundle(bundle, args.lookahead)
            duration_rows.extend(d_rows)
            velocity_rows.extend(v_rows)
        except Exception as exc:
            skipped += 1
            print(f"Skipping {midi_path.name}: {exc}")

    rng = random.Random(args.seed)
    rng.shuffle(duration_rows)
    rng.shuffle(velocity_rows)

    def split_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
        n_train = int(len(rows) * args.train_ratio)
        if rows and args.train_ratio > 0.0 and n_train == 0:
            n_train = 1
        if len(rows) > 1 and args.train_ratio < 1.0 and n_train == len(rows):
            n_train = len(rows) - 1
        return rows[:n_train], rows[n_train:]

    d_train, d_valid = split_rows(duration_rows)
    v_train, v_valid = split_rows(velocity_rows)

    def write_jsonl(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_jsonl(out_dir / "duration_train.jsonl", d_train)
    write_jsonl(out_dir / "duration_valid.jsonl", d_valid)
    write_jsonl(out_dir / "velocity_train.jsonl", v_train)
    write_jsonl(out_dir / "velocity_valid.jsonl", v_valid)

    print(f"MIDI files scanned: {len(files)} (skipped: {skipped})")
    print(f"Duration rows: train={len(d_train)} valid={len(d_valid)}")
    print(f"Velocity rows: train={len(v_train)} valid={len(v_valid)}")
    print(f"Wrote dataset to: {out_dir}")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_jsonl_chunk(path: Path, start: int, max_rows: int | None) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < start:
                continue
            if max_rows is not None and len(rows) >= max_rows:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def infer_lookahead(path: Path) -> int:
    rows = read_jsonl_chunk(path, 0, 1)
    if not rows:
        raise RuntimeError("Training JSONL is empty.")
    lookahead = len(rows[0]["context"])
    if lookahead <= 0:
        raise RuntimeError("Invalid context length in dataset.")
    return lookahead


def build_vocab_from_jsonl(paths: list[Path]) -> dict[str, int]:
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                for tok in row["context"]:
                    if tok not in vocab:
                        vocab[tok] = len(vocab)
    return vocab


def evaluate_jsonl(
    model: nn.Module,
    path: Path,
    vocab: dict[str, int],
    lookahead: int,
    batch_size: int,
    device: torch.device,
    chunk_size: int | None,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    ce = nn.CrossEntropyLoss()
    offset = 0

    while True:
        rows = read_jsonl_chunk(path, offset, chunk_size)
        if not rows:
            break
        encoded = encode_rows(rows, vocab, lookahead)
        loader = DataLoader(ContextDataset(encoded), batch_size=batch_size)
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = ce(logits, y)
                total_loss += loss.item() * y.size(0)
                preds = logits.argmax(dim=-1)
                correct += int((preds == y).sum().item())
                total += int(y.size(0))
        offset += len(rows)

    return (total_loss / max(total, 1), correct / max(total, 1))


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = ce(logits, y)
            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=-1)
            correct += int((preds == y).sum().item())
            total += int(y.size(0))
    return (total_loss / max(total, 1), correct / max(total, 1))


def run_train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    if args.embed_dim % args.num_heads != 0:
        raise ValueError(
            f"embed_dim ({args.embed_dim}) must be divisible by num_heads ({args.num_heads})."
        )

    train_path = Path(args.train_jsonl)
    valid_path = Path(args.valid_jsonl) if args.valid_jsonl else None
    train_chunk_size = args.train_chunk_size
    total_train_rows = count_jsonl_rows(train_path)
    if total_train_rows <= 0:
        raise RuntimeError("Training JSONL is empty.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = checkpoint_dir / "latest.pt"

    start_epoch = 1
    next_train_row = 0
    lookahead = infer_lookahead(train_path)
    vocab = build_vocab_from_jsonl([train_path] + ([valid_path] if valid_path else []))
    num_classes = NUM_DUR_BINS if args.target == "duration" else NUM_VEL_BINS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FutureContextTransformer(
        vocab_size=len(vocab),
        max_len=lookahead,
        num_classes=num_classes,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_id=0,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    if args.resume_from:
        resume_path = Path(args.resume_from)
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        next_train_row = int(checkpoint.get("next_train_row", 0))
        saved_cfg = checkpoint.get("config", {})
        if saved_cfg:
            lookahead = int(saved_cfg["lookahead"])
            vocab = saved_cfg["vocab"]
        if next_train_row == 0:
            start_epoch += 1
        print(f"Resuming from {resume_path} at epoch={start_epoch} next_train_row={next_train_row}")

    effective_chunk_size = train_chunk_size or total_train_rows

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        count = 0
        chunk_start = next_train_row if epoch == start_epoch else 0

        while chunk_start < total_train_rows:
            chunk_rows = read_jsonl_chunk(train_path, chunk_start, effective_chunk_size)
            if not chunk_rows:
                break
            train_encoded = encode_rows(chunk_rows, vocab, lookahead)
            train_loader = DataLoader(ContextDataset(train_encoded), batch_size=args.batch_size, shuffle=True)

            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss = ce(logits, y)
                loss.backward()
                opt.step()
                running_loss += loss.item() * y.size(0)
                count += int(y.size(0))

            chunk_start += len(chunk_rows)
            torch.save(
                {
                    "epoch": epoch,
                    "next_train_row": 0 if chunk_start >= total_train_rows else chunk_start,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "config": {
                        "target": args.target,
                        "lookahead": lookahead,
                        "vocab": vocab,
                        "num_classes": num_classes,
                        "embed_dim": args.embed_dim,
                        "num_heads": args.num_heads,
                        "num_layers": args.num_layers,
                        "dropout": args.dropout,
                        "pad_id": 0,
                    },
                },
                latest_ckpt,
            )
            print(
                f"Epoch {epoch}/{args.epochs} chunk_end={chunk_start}/{total_train_rows} "
                f"rows_trained={count}",
                flush=True,
            )

        train_loss = running_loss / max(count, 1)
        if valid_path is not None:
            val_loss, val_acc = evaluate_jsonl(
                model,
                valid_path,
                vocab,
                lookahead,
                args.batch_size,
                device,
                effective_chunk_size,
            )
            print(
                f"Epoch {epoch}/{args.epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )
        else:
            print(f"Epoch {epoch}/{args.epochs} train_loss={train_loss:.4f}")

    torch.save(model.state_dict(), out_dir / "model.pt")
    config = {
        "target": args.target,
        "lookahead": lookahead,
        "vocab": vocab,
        "num_classes": num_classes,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "pad_id": 0,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Saved model to: {out_dir}")


def load_model(model_dir: Path) -> tuple[FutureContextTransformer, dict]:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    model = FutureContextTransformer(
        vocab_size=len(config["vocab"]),
        max_len=config["lookahead"],
        num_classes=config["num_classes"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        pad_id=config["pad_id"],
    )
    state = torch.load(model_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, config


def sample_label(
    model: FutureContextTransformer,
    config: dict,
    context_tokens: list[str],
    temperature: float,
    device: torch.device,
) -> int:
    vocab = config["vocab"]
    lookahead = int(config["lookahead"])
    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in context_tokens[:lookahead]]
    if len(ids) < lookahead:
        ids.extend([config["pad_id"]] * (lookahead - len(ids)))
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x)[0]
        t = max(1e-5, float(temperature))
        probs = torch.softmax(logits / t, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())


def tokenize_midi_to_base(midi_path: Path) -> list[str]:
    midi_to_tokens, relative_tokens = _import_base_tokenizer()
    return relative_tokens(midi_to_tokens(str(midi_path)))


def read_tokens_file(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").strip().split()


def run_infer(args: argparse.Namespace) -> None:
    if not args.input_midi and not args.input_tokens:
        raise ValueError("Provide --input-midi or --input-tokens.")
    if args.input_midi and args.input_tokens:
        raise ValueError("Provide only one of --input-midi or --input-tokens.")

    set_seed(args.seed)
    base_tokens = (
        tokenize_midi_to_base(Path(args.input_midi))
        if args.input_midi
        else read_tokens_file(Path(args.input_tokens))
    )

    note_positions = [i for i, tok in enumerate(base_tokens) if is_note_token(tok)]
    if not note_positions:
        raise RuntimeError("No note tokens found in input.")

    dur_model, dur_cfg = load_model(Path(args.duration_model_dir))
    vel_model, vel_cfg = load_model(Path(args.velocity_model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dur_model.to(device)
    vel_model.to(device)

    dur_bins: list[int] = []
    for pos in note_positions:
        context = base_tokens[pos + 1 : pos + 1 + int(dur_cfg["lookahead"])]
        dur_bins.append(sample_label(dur_model, dur_cfg, context, args.dur_temperature, device))

    dur_stream = inject_duration_tokens(base_tokens, dur_bins)

    vel_bins: list[int] = []
    for pos, tok in enumerate(dur_stream):
        if not is_note_token(tok):
            continue
        context = dur_stream[pos + 1 : pos + 1 + int(vel_cfg["lookahead"])]
        vel_bins.append(sample_label(vel_model, vel_cfg, context, args.vel_temperature, device))

    enriched = inject_velocity_and_duration(base_tokens, dur_bins, vel_bins)
    out_path = Path(args.output_tokens)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(" ".join(enriched) + "\n", encoding="utf-8")
    print(f"Base note tokens: {len(note_positions)}")
    print(f"Wrote enriched tokens: {out_path}")


def main() -> None:
    args = parse_args()
    if args.cmd == "prepare":
        run_prepare(args)
    elif args.cmd == "train":
        run_train(args)
    elif args.cmd == "infer":
        run_infer(args)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
