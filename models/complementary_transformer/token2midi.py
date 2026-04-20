#!/usr/bin/env python3
"""Convert complementary enriched tokens to MIDI.

Thin entrypoint wrapper around complementary_transformer/detokenizer.py.
"""

from __future__ import annotations

from detokenizer import main


if __name__ == "__main__":
    main()
