---
library_name: pytorch
model_name: complementary-transformer
license: apache2.0
tags:
- symbolic-music
- midi
- duration-prediction
- velocity-prediction
- bachground
---

# Model Card for complementary-transformer

This folder contains the BachGround complementary transformer used after base token generation.

It predicts:

- duration bins
- velocity bins

from symbolic piano token context, and is used to enrich the output of the base text-to-token model before MIDI rendering.

## Hugging Face Links

- Complementary transformer repo: `https://huggingface.co/umutgur/t2p`
- Project repo: `https://github.com/BachGround/t2p`

## Folder Contents

- `train.py`
- `detokenizer.py`
- `token2midi.py`
- `models/duration/config.json`
- `models/duration/model.pt`
- `models/velocity/config.json`
- `models/velocity/model.pt`

## Intended Use

This model is intended for inference in the BachGround text-to-piano pipeline. It is trained on symbolic MIDI-derived contexts and is not a text model.

## License

Set the final license based on the underlying training-data rights.

If the dataset rights are not fully open and redistribution terms are limited, keep this artifact under a custom or restricted license rather than MIT or Apache-2.0.

Replace this note with the final distribution terms before public release.
