[![BachGround](banner.jpeg)](https://www.bachground.com)

# Text to Piano

Text to Piano is a clean, public-facing version of our text-to-piano model.

Its purpose is to generate piano music from text prompts through a two-stage pipeline:

1. A base text-to-piano model generates structural music tokens
2. A complementary transformer predicts duration and velocity to make the result more expressive

This project is part of the [BachGround](https://www.bachground.com) ecosystem. BachGround is the product and research context behind this work, and this repository is intended to expose the core model assets and inference flow in a simpler form.

## About BachGround

[BachGround](https://www.bachground.com) is the product context behind Text to Piano.

When referring to this repository, please consider it as an open source release of model and pipeline components developed for BachGround.

Website reference:
- [`https://www.BachGround.com`](https://www.bachground.com)

## What This Repository Contains

This repository includes the core pieces needed to understand and run the generation pipeline:

- a fine-tuned Llama-based text-to-token model
- a complementary transformer for duration and velocity prediction
- inference scripts for both stages
- lightweight documentation for running the models

This repository does not aim to include the full internal development history or all experimental datasets.

## Model Downloads

Hugging Face model links will be published here:

- Llama adapter model: [`T2P Base Model`](https://huggingface.co/umutgur/t2p/tree/main/t2p_base)
- Complementary transformer model: [`Complementary Transformer`](https://huggingface.co/umutgur/t2p/tree/main/complementary_transformer)

Recommended split:

- GitHub hosts code, lightweight configs, and documentation
- Hugging Face hosts large model weights and downloadable inference assets

## What Problem Text to Piano Solves

Generating piano music directly from text is easier to manage when musical structure and expressive playback are separated.

In Text to Piano:
- the base model generates the musical skeleton
- the complementary transformer adds performance details

This makes the system easier to inspect, debug, and improve.

## High-Level Pipeline

The full pipeline is:

```text
text prompt -> base token generation -> duration/velocity enrichment -> MIDI
```

More concretely:

1. A text prompt is given to the base model
2. The base model generates piano token sequences
3. The complementary transformer enriches the sequence with `dur_*` and `vel_*` tokens
4. The enriched output is converted into MIDI

## Repository Layout

Current top-level structure:

- `models/llama-8b-piast-at-10585`
  Base text-to-piano model assets
- `models/complementary_transformer`
  Complementary transformer code and trained duration/velocity models
- `scripts/infer_llama_music.py`
  Optional standalone inference entrypoint for the base model
- `scripts/infer_music_full_pipeline.py`
  End-to-end prompt -> base tokens -> enrichment -> MIDI pipeline
- `scripts/midi_to_mp3.py`
  Optional MIDI to MP3 rendering utility
- `README.md`
  Project overview and usage notes

Inside `models/complementary_transformer`:
- `train.py`
- `detokenizer.py`
- `token2midi.py`
- `models/duration`
- `models/velocity`

## Inference Flow

### End-to-end pipeline

The recommended inference entrypoint is the combined pipeline script:

```bash
python3 scripts/infer_music_full_pipeline.py \
  --llama-model-dir models/llama-8b-10585 \
  --duration-model-dir models/complementary_transformer/models/duration \
  --velocity-model-dir models/complementary_transformer/models/velocity \
  --prompt "A calm piano melody in C major" \
  --output-dir pipeline_out \
  --do-sample \
  --temperature 0.9 \
  --top-p 0.9 \
  --max-new-tokens 450
```

This single command performs the full chain:

1. generate base symbolic piano tokens from the text prompt
2. detokenize the base sequence to a base MIDI file
3. enrich the sequence with `dur_*` and `vel_*` tokens using the complementary transformer
4. detokenize the enriched sequence to the final MIDI output

Typical outputs written into `pipeline_out/` are:

- timestamped base token text
- timestamped base MIDI
- timestamped enriched token text
- timestamped final MIDI

### Standalone base-model inference

`scripts/infer_llama_music.py` is still useful if you want to inspect only the raw base-token model without the complementary stage.

Typical command:

```bash
python3 scripts/infer_llama_music.py \
  --model-dir models/llama-8b-10585 \
  --prompt "A calm piano melody in C major" \
  --max-new-tokens 450
```

This standalone mode is mainly useful for debugging, token inspection, or comparing pre- and post-enrichment outputs.

## Why There Are Two Models

A single model could try to generate note identity, timing, duration, and velocity all at once. We chose not to do that here.

The two-stage design has practical advantages:

- cleaner separation between composition and performance
- easier dataset preparation for the complementary task
- easier debugging when results sound wrong
- more flexibility when improving one stage without retraining everything

## Intended Audience

This repository is meant for:
- researchers exploring symbolic music generation
- developers who want to inspect the inference pipeline
- collaborators who need a simpler view of the BachGround text-to-piano stack

## Current Status

This repository is being cleaned up for open source use.

The models are present, and the inference path is the main priority. Documentation may continue to improve as the repository becomes more polished.

## Notes

- Some scripts assume a Python environment with `torch`, `transformers`, `peft`, and MIDI-related dependencies installed.
- The complementary transformer is included mainly for inference use in this repository.
- Large training datasets are intentionally not included here.

## License

Licensing is split by artifact type:

- Repository code: `LICENSE_PLACEHOLDER_CODE`
- Llama adapter weights: distributed separately on Hugging Face under the applicable Llama 3.1 license terms
- Complementary transformer weights: `LICENSE_PLACEHOLDER_COMPLEMENTARY`

If you publish the model artifacts, replace the placeholders above and add the final Hugging Face links in the `Model Downloads` section.

## Attribution

If you reference or use this repository, please mention that it is part of the [BachGround](https://www.bachground.com) project.
