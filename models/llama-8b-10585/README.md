---
library_name: peft
model_name: t2p-base
license: llama3.1
base_model: meta-llama/Llama-3.1-8B
pipeline_tag: text-generation
tags:
- lora
- peft
- text-to-music
- symbolic-music
- bachground
---

# Model Card for t2p-base

This repository folder contains a BachGround LoRA adapter trained on top of `meta-llama/Llama-3.1-8B` for symbolic piano token generation.

This is not a standalone base model checkpoint. It is intended to be used as an adapter together with the original Llama 3.1 8B base model.

## Hugging Face Links

- Adapter repo: `https://huggingface.co/umutgur/t2p`
- Project repo: `https://github.com/BachGround/t2p`

## Intended Use

The model generates base symbolic piano token sequences from text prompts. In the full BachGround pipeline, these base tokens are then enriched by a complementary transformer that predicts duration and velocity tokens before MIDI rendering.

## Files in This Folder

- `adapter_model.safetensors`
- `adapter_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `training_args.bin`

## Usage Notes

- Requires the original `meta-llama/Llama-3.1-8B` base model.
- Usage is subject to the Meta Llama 3.1 license terms.
- For end-to-end generation, pair this adapter with the complementary transformer weights described in the main project README.

## License

This adapter is derived from Llama 3.1 and should be distributed under the applicable Llama 3.1 license terms. If published on Hugging Face, use the `llama3.1` license tag and link the original base model in the model card.
