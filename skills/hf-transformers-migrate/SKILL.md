---
name: hf-transformers-migrate
description: Migrate Hugging Face transformers models to mindone.transformers. Use when porting BERT, GPT, LLaMA, or other transformer models.
---

# HF Transformers Migration

Migrate Hugging Face transformers models to MindOne's transformers implementation.

## When to Use

- Porting BERT, GPT, T5 models to MindSpore
- Migrating LLaMA, Qwen, ChatGLM
- Converting transformers models to mindone
- Adding new transformer architectures

## Target Repository

**mindone.transformers**: https://github.com/mindspore-lab/mindone

## Supported Model Types

- **Encoders**: BERT, RoBERTa, ALBERT, DistilBERT
- **Decoders**: GPT-2, GPT-Neo, GPT-J, LLaMA
- **Encoder-Decoders**: T5, BART, mBART
- **Vision**: ViT, CLIP, BLIP, DeiT
- **Multimodal**: CLIP, BLIP, LLaVA

## Instructions

(TODO: Add detailed migration workflow)

### Step 1: Analyze Source Model

1. Identify the HF transformers model architecture
2. Check if similar architecture exists in mindone.transformers
3. Document API differences between HF and mindone

### Step 2: Weight Conversion

1. Download HF model weights (safetensors/pytorch_model.bin)
2. Map weight names to MindOne format
3. Convert using mindone conversion tools or custom script

### Step 3: Model Migration

1. Identify model components (attention, FFN, embeddings)
2. Map to corresponding mindone components
3. Adjust API calls for MindSpore compatibility

### Step 4: Tokenizer Migration

1. Check tokenizer compatibility
2. Use mindone tokenizer wrappers or convert tokenizer config

### Step 5: Validation

1. Run inference with same inputs on both frameworks
2. Compare hidden states and logits numerically
3. Benchmark performance (tokens/second)

## References

- [mindone.transformers documentation](https://github.com/mindspore-lab/mindone/tree/master/mindone/transformers)
- [HF transformers documentation](https://huggingface.co/docs/transformers)
