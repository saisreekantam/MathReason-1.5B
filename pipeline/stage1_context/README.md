# Stage 1: Context Extension (4K → 8K → 16K)

This folder implements stage-wise context extension for a RoPE-based base model.

## What is implemented

- `train_context_extension.py`
  - applies RoPE scaling for the requested stage
  - builds a mixed-length curriculum dataset
  - runs continued pretraining with Hugging Face `Trainer`
- `eval_long_context.py`
  - measures loss / perplexity across multiple context lengths

## Libraries used

- `torch`
- `transformers`
- `datasets`
- `flash-attn`
- standard Hugging Face training stack

## Recommended workflow

### Stage A: 4K → 8K

Run continued pretraining with YaRN scaling factor `2.0` and a mixed curriculum of `2K / 4K / 8K` blocks.

Example:

```bash
torchrun --nproc_per_node=2 scripts/stage1_context/train_context_extension.py \
  --stage 8k \
  --model-name-or-path Qwen/Qwen2.5-1.5B \
  --dataset-name open-web-math/open-web-math \
  --train-split train \
  --validation-split train[:1%] \
  --text-field text \
  --output-dir ~/nlp/checkpoints/stage1_context/4k_to_8k \
  --gradient-checkpointing
```

Then evaluate:

```bash
python scripts/stage1_context/eval_long_context.py \
  --model-name-or-path ~/nlp/checkpoints/stage1_context/4k_to_8k \
  --dataset-name open-web-math/open-web-math \
  --split train[:1%] \
  --lengths 4096 8192
```

### Stage B: 8K → 16K

Load the 8K checkpoint and continue with factor `4.0` and curriculum `4K / 8K / 16K`.

```bash
torchrun --nproc_per_node=2 scripts/stage1_context/train_context_extension.py \
  --stage 16k \
  --model-name-or-path ~/nlp/checkpoints/stage1_context/4k_to_8k \
  --dataset-name open-web-math/open-web-math \
  --train-split train \
  --validation-split train[:1%] \
  --text-field text \
  --output-dir ~/nlp/checkpoints/stage1_context/8k_to_16k \
  --gradient-checkpointing
```

Then evaluate:

```bash
python scripts/stage1_context/eval_long_context.py \
  --model-name-or-path ~/nlp/checkpoints/stage1_context/8k_to_16k \
  --dataset-name open-web-math/open-web-math \
  --split train[:1%] \
  --lengths 4096 8192 16384
```

## Notes

- This code assumes a **true extension recipe**: 4K → 8K → 16K.
- If the base model already supports a larger native window, use this carefully and validate that scaling is actually needed.
- Later SFT / CoT stages should still include some long examples, otherwise long-context gains can fade.
