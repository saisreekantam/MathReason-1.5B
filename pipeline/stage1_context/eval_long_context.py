from __future__ import annotations

import argparse
import json
import math
from typing import Sequence

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_LENGTHS = (4096, 8192, 16384)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate perplexity across multiple context lengths.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        choices=("flash_attention_2", "sdpa", "eager"),
        help="Attention kernel. If FlashAttention2 is unavailable, code will automatically fall back to SDPA.",
    )
    return parser.parse_args()


def load_model_with_attention_fallback(args: argparse.Namespace):
    try:
        return AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
            cache_dir=args.cache_dir,
        )
    except ImportError as exc:
        if args.attn_implementation == "flash_attention_2":
            print("[WARN] flash_attn is not installed. Falling back to SDPA.")
            print(f"[WARN] Original error: {exc}")
            return AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                cache_dir=args.cache_dir,
            )
        raise


def resolve_text_field(dataset: Dataset, preferred: str) -> str:
    if preferred in dataset.column_names:
        return preferred
    for fallback in ("text", "content", "body", "document"):
        if fallback in dataset.column_names:
            return fallback
    raise ValueError(f"Could not find text field. Available columns: {dataset.column_names}")


def build_blocks(dataset: Dataset, tokenizer, text_field: str, block_size: int, max_samples: int) -> list[torch.Tensor]:
    token_buffer: list[int] = []
    blocks: list[torch.Tensor] = []

    for row in dataset:
        text = row.get(text_field)
        if not isinstance(text, str) or not text.strip():
            continue
        token_buffer.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        while len(token_buffer) >= block_size and len(blocks) < max_samples:
            block = token_buffer[:block_size]
            token_buffer = token_buffer[block_size:]
            blocks.append(torch.tensor(block, dtype=torch.long))
        if len(blocks) >= max_samples:
            break

    return blocks


def evaluate_length(model, blocks: Sequence[torch.Tensor], batch_size: int, device: torch.device) -> dict[str, float]:
    if not blocks:
        return {"loss": float("nan"), "perplexity": float("nan"), "num_blocks": 0}

    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(blocks), batch_size):
            batch_blocks = blocks[start : start + batch_size]
            input_ids = torch.stack(batch_blocks).to(device)
            attention_mask = torch.ones_like(input_ids, device=device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            losses.append(outputs.loss.item())

    mean_loss = sum(losses) / len(losses)
    return {
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "num_blocks": len(blocks),
    }


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, cache_dir=args.cache_dir)
    model = load_model_with_attention_fallback(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split, cache_dir=args.cache_dir)
    text_field = resolve_text_field(dataset, args.text_field)

    results: dict[str, dict[str, float]] = {}
    for length in args.lengths:
        blocks = build_blocks(dataset, tokenizer, text_field, length, args.max_samples)
        metrics = evaluate_length(model, blocks, args.batch_size, device)
        results[str(length)] = metrics
        print(f"length={length} blocks={metrics['num_blocks']} loss={metrics['loss']:.4f} ppl={metrics['perplexity']:.4f}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
