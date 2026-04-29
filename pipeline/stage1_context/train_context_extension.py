from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from datasets import Dataset, DatasetDict, interleave_datasets, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


@dataclass(frozen=True)
class StageSpec:
    name: str
    target_length: int
    curriculum_lengths: tuple[int, ...]
    curriculum_weights: tuple[float, ...]
    learning_rate: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    eval_steps: int
    save_steps: int


STAGE_SPECS: dict[str, StageSpec] = {
    "8k": StageSpec(
        name="4k_to_8k",
        target_length=8192,
        curriculum_lengths=(2048, 4096, 8192),
        curriculum_weights=(0.10, 0.40, 0.50),
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        eval_steps=100,
        save_steps=200,
    ),
    "16k": StageSpec(
        name="8k_to_16k",
        target_length=16384,
        curriculum_lengths=(4096, 8192, 16384),
        curriculum_weights=(0.15, 0.35, 0.50),
        learning_rate=7e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        eval_steps=100,
        save_steps=200,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-wise context extension with YaRN scaling.")
    parser.add_argument("--stage", choices=tuple(STAGE_SPECS), required=True, help="Extension stage to run.")
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--dataset-name", required=True, help="HF dataset name, e.g. open-web-math/open-web-math")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default=None)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=1024)
    parser.add_argument("--preprocessing-batch-size", type=int, default=1000)
    parser.add_argument("--num-proc", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--original-max-position-embeddings", type=int, default=4096)
    parser.add_argument("--rope-scaling-type", default="yarn", choices=("yarn", "dynamic", "linear"))
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


def apply_rope_extension(model, target_length: int, original_length: int, scaling_type: str) -> None:
    factor = target_length / float(original_length)
    rope_scaling = {
        "type": scaling_type,
        "factor": factor,
        "original_max_position_embeddings": original_length,
    }

    model.config.rope_scaling = rope_scaling
    model.config.max_position_embeddings = target_length
    if hasattr(model.config, "max_length"):
        model.config.max_length = target_length


def _get_text_column(dataset: Dataset | DatasetDict, preferred: str) -> str:
    if isinstance(dataset, DatasetDict):
        sample_columns = next(iter(dataset.values())).column_names
    else:
        sample_columns = dataset.column_names

    if preferred in sample_columns:
        return preferred

    for fallback in ("text", "content", "body", "document"):
        if fallback in sample_columns:
            return fallback

    raise ValueError(f"Could not find a text column. Available columns: {sample_columns}")


def load_raw_splits(args: argparse.Namespace) -> tuple[Dataset, Dataset | None, str]:
    dataset_kwargs = {
        "path": args.dataset_name,
        "name": args.dataset_config,
        "cache_dir": args.cache_dir,
    }
    train_dataset = load_dataset(split=args.train_split, **dataset_kwargs)
    eval_dataset = load_dataset(split=args.validation_split, **dataset_kwargs) if args.validation_split else None

    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    if eval_dataset is not None and args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))

    text_field = _get_text_column(train_dataset, args.text_field)
    return train_dataset, eval_dataset, text_field


def tokenize_corpus(dataset: Dataset, tokenizer, text_field: str, batch_size: int, num_proc: int | None) -> Dataset:
    def tokenize_batch(batch: dict[str, Sequence[str]]) -> dict[str, list[list[int]]]:
        texts = [text for text in batch[text_field] if isinstance(text, str) and text.strip()]
        tokenized = tokenizer(texts, add_special_tokens=False)
        return {"input_ids": tokenized["input_ids"]}

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    tokenized = tokenized.filter(lambda example: len(example["input_ids"]) > 0, desc="Dropping empty rows")
    return tokenized


def build_blocks(tokenized_dataset: Dataset, block_size: int) -> Dataset:
    def group_texts(batch: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        concatenated: list[int] = []
        for token_ids in batch["input_ids"]:
            concatenated.extend(token_ids)

        total_length = (len(concatenated) // block_size) * block_size
        if total_length == 0:
            return {"input_ids": [], "labels": [], "attention_mask": []}

        blocks = [concatenated[i : i + block_size] for i in range(0, total_length, block_size)]
        return {
            "input_ids": blocks,
            "labels": [block[:] for block in blocks],
            "attention_mask": [[1] * block_size for _ in blocks],
        }

    return tokenized_dataset.map(
        group_texts,
        batched=True,
        batch_size=1000,
        remove_columns=tokenized_dataset.column_names,
        desc=f"Packing {block_size}-token blocks",
    )


def build_curriculum_dataset(tokenized_dataset: Dataset, stage_spec: StageSpec) -> Dataset:
    bucketed_datasets: list[Dataset] = []
    for block_size in stage_spec.curriculum_lengths:
        bucket = build_blocks(tokenized_dataset, block_size)
        if len(bucket) == 0:
            continue
        bucketed_datasets.append(bucket)

    if not bucketed_datasets:
        raise ValueError("No training blocks were produced. Check dataset size and tokenizer output.")

    weights = list(stage_spec.curriculum_weights[: len(bucketed_datasets)])
    total = sum(weights)
    probabilities = [weight / total for weight in weights]
    return interleave_datasets(bucketed_datasets, probabilities=probabilities, seed=42, stopping_strategy="all_exhausted")


def save_stage_metadata(args: argparse.Namespace, stage_spec: StageSpec, tokenizer, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    metadata = {
        "stage": stage_spec.name,
        "target_length": stage_spec.target_length,
        "curriculum_lengths": list(stage_spec.curriculum_lengths),
        "curriculum_weights": list(stage_spec.curriculum_weights),
        "rope_scaling_type": args.rope_scaling_type,
        "rope_scaling_factor": stage_spec.target_length / float(args.original_max_position_embeddings),
        "original_max_position_embeddings": args.original_max_position_embeddings,
        "tokenizer_model_max_length": tokenizer.model_max_length,
        "base_model": args.model_name_or_path,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "text_field": args.text_field,
    }
    with open(os.path.join(output_dir, "stage_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def build_training_arguments(**kwargs) -> TrainingArguments:
    supported_args = inspect.signature(TrainingArguments.__init__).parameters
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in supported_args and value is not None}
    ignored_args = sorted(set(kwargs) - set(filtered_kwargs))
    if ignored_args:
        print(f"[WARN] Ignoring unsupported TrainingArguments keys: {', '.join(ignored_args)}")
    return TrainingArguments(**filtered_kwargs)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    stage_spec = STAGE_SPECS[args.stage]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = stage_spec.target_length

    model = load_model_with_attention_fallback(args)
    apply_rope_extension(
        model,
        target_length=stage_spec.target_length,
        original_length=args.original_max_position_embeddings,
        scaling_type=args.rope_scaling_type,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    train_raw, eval_raw, text_field = load_raw_splits(args)
    train_tokenized = tokenize_corpus(train_raw, tokenizer, text_field, args.preprocessing_batch_size, args.num_proc)
    train_dataset = build_curriculum_dataset(train_tokenized, stage_spec)

    eval_dataset = None
    if eval_raw is not None:
        eval_tokenized = tokenize_corpus(eval_raw, tokenizer, text_field, args.preprocessing_batch_size, args.num_proc)
        eval_dataset = build_blocks(eval_tokenized, stage_spec.target_length)
        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))

    training_args = build_training_arguments(
        output_dir=args.output_dir,
        overwrite_output_dir=False,
        per_device_train_batch_size=stage_spec.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=stage_spec.gradient_accumulation_steps,
        learning_rate=stage_spec.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=stage_spec.save_steps,
        eval_steps=stage_spec.eval_steps if eval_dataset is not None else None,
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        report_to="none",
        dataloader_num_workers=2,
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    save_stage_metadata(args, stage_spec, tokenizer, args.output_dir)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = trainer.evaluate() if eval_dataset is not None else {}
    if metrics.get("eval_loss") is not None:
        metrics["eval_perplexity"] = math.exp(metrics["eval_loss"])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()

