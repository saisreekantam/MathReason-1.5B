"""
Stage 2 — Supervised Fine-Tuning (SFT)
───────────────────────────────────────
Server : revanth@172.16.192.168
Path   : ~/nlp/scripts/stage2_sft/stage2_sft.py

Trains Qwen2.5-1.5B on math reasoning data using LoRA + SFTTrainer.
Teaches the model to reason inside <think> tags and answer inside <solution> tags.

Usage:
  # Sanity run first (always do this before full run)
  torchrun --nproc_per_node=2 stage2_sft.py --sanity

  # Full training run
  torchrun --nproc_per_node=2 stage2_sft.py

  # Full run with wandb logging
  torchrun --nproc_per_node=2 stage2_sft.py --wandb --run-name sft_run1

  # Resume from checkpoint
  torchrun --nproc_per_node=2 stage2_sft.py --resume
"""

import argparse
import inspect
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig


# ─── Constants ───────────────────────────────────────────────────────────────

BASE_MODEL    = "Qwen/Qwen2.5-1.5B"
WORK_DIR      = Path("~/nlp").expanduser()
CKPT_DIR      = WORK_DIR / "checkpoints" / "stage2_sft"
MERGED_DIR    = WORK_DIR / "checkpoints" / "stage2_sft_merged"
LOG_DIR       = WORK_DIR / "logs"
DATA_CACHE    = WORK_DIR / "data" / "sft"

SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Think step by step inside <think> tags, "
    "then give your final answer inside <solution> tags."
)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",        type=str,   default=BASE_MODEL)
    p.add_argument("--sanity",       action="store_true",
                   help="Quick sanity run: 10K examples, 200 steps. Run this first.")
    p.add_argument("--resume",       action="store_true",
                   help="Resume from latest checkpoint in CKPT_DIR")
    p.add_argument("--wandb",        action="store_true",
                   help="Enable wandb logging")
    p.add_argument("--run-name",     type=str,   default="stage2_sft",
                   help="Name for this run (wandb + checkpoint subfolder)")
    p.add_argument("--attn-impl",    type=str,   default="auto",
                   choices=["auto", "sdpa", "flash_attention_2", "eager"])
    p.add_argument("--lora-rank",    type=int,   default=64)
    p.add_argument("--lora-alpha",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--epochs",       type=int,   default=3)
    p.add_argument("--batch",        type=int,   default=4,
                   help="Per-device batch size")
    p.add_argument("--grad-accum",   type=int,   default=8,
                   help="Gradient accumulation steps. Effective batch = batch × grad_accum × n_gpus")
    p.add_argument("--max-seq-len",  type=int,   default=8192)
    p.add_argument("--skip-eval",    action="store_true",
                   help="Skip GSM8K eval after training")
    p.add_argument("--skip-merge",   action="store_true",
                   help="Skip merging LoRA adapter into base model after training")
    return p.parse_args()


# ─── Attention backend ───────────────────────────────────────────────────────

def resolve_attn(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import flash_attn  # noqa
        print("✅ flash_attn found — using flash_attention_2")
        return "flash_attention_2"
    except ImportError:
        print("⚠️  flash_attn not installed — using sdpa")
        return "sdpa"


# ─── Dataset loading & formatting ────────────────────────────────────────────

def format_example(problem: str, solution: str, tokenizer) -> str:
    """
    Format a single (problem, solution) pair into the chat template.
    The solution should already contain <think>...</think><solution>...</solution>.
    If it doesn't (raw dataset), we wrap it.
    """
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": problem.strip()},
        {"role": "assistant", "content": solution.strip()},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def wrap_solution(solution: str) -> str:
    """
    Wrap a raw solution string in <think>/<solution> tags if not already present.
    """
    if "<think>" in solution:
        return solution   # already formatted (e.g. R1 traces)

    # Split on #### which is GSM8K/MetaMath answer delimiter
    if "####" in solution:
        parts = solution.split("####")
        reasoning = parts[0].strip()
        answer = parts[1].strip() if len(parts) > 1 else ""
        return f"<think>\n{reasoning}\n</think>\n<solution>\n{answer}\n</solution>"

    # Fallback: put everything in think, extract last number as solution
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", solution)
    answer = numbers[-1].replace(",", "") if numbers else solution[-50:]
    return f"<think>\n{solution.strip()}\n</think>\n<solution>\n{answer}\n</solution>"


def load_numinamath(tokenizer, n_samples=None):
    print("  Loading NuminaMath-CoT...")
    ds = load_dataset(
        "AI-MO/NuminaMath-CoT",
        split="train",
        cache_dir=str(DATA_CACHE),
    )
    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))

    def process(ex):
        problem  = ex.get("problem", ex.get("question", ""))
        solution = ex.get("solution", ex.get("answer", ""))
        solution = wrap_solution(solution)
        return {"text": format_example(problem, solution, tokenizer)}

    ds = ds.map(process, remove_columns=ds.column_names, num_proc=4)
    ds = ds.filter(lambda x: len(x["text"]) > 50)
    print(f"  ✅ NuminaMath-CoT: {len(ds):,} examples")
    return ds


def load_metamathqa(tokenizer, n_samples=None):
    print("  Loading MetaMathQA...")
    ds = load_dataset(
        "meta-math/MetaMathQA",
        split="train",
        cache_dir=str(DATA_CACHE),
    )
    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))

    def process(ex):
        problem  = ex.get("query", "")
        solution = ex.get("response", "")
        solution = wrap_solution(solution)
        return {"text": format_example(problem, solution, tokenizer)}

    ds = ds.map(process, remove_columns=ds.column_names, num_proc=4)
    ds = ds.filter(lambda x: len(x["text"]) > 50)
    print(f"  ✅ MetaMathQA: {len(ds):,} examples")
    return ds





def build_dataset(tokenizer, sanity: bool = False):
    """
    Sanity mode:  10K examples total (fast pipeline verification)
    Full mode:    ~500K examples  (NuminaMath 400K + MetaMath 90K + MATH 7.5K)
    """
    print("\nBuilding SFT dataset...")

    if sanity:
        print("  [SANITY MODE] — small dataset for pipeline verification")
        numina  = load_numinamath(tokenizer,  n_samples=7500)
        meta    = load_metamathqa(tokenizer,  n_samples=2500)
    else:
        print("  [FULL MODE] — ~490K examples")
        numina  = load_numinamath(tokenizer,  n_samples=400000)
        meta    = load_metamathqa(tokenizer,  n_samples=90000)

    combined = concatenate_datasets([numina, meta])
    combined = combined.shuffle(seed=42)

    print(f"\n  Total dataset size: {len(combined):,} examples")
    print(f"  Sample:\n{combined[0]['text'][:300]}...\n")
    return combined


# ─── LoRA config ─────────────────────────────────────────────────────────────

def build_lora_config(rank: int, alpha: int) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        # Cover attention + MLP projections — ~85% of trainable influence
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",   # attention
            "gate_proj", "up_proj", "down_proj",        # MLP (SwiGLU)
        ],
        inference_mode=False,
    )


# ─── Training arguments ──────────────────────────────────────────────────────

def build_training_args(args, sanity: bool, output_dir: Path):
    """
    Build SFTConfig (subclass of TrainingArguments).
    Filters out unsupported kwargs for older transformers versions.
    """
    # Effective batch size = batch × grad_accum × n_gpus
    # With 2 GPUs, batch=4, grad_accum=8: effective = 64
    n_gpus = max(torch.cuda.device_count(), 1)
    eff_batch = args.batch * args.grad_accum * n_gpus
    print(f"\nEffective batch size: {args.batch} × {args.grad_accum} × {n_gpus} GPUs = {eff_batch}")

    # Sanity: only 200 steps, no saving
    max_steps = 200 if sanity else -1
    save_steps = 9999 if sanity else 500
    eval_steps = 9999 if sanity else 500
    logging_steps = 10 if sanity else 50

    report_to = "wandb" if args.wandb else "none"
    if args.wandb:
        os.environ["WANDB_PROJECT"] = "mathReason-1.5B"

    raw_kwargs = dict(
        output_dir=str(output_dir),
        run_name=args.run_name,

        # Steps / epochs
        num_train_epochs=args.epochs,
        max_steps=max_steps,

        # Batch
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,

        # Learning rate
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,

        # Precision
        bf16=True,
        tf32=True,             # speeds up matmul on Ada GPUs

        # Sequence length
        max_seq_length=args.max_seq_len,

        # Packing — combines short examples into one sequence for efficiency
        packing=True,

        # Saving & eval
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        evaluation_strategy="no",
        logging_steps=logging_steps,

        # Optimizer
        optim="adamw_torch_fused",   # faster on CUDA
        weight_decay=0.01,
        max_grad_norm=1.0,

        # Misc
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to=report_to,
        seed=42,

        # Resume
        resume_from_checkpoint=str(output_dir) if args.resume else None,
    )

    # Filter unsupported kwargs for older transformers versions
    valid = set(inspect.signature(SFTConfig.__init__).parameters.keys())
    filtered = {k: v for k, v in raw_kwargs.items() if k in valid}
    removed = set(raw_kwargs.keys()) - set(filtered.keys())
    if removed:
        print(f"  ⚠️  Filtered unsupported TrainingArguments: {removed}")

    return SFTConfig(**filtered)


# ─── Model loading ───────────────────────────────────────────────────────────

def load_base_model(model_name: str, attn_impl: str):
    print(f"\n{'─'*60}")
    print(f"Loading base model : {model_name}")
    print(f"Attn backend       : {attn_impl}")
    print(f"{'─'*60}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",   # important for SFT packing
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
        trust_remote_code=True,
    )

    # Gradient checkpointing — trades compute for memory
    # Saves ~30% VRAM at cost of ~20% slower training. Worth it at 8K seq len.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()

    total = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Parameters: {total:.2f}B")
    return model, tokenizer


# ─── LoRA setup ──────────────────────────────────────────────────────────────

def apply_lora(model, lora_cfg: LoraConfig):
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    # Sanity check — LoRA params should be ~1-4% of total
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e9:.2f}B ({trainable/total*100:.1f}%)")
    return model


# ─── Merge LoRA into base model ──────────────────────────────────────────────

def merge_and_save(model, tokenizer, merged_dir: Path):
    """
    Merges LoRA adapter weights back into the base model.
    This produces a clean full-weight checkpoint for Stage 3.
    """
    print(f"\n{'─'*60}")
    print("Merging LoRA adapter into base model...")

    merged = model.merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))

    print(f"✅ Merged model saved to: {merged_dir}")
    print("   This is the checkpoint to use for Stage 3 CoT Distillation.")
    return merged


# ─── GSM8K quick eval ────────────────────────────────────────────────────────

def eval_gsm8k(model, tokenizer, n_samples=100):
    """
    Quick GSM8K eval after SFT to verify the model learned the format.
    Expect: 78–85% after full SFT, 40–55% after sanity run.
    """
    print(f"\n{'═'*60}")
    print(f"POST-SFT GSM8K EVALUATION  ({n_samples} problems, greedy)")
    print(f"{'═'*60}")

    dataset = load_dataset("gsm8k", "main", split="test")
    dataset = dataset.select(range(min(n_samples, len(dataset))))

    device = next(model.parameters()).device
    correct = 0

    for i, item in enumerate(dataset):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": item["question"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        response   = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Extract answer from <solution> tag first, then fallback to last number
        sol_match = re.search(r"<solution>(.*?)</solution>", response, re.DOTALL)
        pred_text = sol_match.group(1).strip() if sol_match else response
        pred_nums = re.findall(r"-?[\d,]+(?:\.\d+)?", pred_text.replace(",", ""))
        predicted = pred_nums[-1] if pred_nums else None

        gt_nums = re.findall(r"-?[\d,]+", item["answer"].split("####")[-1])
        gt      = gt_nums[-1].replace(",", "") if gt_nums else None

        if predicted and gt and predicted == gt:
            correct += 1

        if (i + 1) % 20 == 0:
            print(f"  [{i+1:>3}/{n_samples}]  running accuracy: {correct/(i+1)*100:.1f}%")

    acc = correct / n_samples * 100
    print(f"\n  Final: {correct}/{n_samples} = {acc:.1f}%")
    print(f"  Baseline (Stage 1): ~35%")
    print(f"  Expected (after full SFT): 78–85%")
    return acc


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args      = parse_args()
    attn_impl = resolve_attn(args.attn_impl)
    sanity    = args.sanity

    # Directories
    run_suffix  = "_sanity" if sanity else ""
    output_dir  = CKPT_DIR / f"{args.run_name}{run_suffix}"
    merged_dir  = MERGED_DIR
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main    = local_rank == 0

    if is_main:
        print("\n" + "█" * 60)
        print("  STAGE 2 — SUPERVISED FINE-TUNING (SFT)")
        if sanity:
            print("  *** SANITY RUN — 10K examples, 200 steps ***")
        print("█" * 60)
        print(f"  Timestamp  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Model      : {args.model}")
        print(f"  LoRA rank  : {args.lora_rank}  alpha: {args.lora_alpha}")
        print(f"  LR         : {args.lr}")
        print(f"  Seq length : {args.max_seq_len}")
        print(f"  Output     : {output_dir}")
        print(f"  Merged to  : {merged_dir}")

    # ── Load model + tokenizer ──
    model, tokenizer = load_base_model(args.model, attn_impl)

    # ── Apply LoRA ──
    lora_cfg = build_lora_config(args.lora_rank, args.lora_alpha)
    model    = apply_lora(model, lora_cfg)

    # ── Build dataset ──
    dataset = build_dataset(tokenizer, sanity=sanity)

    # ── Training arguments ──
    train_args = build_training_args(args, sanity, output_dir)

    # ── SFTTrainer ──
    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        # dataset_text_field tells SFTTrainer which column has the formatted text
        dataset_text_field="text",
    )

    # ── Train ──
    if is_main:
        print(f"\n{'─'*60}")
        print("Starting training...")
        print(f"{'─'*60}\n")

    train_result = trainer.train(
        resume_from_checkpoint=str(output_dir) if args.resume else None
    )

    # ── Save LoRA adapter ──
    if is_main:
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        print(f"\n✅ LoRA adapter saved to: {output_dir}")

        # Save training metrics
        metrics = train_result.metrics
        metrics["train_samples"] = len(dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        with open(LOG_DIR / "stage2_train_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"✅ Metrics saved to: {LOG_DIR / 'stage2_train_metrics.json'}")

    # ── Merge LoRA into base model ──
    if is_main and not args.skip_merge and not sanity:
        merged = merge_and_save(model, tokenizer, merged_dir)
    elif sanity and is_main:
        print("\n  [Sanity mode] — skipping merge. Run full training to produce merged checkpoint.")

    # ── Post-SFT eval ──
    if is_main and not args.skip_eval:
        n_eval = 50 if sanity else 100
        # Use merged model for eval if available, else use LoRA model
        eval_model = merged if (not sanity and not args.skip_merge) else model
        acc = eval_gsm8k(eval_model, tokenizer, n_samples=n_eval)

        # Append to results log
        results = {
            "stage": "2_sft",
            "sanity": sanity,
            "timestamp": datetime.now().isoformat(),
            "gsm8k_accuracy": acc,
            "gsm8k_samples": n_eval,
            "model": args.model,
            "lora_rank": args.lora_rank,
            "epochs": args.epochs,
            "lr": args.lr,
        }
        with open(LOG_DIR / "stage2_eval_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Eval results saved to: {LOG_DIR / 'stage2_eval_results.json'}")

    if is_main:
        print(f"\n{'█'*60}")
        if sanity:
            print("  SANITY RUN COMPLETE")
            print("  ─────────────────────────────────────────────")
            print("  If loss was dropping → run full training:")
            print("  torchrun --nproc_per_node=2 stage2_sft.py")
        else:
            print("  STAGE 2 SFT COMPLETE")
            print("  ─────────────────────────────────────────────")
            print(f"  Merged checkpoint: {merged_dir}")
            print("  Next step → Stage 3 CoT Distillation")
        print(f"{'█'*60}\n")


if __name__ == "__main__":
    main()
