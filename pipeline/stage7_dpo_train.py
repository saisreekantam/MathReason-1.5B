"""
Stage 7 — DPO Alignment
───────────────────────
Server : revanth@172.16.192.168
Path   : ~/nlp/scripts/stage7_dpo_train.py

Trains gdpo_p2_v1_merged on self-generated preference pairs to fix:
  1. Loop failures   — brute-force iteration instead of algebra (Q03 type)
  2. Method selection — wrong formula choice on mixture/work-rate (Q13 type)

Uses LoRA (rank 32) to keep update surgical — DPO on full params overfits fast.

Usage:
    # Sanity check first (always)
    CUDA_VISIBLE_DEVICES=1 python3 stage7_dpo_train.py --sanity

    # Full run
    CUDA_VISIBLE_DEVICES=1 python3 stage7_dpo_train.py

    # With wandb
    CUDA_VISIBLE_DEVICES=1 python3 stage7_dpo_train.py --wandb --run-name dpo_v1

    # Resume from checkpoint
    CUDA_VISIBLE_DEVICES=1 python3 stage7_dpo_train.py --resume

Checkpoint lineage:
    gdpo_p2_v1_merged  →  stage7_dpo/  →  stage7_dpo_merged
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

# ─── Paths ───────────────────────────────────────────────────────────────────

WORK_DIR    = Path("~/nlp").expanduser()
MODEL_PATH  = WORK_DIR / "checkpoints" / "gdpo_p2_v1_merged"
CKPT_DIR    = WORK_DIR / "checkpoints" / "stage7_dpo"
MERGED_DIR  = WORK_DIR / "checkpoints" / "stage7_dpo_merged"
LOG_DIR     = WORK_DIR / "logs"
DATA_FILE   = WORK_DIR / "data" / "dpo_pairs" / "dpo_pairs_train.jsonl"

SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Think step by step inside <think> tags, "
    "then give your final answer inside <solution> tags."
)


# ─── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sanity",    action="store_true",
                   help="10-step smoke test — always run before full training")
    p.add_argument("--resume",    action="store_true",
                   help="Resume from latest checkpoint in CKPT_DIR")
    p.add_argument("--wandb",     action="store_true")
    p.add_argument("--run-name",  type=str, default=f"dpo_v1_{datetime.now():%m%d_%H%M}")

    # Core hyperparams — do not change without good reason
    p.add_argument("--beta",      type=float, default=0.1,
                   help="KL penalty. 0.1 = surgical update. Higher = more conservative.")
    p.add_argument("--lr",        type=float, default=5e-7,
                   help="10x lower than SFT. DPO is fine-grained alignment, not training.")
    p.add_argument("--epochs",    type=int,   default=1,
                   help="1 epoch ONLY. DPO overfits extremely fast.")
    p.add_argument("--lora-rank", type=int,   default=32,
                   help="LoRA rank. 32 = surgical. Don't go higher for DPO.")
    return p.parse_args()


# ─── Dataset loading ──────────────────────────────────────────────────────────

def load_dpo_dataset(tokenizer, sanity: bool) -> Dataset:
    """
    Loads dpo_pairs_train.jsonl and formats into the 3-column schema
    DPOTrainer expects: prompt, chosen, rejected.

    Each field is a list of chat messages (role/content dicts).
    The prompt contains system + user. chosen/rejected are assistant turns only.
    """
    print(f"\n{'─'*60}")
    print(f"Loading DPO pairs from: {DATA_FILE}")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"DPO train file not found: {DATA_FILE}\n"
            "Run stage7_build_dpo_data.py first."
        )

    rows = []
    with open(DATA_FILE) as f:
        for line in f:
            obj = json.loads(line.strip())
            rows.append(obj)

    if sanity:
        rows = rows[:32]
        print(f"  SANITY MODE — using {len(rows)} pairs")
    else:
        print(f"  Loaded {len(rows)} pairs")

    # Source breakdown
    source_counts: dict[str, int] = {}
    signal_counts: dict[str, int] = {}
    for r in rows:
        source_counts[r.get("source", "unknown")] = source_counts.get(r.get("source", "unknown"), 0) + 1
        signal_counts[r.get("signal", "?")] = signal_counts.get(r.get("signal", "?"), 0) + 1

    print("  Source breakdown:")
    for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"    {src:<35}: {cnt}")
    print(f"  Strong signal pairs: {signal_counts.get('strong', 0)}")
    print(f"  Weak signal pairs:   {signal_counts.get('weak', 0)}")

    # Build dataset in DPOTrainer's expected format:
    #   prompt   = list of messages up to (not including) the assistant turn
    #   chosen   = list with single assistant message (correct completion)
    #   rejected = list with single assistant message (wrong completion)
    prompts, chosens, rejecteds = [], [], []

    for r in rows:
        question = r.get("question", "")

        # Reconstruct question from prompt string if question field missing
        if not question:
            # Fall back: extract user content from raw prompt string
            m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", r["prompt"], re.DOTALL)
            question = m.group(1).strip() if m else r["prompt"]

        prompt_messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": question},
        ]
        chosen_messages   = [{"role": "assistant", "content": r["chosen"]}]
        rejected_messages = [{"role": "assistant", "content": r["rejected"]}]

        prompts.append(prompt_messages)
        chosens.append(chosen_messages)
        rejecteds.append(rejected_messages)

    dataset = Dataset.from_dict({
        "prompt":   prompts,
        "chosen":   chosens,
        "rejected": rejecteds,
    })

    print(f"  Dataset size: {len(dataset)}")
    print(f"{'─'*60}\n")
    return dataset


# ─── Model + LoRA ─────────────────────────────────────────────────────────────

def load_model_and_tokenizer(args):
    print(f"{'─'*60}")
    print(f"Loading model: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Expected: gdpo_p2_v1_merged"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
        padding_side="left",    # DPO requires left-padding for batch generation
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",   # flash_attn blocked by glibc — SDPA throughout
        device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
        trust_remote_code=True,
    )

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Parameters: {total_params:.2f}B")

    # LoRA — rank 32 is intentionally conservative for DPO
    # Full-param DPO with 800 pairs would overfit in <200 steps
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,   # alpha = 2×rank is standard
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        # Do NOT add input embeddings or LM head — DPO only needs attention/MLP
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e9:.2f}B ({trainable/total*100:.1f}%)")
    print(f"{'─'*60}\n")

    return model, tokenizer


# ─── DPO config ───────────────────────────────────────────────────────────────

def build_dpo_config(args, dataset_len: int) -> DPOConfig:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Batch math: effective batch = 2 * 8 = 16 per step
    # At 800 pairs, 1 epoch = 50 steps. DPO should converge in 30-60 steps.
    batch_size  = 2
    grad_accum  = 8
    eff_batch   = batch_size * grad_accum
    steps_per_epoch = dataset_len // eff_batch
    total_steps = steps_per_epoch * args.epochs

    if args.sanity:
        total_steps  = 10
        save_steps   = 5
        logging_steps = 1
    else:
        # Save 3 checkpoints evenly through training
        save_steps    = max(total_steps // 3, 10)
        logging_steps = 5

    print(f"  DPO Training Config:")
    print(f"    input model   : gdpo_p2_v1_merged")
    print(f"    beta          : {args.beta}  (KL penalty — 0.1 = surgical)")
    print(f"    lr            : {args.lr}  (10x lower than SFT)")
    print(f"    epochs        : {args.epochs}  (1 only — DPO overfits fast)")
    print(f"    lora_rank     : {args.lora_rank}")
    print(f"    eff_batch     : {eff_batch}  ({batch_size} × {grad_accum} accum)")
    print(f"    est. steps    : {total_steps}  (~{steps_per_epoch}/epoch)")
    print(f"    loss_type     : sigmoid  (standard DPO)")
    print(f"    label_smoothing: 0.0  (MANDATORY — learned from Stage 3)")
    print()

    report_to = "wandb" if args.wandb else "none"

    resume_path = None
    if args.resume and CKPT_DIR.exists():
        ckpts = sorted(CKPT_DIR.glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[-1]))
        if ckpts:
            resume_path = str(ckpts[-1])
            print(f"  Resuming from: {resume_path}")

    cfg = DPOConfig(
        output_dir=str(CKPT_DIR),
        run_name=args.run_name,

        # Core DPO
        beta=args.beta,
        loss_type="sigmoid",          # standard DPO loss
        label_smoothing=0.0,          # MANDATORY — any value >0 degrades math perf

        # Training
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=min(10, total_steps // 10),

        # Batch
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,

        # Sequence lengths — must cover full chosen/rejected completions
        max_length=2048,              # total prompt + completion

        # Precision
        bf16=True,

        # Optimizer
        optim="adamw_torch_fused",
        weight_decay=0.01,
        max_grad_norm=1.0,

        # Saving
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,

        # Eval — no held-out set, rely on post-training GSM8K eval
        eval_strategy="no",

        # Logging
        logging_steps=logging_steps,
        report_to=report_to,

        # Required for PEFT/LoRA compatibility
        remove_unused_columns=False,
        seed=42,

        resume_from_checkpoint=resume_path,
    )

    return cfg


# ─── Merge LoRA ───────────────────────────────────────────────────────────────

def merge_and_save(model, tokenizer):
    print(f"\n{'─'*60}")
    print("Merging LoRA adapter into base model...")

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(str(MERGED_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(MERGED_DIR))

    size_gb = sum(
        f.stat().st_size for f in MERGED_DIR.glob("*.safetensors")
    ) / 1e9
    print(f"  ✅ Merged model saved to: {MERGED_DIR}  ({size_gb:.1f} GB)")
    print(f"{'─'*60}\n")


# ─── Quick sanity eval ────────────────────────────────────────────────────────

def quick_eval(model, tokenizer, n: int = 5):
    """
    3-problem smoke test after training.
    Checks: solution tag present, no obvious infinite loop, answer roughly plausible.
    Not a real benchmark — run eval_gsm8k_smart.py separately.
    """
    problems = [
        {"q": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes 4 into muffins. She sells the rest at $2 each. How much does she make daily?",
         "gt": "18"},
        {"q": "A train travels 120 miles in 2 hours. How far will it travel in 5 hours at the same speed?",
         "gt": "300"},
        {"q": "If 8 workers can build a wall in 12 days, how many days would 6 workers take to build the same wall?",
         "gt": "16"},
        {"q": "A jar contains red and blue marbles. 40% are red. If there are 30 blue marbles, how many marbles are in the jar?",
         "gt": "50"},
        {"q": "Pipe A fills a tank in 4 hours. Pipe B fills it in 6 hours. How long to fill the tank if both pipes are open?",
         "gt": "2.4"},
    ][:n]

    print(f"\n{'─'*60}")
    print("Quick eval (smoke test — not a real benchmark):")
    print(f"{'─'*60}")

    model.eval()
    correct = 0

    for i, prob in enumerate(problems):
        msgs = [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": prob["q"]},
        ]
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )

        resp = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        # Extract answer
        m = re.search(r"<solution>(.*?)</solution>", resp, re.DOTALL | re.IGNORECASE)
        if m:
            pred = re.sub(r"[,$\s]", "", m.group(1).strip())
        else:
            nums = re.findall(r"[\d.]+", resp[:int(len(resp)*0.6)])
            pred = nums[-1] if nums else "?"

        try:
            is_correct = abs(float(pred) - float(prob["gt"])) < 0.5
        except Exception:
            is_correct = pred == prob["gt"]

        tag_ok   = "</solution>" in resp
        loop_ok  = resp.count("Wait") < 5 and len(resp.split()) < 600

        status = "✅" if is_correct else "❌"
        warn   = "" if (tag_ok and loop_ok) else f"  ⚠️  tag={tag_ok} loop_ok={loop_ok}"
        print(f"  Q{i+1} {status}  pred={pred:>8}  gt={prob['gt']:>8}{warn}")

        if is_correct:
            correct += 1

    print(f"\n  Score: {correct}/{n}  ({correct/n*100:.0f}%)")
    print(f"  (Run eval_gsm8k_smart.py for real benchmark numbers)")
    print(f"{'─'*60}\n")
    return correct / n


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print("Stage 7 — DPO Alignment  |  MathReason-1.5B")
    print(f"Run   : {args.run_name}")
    print(f"Mode  : {'SANITY (10 steps)' if args.sanity else 'FULL'}")
    print(f"GPU   : {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print("=" * 60)

    if args.sanity:
        print("\n⚠️  SANITY MODE — 10 steps, 32 pairs, no merge")
        print("   If this completes cleanly, re-run without --sanity\n")

    # Load data
    model, tokenizer = load_model_and_tokenizer(args)
    dataset = load_dpo_dataset(tokenizer, sanity=args.sanity)
    dpo_cfg = build_dpo_config(args, dataset_len=len(dataset))

    # DPOTrainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,     # None = use implicit reference via KL from LoRA init
                            # This works because LoRA base weights ARE the reference
        args=dpo_cfg,
        train_dataset=dataset,
        processing_class=tokenizer,   # TRL 0.29: processing_class, NOT tokenizer=
    )

    # Train
    print("Starting DPO training...\n")
    t0 = __import__("time").time()
    trainer.train()
    elapsed = (__import__("time").time() - t0) / 60

    print(f"\n✅ Training complete in {elapsed:.1f} minutes")
    print(f"   Checkpoints saved to: {CKPT_DIR}")

    if args.sanity:
        print("\n✅ Sanity run passed — re-run without --sanity for full training")
        return

    # Save final adapter explicitly (in case training ended mid-save-interval)
    final_adapter_dir = CKPT_DIR / "final_adapter"
    model.save_pretrained(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))
    print(f"   Final adapter saved: {final_adapter_dir}")

    # Quick eval before merge
    quick_eval(model, tokenizer, n=5)

    # Merge LoRA → full weights
    merge_and_save(model, tokenizer)

    # Log to file
    log_entry = {
        "timestamp":   datetime.now().isoformat(),
        "run_name":    args.run_name,
        "input_model": str(MODEL_PATH),
        "output":      str(MERGED_DIR),
        "beta":        args.beta,
        "lr":          args.lr,
        "lora_rank":   args.lora_rank,
        "pairs":       len(dataset),
        "elapsed_min": round(elapsed, 1),
    }
    log_path = LOG_DIR / "stage7_dpo_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    print(f"   Run logged to: {log_path}")

    print("\n" + "=" * 60)
    print("Stage 7 complete.")
    print(f"Merged model: {MERGED_DIR}")
    print()
    print("Next steps:")
    print("  1. Run GSM8K eval:  CUDA_VISIBLE_DEVICES=0 python3 eval_gsm8k_smart.py")
    print("  2. Run MATH500 eval: CUDA_VISIBLE_DEVICES=0 python3 eval_math500.py")
    print("  3. If GSM8K ≥ 90% and loops reduced → proceed to test-time compute")
    print("  4. If regression → check beta (try 0.05) or reduce epochs")
    print("=" * 60)


if __name__ == "__main__":
    main()
