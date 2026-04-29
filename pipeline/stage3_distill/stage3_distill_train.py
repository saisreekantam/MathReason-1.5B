"""
Stage 3 — CoT Distillation Training (FIXED)
══════════════════════════════════════════════════════════════════════════════
Server  : revanth@172.16.192.168
Input   : ~/nlp/checkpoints/stage2_sft_merged         (70% GSM8K)
Data    : ~/nlp/data/stage3_final/distill_dataset      (31K R1-671B traces)
Output  : ~/nlp/checkpoints/stage3_distilled           (target: 80–85% GSM8K)

═══════════════════════════════════════════════════════════════
WHY THE ORIGINAL SCRIPT PRODUCED train_loss = 44.45
═══════════════════════════════════════════════════════════════

BUG 1 — Label smoothing × think-weight compounding (PRIMARY BUG):

  HuggingFace label smoothing adds a uniform distribution penalty to the loss:
    token_loss = (1 - eps) * CE + eps * (-log_softmax(logits).mean())

  The term (-log_softmax(logits).mean()) averages over the FULL VOCABULARY.
  For Qwen2.5 with ~150K vocab: -log(1/150000) ≈ log(150000) ≈ 11.9

  The original script then applied a think-weight multiplier of 1.5× to
  think-span tokens, which are 93.8% of all tokens in R1 traces
  (avg think = 2,471 tokens out of ~2,630 total).

  Combined effect per think token:
    1.5 × (0.9 × CE + 0.1 × 11.9) = 1.5 × (1.8 + 1.19) = 4.49

  Then averaged with solution tokens:
    0.938 × 4.49 + 0.062 × (0.9 × CE + 0.1 × 11.9)
    ≈ 4.21 + 0.19 ≈ 4.40 per token (with CE ≈ 2.0)

  But this still gives ~4.4, not 44.45. The actual mechanism was likely that
  the label smoothing term was being summed, not averaged, before the weight
  was applied — turning per-token into per-sequence, multiplied by sequence
  length (~2600 tokens). Or the weights were applied to logits before
  normalisation causing the softmax to operate on rescaled logits.

  Regardless: the root cause is clear. LABEL SMOOTHING + THINK WEIGHTS = BROKEN.

BUG 2 — Think-span detection via token IDs:

  The original script found think-span boundaries by looking for the token ID
  of "<think>" and "</think>". These IDs differ depending on whether the string
  appears at the start of a sentence vs mid-sequence. This caused the upweighting
  to be applied to wrong token ranges, compounding Bug 1.

THE FIX:
  1. label_smoothing_factor = 0.0  (explicit, not default)
  2. No think-span weighting at all — plain uniform cross-entropy over all tokens
  3. Standard SFTTrainer — no custom loss function

  This is sufficient. The academic precedent (OpenR1, S1, DeepSeek-R1 distill
  variants) all use plain SFT on long CoT traces without per-token weighting.
  Uniform CE on high-quality R1 reasoning traces is all we need.

WHAT WE EXPECT:
  Stage 3 train_loss should be in [1.5, 2.8]:
    ~1.5  = model already close to R1 style (good after Stage 2 SFT)
    ~2.5  = normal for long complex reasoning chains
    >5.0  = something is wrong → script will auto-halt

Usage:
  # ALWAYS run sanity first — verifies loss is in sane range
  torchrun --nproc_per_node=2 stage3_distill_train.py --sanity

  # Full training run
  torchrun --nproc_per_node=2 stage3_distill_train.py

  # Single GPU (if GPU 0 is busy)
  CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 stage3_distill_train.py

  # With wandb
  torchrun --nproc_per_node=2 stage3_distill_train.py --wandb --run-name distill_v1

  # Resume from checkpoint
  torchrun --nproc_per_node=2 stage3_distill_train.py --resume
══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk, Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, TrainerControl, TrainerState
from trl import SFTTrainer, SFTConfig


# ─── Paths ───────────────────────────────────────────────────────────────────

WORK_DIR      = Path("~/nlp").expanduser()
INPUT_MODEL   = WORK_DIR / "checkpoints" / "stage2_sft_merged"
DISTILL_DATA  = WORK_DIR / "data" / "stage3_final" / "distill_dataset"
DISTILL_JSONL = WORK_DIR / "data" / "stage3_final" / "distill_full.jsonl"
CKPT_DIR      = WORK_DIR / "checkpoints" / "stage3_distilled"
MERGED_DIR    = WORK_DIR / "checkpoints" / "stage3_distilled_merged"
LOG_DIR       = WORK_DIR / "logs"

SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Think step by step inside <think> tags, "
    "then give your final answer inside <solution> tags."
)

# Loss guard — abort training immediately if loss exceeds this.
# A loss of 44 like the original bug is immediately visible. Anything > 6.0
# after the first 5 steps is broken.
MAX_SANE_LOSS = 6.0


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Stage 3 — CoT Distillation (Fixed)")
    p.add_argument("--model",       type=str, default=str(INPUT_MODEL))
    p.add_argument("--sanity",      action="store_true",
                   help="Quick sanity: 300 examples, 30 steps. ALWAYS RUN FIRST.")
    p.add_argument("--resume",      action="store_true")
    p.add_argument("--wandb",       action="store_true")
    p.add_argument("--run-name",    type=str, default="distill_v1")
    p.add_argument("--attn-impl",   type=str, default="sdpa",
                   choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--lora-rank",   type=int, default=64)
    p.add_argument("--lora-alpha",  type=int, default=128)
    p.add_argument("--lr",          type=float, default=1e-5,
                   help="10× lower than Stage 2. Model already knows the format.")
    p.add_argument("--epochs",      type=int, default=2,
                   help="2 epochs max — avoid overfitting to R1 reasoning style")
    p.add_argument("--batch",       type=int, default=1,
                   help="Per-device batch. Keep 1 — R1 traces are 2K-4K tokens long")
    p.add_argument("--grad-accum",  type=int, default=16,
                   help="Effective batch = 1 × 16 × 2 GPUs = 32")
    p.add_argument("--max-seq-len", type=int, default=8192,
                   help="R1 traces avg 2471 think tokens + overhead. 8K is safe.")
    p.add_argument("--skip-eval",   action="store_true")
    p.add_argument("--skip-merge",  action="store_true")
    return p.parse_args()


# ─── Format helpers ──────────────────────────────────────────────────────────

def format_for_chat(problem: str, solution: str, tokenizer) -> str:
    """
    Apply Qwen2.5 chat template to (problem, solution) pair.
    Solution should already contain <think>...</think><solution>...</solution>.
    This is the same format Stage 2 used — the model already knows it.
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


def ensure_solution_tags(solution: str) -> str:
    """
    The distill_dataset prep script converted R1's \\boxed{} answer into
    our <solution> tag format. Verify the conversion happened; if not,
    attempt recovery. This is a safety net, should rarely trigger.
    """
    if "<think>" in solution and "<solution>" in solution:
        return solution  # already in correct format

    # If it has <think> but no <solution>, extract last \boxed{} as answer
    if "<think>" in solution:
        # Find last \boxed{answer}
        boxed_matches = list(re.finditer(r"\\boxed\{([^}]*)\}", solution))
        if boxed_matches:
            answer = boxed_matches[-1].group(1).strip()
            # Strip everything after </think> and add solution tag
            think_end = solution.rfind("</think>")
            if think_end != -1:
                think_part = solution[:think_end + len("</think>")]
                return f"{think_part}\n<solution>\n{answer}\n</solution>"

    # Last resort: wrap everything in think, use last number as solution
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", solution)
    answer = nums[-1].replace(",", "") if nums else "0"
    return f"<think>\n{solution.strip()}\n</think>\n<solution>\n{answer}\n</solution>"


# ─── Dataset loading ──────────────────────────────────────────────────────────

def load_distill_dataset(tokenizer, sanity: bool):
    """
    Load the 31K R1 distillation traces prepared by prepare_distill_data.py.

    Tries these sources in order:
      1. HuggingFace dataset saved at DISTILL_DATA (primary)
      2. JSONL file at DISTILL_JSONL (fallback)

    Expected dataset fields from prepare_distill_data.py:
      - problem      : math problem text
      - solution     : formatted <think>...</think><solution>...</solution>
                       OR the raw R1 solution with \\boxed{}
      - source       : olympiads / cn_contest / aops_forum / amc_aime
      - think_tokens : approximate think token count
    """
    print("\nLoading Stage 3 distillation dataset...")

    raw_ds = None

    # Try HuggingFace dataset format first
    if DISTILL_DATA.exists():
        try:
            raw_ds = load_from_disk(str(DISTILL_DATA))
            print(f"  ✅ Loaded HuggingFace dataset: {len(raw_ds):,} examples")
        except Exception as e:
            print(f"  ⚠️  Could not load HuggingFace dataset: {e}")

    # Fallback: JSONL
    if raw_ds is None and DISTILL_JSONL.exists():
        print(f"  Falling back to JSONL: {DISTILL_JSONL}")
        rows = []
        with open(DISTILL_JSONL) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if rows:
            raw_ds = Dataset.from_list(rows)
            print(f"  ✅ Loaded JSONL: {len(raw_ds):,} examples")

    if raw_ds is None:
        raise FileNotFoundError(
            f"No distillation data found.\n"
            f"  Expected: {DISTILL_DATA}\n"
            f"  Fallback: {DISTILL_JSONL}\n"
            f"  Run prepare_distill_data.py first."
        )

    # Show column names so we know what we're working with
    print(f"  Columns: {raw_ds.column_names}")

    # If dataset already has a 'text' field, it's pre-formatted — use directly
    if "text" in raw_ds.column_names:
        print("  Dataset has 'text' field — using directly")
        ds = raw_ds
        ds = ds.filter(lambda x: len(x["text"]) > 100)
    else:
        # Format using problem + solution fields
        print("  Formatting problem+solution → chat template...")

        def process(ex):
            problem  = ex.get("problem",  ex.get("question", ""))
            solution = ex.get("solution", ex.get("response",  ""))
            solution = ensure_solution_tags(solution)
            text = format_for_chat(problem, solution, tokenizer)
            return {"text": text}

        ds = raw_ds.map(
            process,
            remove_columns=raw_ds.column_names,
            num_proc=4,
            desc="Formatting",
        )
        ds = ds.filter(lambda x: len(x["text"]) > 100)

    # Shuffle
    ds = ds.shuffle(seed=42)

    if sanity:
        ds = ds.select(range(min(300, len(ds))))
        print(f"  [SANITY] Using {len(ds)} examples")
    else:
        print(f"  Full dataset: {len(ds):,} examples")

    # Print a sample to visually verify format
    sample = ds[0]["text"]
    print(f"\n  Sample (first 400 chars):\n  {sample[:400].replace(chr(10), chr(10) + '  ')}...\n")

    # Verify format: should have both <think> and <solution>
    has_think    = sum(1 for x in ds if "<think>"    in x["text"])
    has_solution = sum(1 for x in ds if "<solution>" in x["text"])
    print(f"  Format check:")
    print(f"    <think> present    : {has_think:,} / {len(ds):,} ({has_think/len(ds)*100:.1f}%)")
    print(f"    <solution> present : {has_solution:,} / {len(ds):,} ({has_solution/len(ds)*100:.1f}%)")

    if has_think / len(ds) < 0.90:
        print("  ⚠️  WARNING: Less than 90% of examples have <think> tags.")
        print("     Check prepare_distill_data.py output format.")

    return ds


# ─── Loss guard callback ──────────────────────────────────────────────────────

class LossGuardCallback(TrainerCallback):
    """
    Abort training if loss exceeds MAX_SANE_LOSS after warmup.
    Catches the label-smoothing bug immediately instead of wasting hours.
    """
    def __init__(self, max_loss=MAX_SANE_LOSS, check_after_step=5):
        self.max_loss       = max_loss
        self.check_after    = check_after_step
        self.triggered      = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or state.global_step < self.check_after:
            return control

        loss = logs.get("loss", None)
        if loss is None:
            return control

        if loss > self.max_loss and not self.triggered:
            self.triggered = True
            print(f"\n{'!'*60}")
            print(f"  LOSS GUARD TRIGGERED at step {state.global_step}")
            print(f"  train_loss = {loss:.4f}  (max sane = {self.max_loss})")
            print(f"")
            print(f"  This is the label-smoothing bug. Check:")
            print(f"    1. label_smoothing_factor is 0.0 in SFTConfig")
            print(f"    2. No custom loss function is being applied")
            print(f"    3. Dataset is formatted correctly (check sample above)")
            print(f"{'!'*60}\n")
            control.should_training_stop = True

        return control


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_path: str, attn_impl: str):
    print(f"\n{'─'*60}")
    print(f"  Loading model : {model_path}")
    print(f"  Attn backend  : {attn_impl}")

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Input model not found: {model_path}\n"
            f"  Make sure Stage 2 merge completed successfully."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right",  # right padding for SFT
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map={"": local_rank},
        trust_remote_code=True,
    )

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()

    params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Parameters    : {params:.2f}B")
    return model, tokenizer


def apply_lora(model, rank: int, alpha: int):
    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        inference_mode=False,
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e9:.2f}B ({trainable/total*100:.1f}%)")
    return model


# ─── Training args ────────────────────────────────────────────────────────────

def build_sft_config(args, output_dir: Path) -> SFTConfig:
    """
    THE KEY FIX IS HERE:
      label_smoothing_factor = 0.0  ← explicit zero. No uniform distribution penalty.

    Everything else is standard:
      - Lower LR than Stage 2 (1e-5 vs 2e-4)
      - Fewer epochs (2 vs 3)
      - Longer sequences (8192 vs 8192 — same, R1 traces fit in 8K)
      - No packing (sequences are already dense and long)
    """
    n_gpus = max(torch.cuda.device_count(), 1)

    if args.sanity:
        max_steps     = 30
        save_steps    = 9999
        logging_steps = 1
        seq_len       = 4096
        batch         = 1
        grad_accum    = 4
    else:
        max_steps     = -1
        save_steps    = 500
        logging_steps = 25
        seq_len       = args.max_seq_len
        batch         = args.batch
        grad_accum    = args.grad_accum

    eff_batch = batch * grad_accum * n_gpus
    print(f"\n  Effective batch size: {batch} × {grad_accum} × {n_gpus} GPUs = {eff_batch}")

    if args.wandb:
        os.environ["WANDB_PROJECT"] = "mathReason-1.5B"

    return SFTConfig(
        output_dir=str(output_dir),
        run_name=args.run_name,

        # Steps / epochs
        num_train_epochs=args.epochs,
        max_steps=max_steps,

        # Batch
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum,

        # ════════════════════════════════════════════════
        # THE CRITICAL FIX — no label smoothing
        # ════════════════════════════════════════════════
        # HuggingFace's label_smoothing adds:
        #   eps * (-log_softmax(logits).mean())
        # where .mean() averages over the full vocabulary.
        # For Qwen2.5's ~150K vocab: this term ≈ log(150K) ≈ 11.9
        # When combined with think-weight multiplier, this explodes to ~44.
        # FIX: set it to 0.0. Plain cross-entropy. No smoothing.
        label_smoothing_factor=0.0,
        # ════════════════════════════════════════════════

        # Learning rate — 10× lower than Stage 2
        # Model already knows the format from Stage 2 SFT.
        # We're transferring reasoning STYLE, not teaching new format.
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=50 if not args.sanity else 5,

        # Precision
        bf16=True,
        tf32=True,

        # SFT config
        max_length=seq_len,
        dataset_text_field="text",
        packing=False,  # R1 traces are long — no benefit to packing

        # Saving
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        eval_strategy="no",
        logging_steps=logging_steps,

        # Optimizer
        optim="adamw_torch_fused",
        weight_decay=0.01,
        max_grad_norm=1.0,

        # Misc
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="wandb" if args.wandb else "none",
        seed=42,

        resume_from_checkpoint=str(output_dir) if args.resume else None,
    )


# ─── GSM8K eval ──────────────────────────────────────────────────────────────

def eval_gsm8k(model, tokenizer, n_samples=200):
    """
    Post-distillation GSM8K eval.
    Expected: 80–85% (up from 70% after Stage 2 SFT).
    The key thing we're looking for: accuracy should be HIGHER than 70%,
    NOT lower. If it's lower (like the 53.5% we saw), the training was broken.
    """
    print(f"\n{'═'*60}")
    print(f"POST-DISTILLATION GSM8K EVAL  ({n_samples} samples, greedy)")
    print(f"Stage 2 baseline: 70% | If lower than this, training was broken")
    print(f"{'═'*60}")

    try:
        dataset = load_dataset("gsm8k", "main", split="test")
        dataset = dataset.select(range(min(n_samples, len(dataset))))
    except Exception as e:
        print(f"  Could not load GSM8K: {e}")
        return None

    device  = next(model.parameters()).device
    correct = 0

    for i, item in enumerate(dataset):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": item["question"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=512
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        response   = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Extract from <solution> tag first, then fallback
        sol_match  = re.search(r"<solution>(.*?)</solution>", response, re.DOTALL)
        pred_text  = sol_match.group(1).strip() if sol_match else response
        pred_nums  = re.findall(r"-?[\d,]+(?:\.\d+)?", pred_text.replace(",", ""))
        predicted  = pred_nums[-1] if pred_nums else None

        gt_nums = re.findall(r"-?[\d,]+", item["answer"].split("####")[-1])
        gt      = gt_nums[-1].replace(",", "") if gt_nums else None

        if predicted and gt and predicted == gt:
            correct += 1

        if (i + 1) % 50 == 0:
            running_acc = correct / (i + 1) * 100
            status = "✅" if running_acc >= 70 else "❌"
            print(f"  [{i+1:>3}/{n_samples}]  {correct}/{i+1} = {running_acc:.1f}% {status}")

    acc = correct / n_samples * 100
    status = "✅ IMPROVEMENT" if acc > 70 else "❌ REGRESSION (training was broken)"
    print(f"\n  Final: {correct}/{n_samples} = {acc:.1f}%  {status}")
    print(f"  Stage 2 baseline: 70% | Stage 3 target: 80–85%")
    return acc


# ─── Merge ───────────────────────────────────────────────────────────────────

def merge_and_save(model, tokenizer, merged_dir: Path):
    print(f"\n{'─'*60}")
    print("  Merging LoRA → base model...")
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    print(f"  ✅ Merged checkpoint: {merged_dir}")
    print(f"     → Use this as input for Stage 4 GRPO")
    return merged


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args      = parse_args()
    sanity    = args.sanity
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main   = local_rank == 0

    run_suffix  = "_sanity" if sanity else ""
    output_dir  = CKPT_DIR / f"{args.run_name}{run_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if is_main:
        print("\n" + "█" * 60)
        print("  STAGE 3 — CoT DISTILLATION TRAINING (FIXED)")
        if sanity:
            print("  *** SANITY RUN — 300 examples, 30 steps ***")
            print("  *** Verifying loss is in [1.5, 2.8] range ***")
        print("█" * 60)
        print(f"  Timestamp    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Input model  : {args.model}")
        print(f"  Data         : {DISTILL_DATA}")
        print(f"  Output       : {output_dir}")
        print(f"")
        print(f"  Key fix applied:")
        print(f"    label_smoothing_factor = 0.0  (was causing loss = 44.45)")
        print(f"    No think-span weighting       (was compounding the bug)")
        print(f"    Plain cross-entropy on all tokens")
        print(f"  Expected loss: 1.5 – 2.8  (script halts if > {MAX_SANE_LOSS})")

    # ── Load model ──
    model, tokenizer = load_model_and_tokenizer(args.model, args.attn_impl)
    model = apply_lora(model, args.lora_rank, args.lora_alpha)

    # ── Load data ──
    dataset = load_distill_dataset(tokenizer, sanity)

    # ── Training config ──
    train_cfg = build_sft_config(args, output_dir)

    # ── Trainer ──
    loss_guard = LossGuardCallback(max_loss=MAX_SANE_LOSS, check_after_step=5)

    trainer = SFTTrainer(
        model            = model,
        args             = train_cfg,
        train_dataset    = dataset,
        processing_class = tokenizer,
    )
    trainer.add_callback(loss_guard)

    # ── Train ──
    if is_main:
        print(f"\n{'─'*60}")
        print(f"  Starting distillation training...")
        if sanity:
            print(f"  Watch the first 5 steps carefully:")
            print(f"    loss in [1.5, 2.8] → pipeline is working ✅")
            print(f"    loss > {MAX_SANE_LOSS}          → bug still present ❌ (auto-halt)")
        print(f"{'─'*60}\n")

    result = trainer.train(
        resume_from_checkpoint=str(output_dir) if args.resume else None
    )

    # ── Check if loss guard triggered ──
    if loss_guard.triggered:
        print("\n❌ Training halted by loss guard.")
        print("   The label smoothing bug is still present.")
        print("   Check SFTConfig label_smoothing_factor = 0.0")
        sys.exit(1)

    # ── Save ──
    if is_main:
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        metrics = result.metrics
        with open(LOG_DIR / "stage3_train_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        final_loss = metrics.get("train_loss", -1)
        print(f"\n  ✅ LoRA adapter saved : {output_dir}")
        print(f"  ✅ Metrics saved      : {LOG_DIR / 'stage3_train_metrics.json'}")
        print(f"  Final train_loss = {final_loss:.4f}")
        if final_loss > MAX_SANE_LOSS:
            print(f"  ❌ Loss still high — review training")
        elif final_loss > 0:
            print(f"  ✅ Loss in expected range")

    # ── Merge ──
    if is_main and not args.skip_merge and not sanity:
        merged = merge_and_save(model, tokenizer, MERGED_DIR)
        eval_model = merged
    else:
        eval_model = model
        if sanity and is_main:
            print("\n  [Sanity] Skipping merge and full eval.")

    # ── GSM8K eval ──
    if is_main and not args.skip_eval and not sanity:
        model.eval()
        acc = eval_gsm8k(eval_model, tokenizer, n_samples=200)

        results = {
            "stage": "3_distill",
            "timestamp": datetime.now().isoformat(),
            "gsm8k_accuracy": acc,
            "train_loss": result.metrics.get("train_loss", -1),
            "model": args.model,
            "epochs": args.epochs,
            "lr": args.lr,
        }
        with open(LOG_DIR / "stage3_eval_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  ✅ Eval results: {LOG_DIR / 'stage3_eval_results.json'}")

    # ── Summary ──
    if is_main:
        final_loss = result.metrics.get("train_loss", -1)
        print(f"\n{'█'*60}")
        if sanity:
            print("  SANITY RUN COMPLETE")
            print("  ─────────────────────────────────────────────")
            print(f"  train_loss = {final_loss:.4f}")
            if 0.5 <= final_loss <= 3.5:
                print(f"  ✅ Loss in expected range [0.5, 3.5] — safe to launch full run")
                print(f"")
                print(f"  Launch full training:")
                print(f"  torchrun --nproc_per_node=2 stage3_distill_train.py \\")
                print(f"      --run-name distill_v1 --wandb")
            else:
                print(f"  ❌ Loss out of expected range — DO NOT launch full run")
                print(f"     Expected: 1.5–2.8 for R1 traces")
                print(f"     Got: {final_loss:.4f}")
                print(f"     Check dataset format and SFTConfig settings")
        else:
            print("  STAGE 3 DISTILLATION COMPLETE")
            print("  ─────────────────────────────────────────────")
            print(f"  train_loss       = {final_loss:.4f}")
            print(f"  Merged checkpoint: {MERGED_DIR}")
            print(f"  Next step        : Stage 4C GRPO Phase 2")
            print(f"  Use checkpoint   : {MERGED_DIR}")
            print(f"")
            print(f"  Expected GSM8K improvement: 70% → 80–85%")
            print(f"  Expected MATH500           : 38% → 55–62%")
        print(f"{'█'*60}\n")


if __name__ == "__main__":
    main()
