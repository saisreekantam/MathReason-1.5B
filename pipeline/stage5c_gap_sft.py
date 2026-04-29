# ══════════════════════════════════════════════════════════════════
# stage5c_gap_sft.py
# Gap-fill SFT — Step 3: LoRA fine-tuning on gap dataset
#
# Input   : GDPO best checkpoint (fullgrpo_v3 or v1 merged)
# Output  : ~/nlp/checkpoints/stage5_gap_sft/
# Merged  : ~/nlp/checkpoints/stage5_gap_sft_merged/
#
# Key constraints vs Stage 3:
#   - LR 5e-6 (10× lower than Stage 3's 5e-5) — surgical patch
#   - LoRA r32/a64 (not r64) — smaller subspace = less forgetting
#   - 2 epochs max
#   - assistant_only_loss=True — loss only on <think>+<solution>
#   - label_smoothing_factor=0.0 — mandatory (Stage 3 lesson)
#   - max_seq_length=1024 — short traces, no long loops
#   - No packing — each example is standalone
#
# Usage:
#   python stage5c_gap_sft.py --sanity     # 50 examples, 10 steps
#   python stage5c_gap_sft.py              # full run
#   python stage5c_gap_sft.py --merge      # merge LoRA after training
#
# CUDA: CUDA_VISIBLE_DEVICES=1 python stage5c_gap_sft.py
# ══════════════════════════════════════════════════════════════════
import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from datasets import Dataset, concatenate_datasets
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from trl import SFTConfig, SFTTrainer

# ─── Paths ────────────────────────────────────────────────────────
WORK_DIR   = Path("~/nlp").expanduser()
DATA_DIR   = WORK_DIR / "data" / "gap_sft"
LOG_DIR    = WORK_DIR / "logs"

CKPT_OUT   = WORK_DIR / "checkpoints" / "stage5_gap_sft"
MERGED_OUT = WORK_DIR / "checkpoints" / "stage5_gap_sft_merged"

# Auto-detect best GDPO input checkpoint
def find_input_checkpoint() -> Path:
    candidates = [
        WORK_DIR / "checkpoints" / "fullgrpo_v3_merged",
        WORK_DIR / "checkpoints" / "fullgrpo_v1_merged",
        WORK_DIR / "checkpoints" / "stage4d_gdpo_merged",
        WORK_DIR / "checkpoints" / "stage4c_fullgrpo_merged",
        WORK_DIR / "checkpoints" / "stage3_distilled_merged",
    ]
    # Also try latest unmerged checkpoint inside fullgrpo dirs
    for name in ["fullgrpo_v3", "fullgrpo_v1", "stage4d_gdpo", "stage4c_fullgrpo"]:
        d = WORK_DIR / "checkpoints" / name
        if d.exists():
            ckpts = sorted(d.glob("checkpoint-*"),
                           key=lambda p: int(p.name.split("-")[-1]))
            if ckpts:
                candidates.insert(0, ckpts[-1])
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("No input checkpoint found. Checked:\n" +
                            "\n".join(str(c) for c in candidates))

# ─── CLI ─────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Stage 5 Gap-Fill SFT")
    p.add_argument("--sanity",    action="store_true",
                   help="Quick sanity: 50 examples, 10 steps")
    p.add_argument("--merge",     action="store_true",
                   help="Only merge existing LoRA — skip training")
    p.add_argument("--model",     type=str, default="auto")
    p.add_argument("--epochs",    type=int, default=2)
    p.add_argument("--lr",        type=float, default=5e-6)
    p.add_argument("--lora-r",    type=int,   default=32)
    p.add_argument("--lora-alpha",type=int,   default=64)
    p.add_argument("--batch",     type=int,   default=2)
    p.add_argument("--grad-acc",  type=int,   default=8)
    p.add_argument("--max-seq",   type=int,   default=1024)
    p.add_argument("--wandb",     action="store_true")
    return p.parse_args()

# ─── Dataset loading ─────────────────────────────────────────────
def load_gap_dataset(sanity: bool = False) -> Dataset:
    """Load and combine all gap categories."""
    records = []
    category_counts = defaultdict(int)

    # D/E/F: R1 traces (already formatted)
    def_file = DATA_DIR / "cat_DEF_r1_traces.jsonl"
    if def_file.exists():
        with open(def_file) as f:
            for line in f:
                r = json.loads(line.strip())
                records.append(r)
                category_counts[r.get("category", "DEF")] += 1
    else:
        print(f"  WARNING: {def_file} not found — run stage5a first")

    # A/B/C: QwQ traces
    abc_file = DATA_DIR / "cat_ABC_traced.jsonl"
    if abc_file.exists():
        with open(abc_file) as f:
            for line in f:
                r = json.loads(line.strip())
                records.append(r)
                category_counts[r.get("category", "ABC")] += 1
    else:
        print(f"  WARNING: {abc_file} not found — run stage5b first")

    if not records:
        raise FileNotFoundError(
            "No training data found. "
            "Run stage5a_build_dataset.py and stage5b_generate_traces.py first."
        )

    random.shuffle(records)

    print(f"\n  Total training examples: {len(records)}")
    for cat, n in sorted(category_counts.items()):
        print(f"    {cat:<28}: {n}")

    if sanity:
        records = records[:50]
        print(f"\n  SANITY MODE: using {len(records)} examples")

    # Convert to HuggingFace Dataset
    # SFTTrainer with assistant_only_loss expects "messages" field
    dataset = Dataset.from_list([
        {"messages": r["messages"], "category": r.get("category", "unknown")}
        for r in records
    ])
    return dataset

# ─── Monitoring callback ─────────────────────────────────────────
class GapSFTMonitor(TrainerCallback):
    def __init__(self, log_path: Path, check_every: int = 25):
        self.log_path    = log_path
        self.check_every = check_every
        self.history     = []
        self.best_loss   = float("inf")

    def on_log(self, args, state: TrainerState, control: TrainerControl,
               logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        if "loss" not in logs:
            return

        train_loss = logs.get("loss", None)
        lr         = logs.get("learning_rate", None)

        entry = {"step": step, "loss": train_loss, "lr": lr}
        self.history.append(entry)

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump(self.history, f, indent=2)

        if step % self.check_every == 0:
            print(f"\n  ── Step {step} ──")
            print(f"    Loss  : {train_loss:.4f}" if train_loss else "    Loss: N/A")
            print(f"    LR    : {lr:.2e}"         if lr         else "")

            if train_loss and train_loss < self.best_loss:
                self.best_loss = train_loss
                print(f"    ⭐ New best loss: {self.best_loss:.4f}")

            # Health check: loss should drop steadily
            if step >= 100 and train_loss and train_loss > 2.0:
                print(f"\n    ⚠️  Loss still high at step {step}: {train_loss:.4f}")
                print(f"       Check: label_smoothing_factor=0.0, LR not too high")

            if step >= 50 and train_loss and train_loss < 0.05:
                print(f"\n    ⚠️  Loss very low at step {step}: {train_loss:.4f}")
                print(f"       Possible overfitting — consider early stopping")

    def on_train_end(self, args, state, control, **kwargs):
        print(f"\n{'═'*55}")
        print(f"  STAGE 5 GAP-SFT TRAINING COMPLETE")
        print(f"  Best loss : {self.best_loss:.4f}")
        print(f"  Steps     : {state.global_step}")
        print(f"{'═'*55}")

# ─── LoRA merge utility ───────────────────────────────────────────
def merge_lora(base_model_path: Path, lora_path: Path, merged_path: Path):
    print(f"\nMerging LoRA …")
    print(f"  Base   : {base_model_path}")
    print(f"  LoRA   : {lora_path}")
    print(f"  Output : {merged_path}")

    tok = AutoTokenizer.from_pretrained(
        str(base_model_path), trust_remote_code=True
    )
    base = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    peft_model = PeftModel.from_pretrained(base, str(lora_path))
    merged     = peft_model.merge_and_unload()

    merged_path.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_path))
    tok.save_pretrained(str(merged_path))
    print(f"  ✅ Merged checkpoint saved to {merged_path}")

# ─── Main ─────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Resolve input checkpoint
    if args.model == "auto":
        input_ckpt = find_input_checkpoint()
    else:
        input_ckpt = Path(args.model)
    print(f"\n{'─'*60}")
    print(f"  Stage 5 Gap-Fill SFT")
    print(f"  Input checkpoint : {input_ckpt}")
    print(f"  LR               : {args.lr}")
    print(f"  LoRA r/alpha     : {args.lora_r}/{args.lora_alpha}")
    print(f"  Epochs           : {args.epochs}")
    print(f"  Sanity           : {args.sanity}")
    print(f"{'─'*60}\n")

    # Merge-only mode
    if args.merge:
        # Find latest LoRA checkpoint
        latest = sorted(CKPT_OUT.glob("checkpoint-*"),
                        key=lambda p: int(p.name.split("-")[-1]))
        if not latest:
            raise FileNotFoundError(f"No checkpoints in {CKPT_OUT}")
        merge_lora(input_ckpt, latest[-1], MERGED_OUT)
        return

    # ── Load dataset ───────────────────────────────────────────────
    dataset = load_gap_dataset(sanity=args.sanity)
    print(f"\n  Dataset size: {len(dataset)}")

    # 95/5 train/eval split
    splits     = dataset.train_test_split(test_size=0.05, seed=42)
    train_data = splits["train"]
    eval_data  = splits["test"]
    print(f"  Train: {len(train_data)} | Eval: {len(eval_data)}")

    # ── Load model ─────────────────────────────────────────────────
    print(f"\nLoading model …")
    tok = AutoTokenizer.from_pretrained(
        str(input_ckpt), trust_remote_code=True
    )
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(input_ckpt),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # ── LoRA config ────────────────────────────────────────────────
    # r32 (not r64) — surgical update, not full distillation
    # Target: q, k, v projections + output + gate (SwiGLU)
    # Do NOT target mlp.up_proj alone — need gate for SwiGLU updates
    lora_cfg = LoraConfig(
        task_type     = TaskType.CAUSAL_LM,
        r             = args.lora_r,
        lora_alpha    = args.lora_alpha,
        lora_dropout  = 0.05,
        bias          = "none",
        target_modules= [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        # Modules to NOT update — keep GDPO-learned weights intact
        modules_to_save=None,
    )

    # ── SFTConfig ──────────────────────────────────────────────────
    # TRL 0.29 API — key parameter names:
    #   eval_strategy (not evaluation_strategy)
    #   processing_class passed to SFTTrainer (not tokenizer=)
    #   assistant_only_loss=True → loss only on assistant tokens
    #   label_smoothing_factor=0.0 → MANDATORY (Stage 3 lesson)

    report_to = "wandb" if args.wandb else "none"
    steps_per_epoch = len(train_data) // (args.batch * args.grad_acc)
    save_steps      = max(50, steps_per_epoch // 2)
    eval_steps      = max(50, steps_per_epoch // 2)
    total_steps     = steps_per_epoch * args.epochs

    if args.sanity:
        save_steps = eval_steps = 5
        total_steps = 10

    sft_config = SFTConfig(
        # Paths
        output_dir            = str(CKPT_OUT),
        run_name              = "stage5_gap_sft",

        # Training duration
        num_train_epochs      = args.epochs,
        max_steps             = total_steps if not args.sanity else 10,

        # Batch
        per_device_train_batch_size = args.batch,
        per_device_eval_batch_size  = args.batch,
        gradient_accumulation_steps = args.grad_acc,

        # LR — 10× lower than Stage 3: surgical patch, not retraining
        learning_rate         = args.lr,
        lr_scheduler_type     = "cosine",
        warmup_ratio          = 0.05,

        # Precision
        bf16                  = True,
        fp16                  = False,

        # Regularization
        weight_decay          = 0.01,
        max_grad_norm         = 1.0,

        # CRITICAL: must be 0.0 (Stage 3 bug lesson)
        label_smoothing_factor = 0.0,

        # Loss only on assistant responses (think+solution block)
        # TRL 0.29: assistant_only_loss works with conversational format
        assistant_only_loss   = True,

        # Sequence
        max_seq_length        = args.max_seq,
        packing               = False,  # No packing — standalone examples

        # Eval & save
        eval_strategy         = "steps",  # TRL 0.29: eval_strategy (not evaluation_strategy)
        eval_steps            = eval_steps,
        save_strategy         = "steps",
        save_steps            = save_steps,
        save_total_limit      = 3,
        load_best_model_at_end= True,
        metric_for_best_model = "eval_loss",

        # Gradient checkpointing (saves ~30% VRAM)
        gradient_checkpointing= True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},

        # Logging
        logging_steps         = 10,
        report_to             = report_to,

        # Dataset
        dataset_num_proc      = 4,
        remove_unused_columns = False,
    )

    # ── Trainer ────────────────────────────────────────────────────
    # TRL 0.29: use processing_class= (not tokenizer=)
    monitor = GapSFTMonitor(
        log_path    = LOG_DIR / "stage5_gap_sft.json",
        check_every = 25,
    )

    trainer = SFTTrainer(
        model             = model,
        args              = sft_config,
        train_dataset     = train_data,
        eval_dataset      = eval_data,
        peft_config       = lora_cfg,
        processing_class  = tok,      # TRL 0.29 API
        callbacks         = [monitor],
    )

    # ── Sanity check: verify labels are masked correctly ───────────
    print("\nVerifying label masking (assistant_only_loss) …")
    sample = train_data[0]
    tokenized = trainer.data_collator([trainer.train_dataset[0]])
    labels = tokenized["labels"][0]
    n_masked  = (labels == -100).sum().item()
    n_trained = (labels != -100).sum().item()
    total_tok = len(labels)
    print(f"  Total tokens : {total_tok}")
    print(f"  Masked (-100): {n_masked} ({n_masked/total_tok*100:.0f}%)")
    print(f"  Trained on   : {n_trained} ({n_trained/total_tok*100:.0f}%)")
    if n_trained < 10:
        raise ValueError(
            "Only {n_trained} tokens are trained on — check assistant_only_loss config"
        )
    print("  ✅ Label masking looks correct\n")

    # ── Train ──────────────────────────────────────────────────────
    print(f"Starting training …")
    print(f"  Total steps  : {total_steps}")
    print(f"  Save every   : {save_steps} steps")
    print(f"  Eval every   : {eval_steps} steps\n")

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\n  Training time: {elapsed/3600:.1f}h")

    # ── Save final ─────────────────────────────────────────────────
    trainer.save_model(str(CKPT_OUT / "final"))
    tok.save_pretrained(str(CKPT_OUT / "final"))
    print(f"\n  LoRA adapter saved to {CKPT_OUT / 'final'}")

    # ── Auto-merge if not sanity ───────────────────────────────────
    if not args.sanity:
        print("\nAuto-merging LoRA weights …")
        merge_lora(input_ckpt, CKPT_OUT / "final", MERGED_OUT)

        print(f"\n{'═'*60}")
        print(f"  STAGE 5 COMPLETE")
        print(f"  Merged checkpoint: {MERGED_OUT}")
        print(f"\n  NEXT STEPS:")
        print(f"  1. Eval on 20 GSM8K problems:")
        print(f"     CUDA_VISIBLE_DEVICES=1 python3 - << 'EOF'")
        print(f"     # (use the eval script from earlier)")
        print(f"  2. If pass@1 > 60% and tag rate > 0% → run DPO")
        print(f"  3. If operator errors persist → add more Cat A examples")
        print(f"{'═'*60}")
    else:
        print("\n  SANITY RUN COMPLETE — no merge in sanity mode")
        print("  Losses look healthy? Run full training:")
        print("  CUDA_VISIBLE_DEVICES=1 python stage5c_gap_sft.py")

if __name__ == "__main__":
    main()
