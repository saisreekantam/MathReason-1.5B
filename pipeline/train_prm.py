"""
train_prm.py  —  Stage 8B: Process Reward Model Training
══════════════════════════════════════════════════════════════════════════════
Server  : revanth@172.16.192.168
GPU     : 0  (48GB — PRM is 0.5B, fits easily)
Base    : Qwen2.5-0.5B  (same family as student — consistent embeddings)
Input   : ~/nlp/data/prm/mc_step_labels.jsonl
Output  : ~/nlp/checkpoints/prm_merged/

Architecture:
  Qwen2.5-0.5B  +  linear scalar head on [EOS] token
  Input  = "<problem>\n\n<solution_prefix_up_to_step_t>"
  Output = scalar ∈ [0, 1]  (is this step on the right track?)
  Loss   = Binary cross-entropy on step-level labels

Training:
  LoRA rank 16 (fast, ~1 hour on 48GB GPU)
  LR = 2e-4, batch=8, 3 epochs
  Saves best checkpoint by val loss

Usage:
  tmux new -s train_prm
  conda activate nlp
  CUDA_VISIBLE_DEVICES=0 python train_prm.py

  # Resume from checkpoint:
  CUDA_VISIBLE_DEVICES=0 python train_prm.py --resume

  # Sanity (100 samples, 5 steps):
  CUDA_VISIBLE_DEVICES=0 python train_prm.py --sanity
══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

# ─── Paths ────────────────────────────────────────────────────────────────────

WORK_DIR   = Path("~/nlp").expanduser()
PRM_BASE   = "Qwen/Qwen2.5-0.5B"               # HuggingFace id
DATA_FILE  = WORK_DIR / "data" / "prm" / "mc_step_labels.jsonl"
CKPT_DIR   = WORK_DIR / "checkpoints" / "prm"
MERGED_DIR = WORK_DIR / "checkpoints" / "prm_merged"
LOG_FILE   = WORK_DIR / "logs" / "prm_train.log"

MAX_SEQ_LEN = 1024   # problem + prefix; truncate from left if longer

# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Process Reward Model")
    p.add_argument("--sanity",      action="store_true",
                   help="100 samples, 5 steps — quick test")
    p.add_argument("--resume",      action="store_true",
                   help="Resume from latest checkpoint in CKPT_DIR")
    p.add_argument("--base-model",  type=str, default=PRM_BASE)
    p.add_argument("--data",        type=str, default=str(DATA_FILE))
    p.add_argument("--epochs",      type=int, default=3)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--batch-size",  type=int, default=8)
    p.add_argument("--lora-rank",   type=int, default=16)
    p.add_argument("--val-frac",    type=float, default=0.1,
                   help="Fraction of data to use for validation")
    p.add_argument("--save-steps",  type=int, default=100)
    return p.parse_args()

# ─── Dataset ──────────────────────────────────────────────────────────────────

class PRMDataset(Dataset):
    """
    Each sample: (problem, step_prefix) → label (0 or 1).
    Input text = problem + "\n\n" + step_prefix.
    Tokenized and left-truncated to MAX_SEQ_LEN.
    """

    def __init__(self, records: List[dict], tokenizer, max_len: int = MAX_SEQ_LEN):
        self.records   = records
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec    = self.records[idx]
        text   = rec["problem"].strip() + "\n\n" + rec["prefix"].strip()
        label  = float(rec["label"])

        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.float32),
        }

def collate_fn(batch):
    """Left-pad sequences in a batch to the same length."""
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids      = []
    attention_masks = []
    labels          = []

    for b in batch:
        pad_len = max_len - b["input_ids"].shape[0]
        # Left-pad (prepend pad tokens)
        input_ids.append(
            torch.cat([torch.zeros(pad_len, dtype=torch.long), b["input_ids"]])
        )
        attention_masks.append(
            torch.cat([torch.zeros(pad_len, dtype=torch.long), b["attention_mask"]])
        )
        labels.append(b["label"])

    return {
        "input_ids":      torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels":         torch.stack(labels),
    }

# ─── PRM model wrapper ────────────────────────────────────────────────────────

class ProcessRewardModel(nn.Module):
    """
    Qwen2.5-0.5B (LoRA) + scalar head on the last non-padding token.
    Outputs a single sigmoid score ∈ [0, 1] per sequence.
    """

    def __init__(self, base_model, hidden_size: int):
        super().__init__()
        self.base  = base_model
        self.head  = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        # Initialize head weights small
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Use hidden state of the last real (non-padding) token
        hidden = outputs.hidden_states[-1]  # (B, T, H)

        # Find last non-pad position for each sample
        seq_lens = attention_mask.sum(dim=1) - 1  # (B,)
        last_hidden = hidden[
            torch.arange(hidden.shape[0], device=hidden.device),
            seq_lens,
        ]  # (B, H)

        score = self.head(last_hidden.float()).squeeze(-1)  # (B,) — cast bf16→fp32 for head
        return score

# ─── Training loop ────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0.0
    criterion  = nn.BCELoss()

    for step, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        scores = model(input_ids, attention_mask)
        loss   = criterion(scores, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if (step + 1) % 20 == 0:
            avg = total_loss / (step + 1)
            print(f"    Epoch {epoch}  step {step+1}/{len(loader)}  "
                  f"loss={avg:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

    return total_loss / len(loader)

@torch.inference_mode()
def validate(model, loader, device):
    model.eval()
    criterion  = nn.BCELoss()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        scores  = model(input_ids, attention_mask)
        loss    = criterion(scores, labels)
        total_loss += loss.item()

        preds   = (scores >= 0.5).float()
        correct += (preds == labels).sum().item()
        total   += labels.shape[0]

    return total_loss / len(loader), correct / total

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.sanity:
        args.epochs    = 2
        args.save_steps = 5
        print("*** SANITY MODE: 100 samples, 2 epochs ***\n")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")

    # ── Load tokenizer ──
    print(f"\n  Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load data ──
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            f"Run generate_mc_step_labels.py first."
        )
    print(f"\n  Loading data from {data_path} ...")
    records = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if args.sanity:
        records = records[:100]

    print(f"  Total records  : {len(records)}")
    print(f"  Label=1 (good) : {sum(r['label'] for r in records)}")
    print(f"  Label=0 (bad)  : {sum(1-r['label'] for r in records)}")

    # ── Split train/val ──
    random.seed(42)
    random.shuffle(records)
    n_val    = max(1, int(len(records) * args.val_frac))
    val_rec  = records[:n_val]
    train_rec = records[n_val:]
    print(f"  Train: {len(train_rec)}  |  Val: {len(val_rec)}")

    train_ds = PRMDataset(train_rec, tokenizer)
    val_ds   = PRMDataset(val_rec,   tokenizer)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  collate_fn=collate_fn, num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, collate_fn=collate_fn, num_workers=2)

    # ── Load base model ──
    print(f"\n  Loading base model: {args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    hidden_size = base.config.hidden_size

    # ── Apply LoRA ──
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )
    base = get_peft_model(base, lora_cfg)
    base.print_trainable_parameters()

    # ── Wrap in PRM ──
    model = ProcessRewardModel(base, hidden_size).to(device)
    # Convert scalar head to float32 for stability
    model.head = model.head.float()

    # ── Optimizer & scheduler ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )
    total_steps  = len(train_dl) * args.epochs
    warmup_steps = max(10, total_steps // 10)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Resume ──
    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume:
        ckpts = sorted(CKPT_DIR.glob("epoch_*.pt"))
        if ckpts:
            latest = ckpts[-1]
            print(f"\n  Resuming from {latest}")
            state = torch.load(latest, map_location=device)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start_epoch   = state["epoch"] + 1
            best_val_loss = state.get("best_val_loss", float("inf"))
            print(f"  Resuming from epoch {start_epoch}")

    print(f"\n  Training PRM  |  epochs={args.epochs}  lr={args.lr}  "
          f"batch={args.batch_size}  LoRA_r={args.lora_rank}")
    print(f"  Total steps: {total_steps}  warmup: {warmup_steps}\n")

    log_entries = []

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_epoch(
            model, train_dl, optimizer, scheduler, device, epoch
        )
        val_loss, val_acc = validate(model, val_dl, device)

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_loss, 4),
            "val_acc":    round(val_acc, 4),
        }
        log_entries.append(log_entry)
        with open(LOG_FILE, "w") as f:
            json.dump(log_entries, f, indent=2)

        print(
            f"\n  ── Epoch {epoch}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.3f} ──\n"
        )

        # Save checkpoint
        ckpt_path = CKPT_DIR / f"epoch_{epoch:02d}.pt"
        torch.save({
            "epoch":         epoch,
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "scheduler":     scheduler.state_dict(),
            "val_loss":      val_loss,
            "val_acc":       val_acc,
            "best_val_loss": best_val_loss,
        }, ckpt_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = CKPT_DIR / "best.pt"
            torch.save(model.state_dict(), best_path)
            print(f"  ★ New best val_loss={val_loss:.4f} → saved to {best_path}")

    # ── Merge LoRA and save final PRM ──
    print("\n  Merging LoRA weights...")
    best_state = torch.load(CKPT_DIR / "best.pt", map_location=device)
    model.load_state_dict(best_state)

    merged_base = model.base.merge_and_unload()
    merged_base.save_pretrained(str(MERGED_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(MERGED_DIR))

    # Save scalar head weights separately
    torch.save(model.head.state_dict(), MERGED_DIR / "prm_head.pt")

    # Save config so eval script can reconstruct
    prm_config = {
        "base_model":  args.base_model,
        "hidden_size": hidden_size,
        "head_path":   "prm_head.pt",
        "best_val_loss": best_val_loss,
        "val_acc":      log_entries[-1]["val_acc"],
    }
    with open(MERGED_DIR / "prm_config.json", "w") as f:
        json.dump(prm_config, f, indent=2)

    print(f"\n  ✅ PRM training complete!")
    print(f"     Best val_loss : {best_val_loss:.4f}")
    print(f"     Merged model  : {MERGED_DIR}")
    print(f"\n  Next: run  eval_prm_best_of_n.py  to test PRM on MATH500.")

if __name__ == "__main__":
    main()
