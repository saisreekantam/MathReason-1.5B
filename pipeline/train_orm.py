"""
ORM Training — Outcome Reward Model
════════════════════════════════════
Server : revanth@172.16.192.168
Path   : ~/nlp/scripts/train_orm.py

Trains a 0.5B pairwise ranking ORM on top of Qwen2.5-0.5B.
Uses existing dpo_pairs_raw.jsonl — no new data generation needed.

Architecture:
  Qwen2.5-0.5B backbone (frozen except LoRA)
  + linear reward head: hidden_size → 1 scalar

Loss:
  Pairwise ranking loss = -log σ(score_chosen - score_rejected)
  Same family as DPO but applied to a discriminative head, not the LM.

Usage:
  # Sanity check first
  CUDA_VISIBLE_DEVICES=0 python3 train_orm.py --sanity

  # Full training
  CUDA_VISIBLE_DEVICES=0 python3 train_orm.py

  # With wandb
  CUDA_VISIBLE_DEVICES=0 python3 train_orm.py --wandb --run-name orm_v1

After training, use with Maj@8:
  score = orm.score(question, solution)  → scalar
  pick = argmax(scores over 8 samples)
"""

import argparse
import json
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

# ─── Paths ────────────────────────────────────────────────────────────────────

WORK_DIR   = Path("~/nlp").expanduser()
DATA_FILE  = WORK_DIR / "data" / "dpo_pairs" / "dpo_pairs_raw.jsonl"
CKPT_DIR   = WORK_DIR / "checkpoints" / "orm_v1"
MERGED_DIR = WORK_DIR / "checkpoints" / "orm_v1_merged"
LOG_DIR    = WORK_DIR / "logs"

BASE_MODEL = "Qwen/Qwen2.5-0.5B"   # small, fast, fits on GPU 0 alongside generator

# ─── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sanity",   action="store_true",
                   help="10-step smoke test — always run first")
    p.add_argument("--wandb",    action="store_true")
    p.add_argument("--run-name", type=str,
                   default=f"orm_v1_{datetime.now():%m%d_%H%M}")
    p.add_argument("--epochs",   type=int,   default=3,
                   help="3 epochs standard for pairwise ORM")
    p.add_argument("--lr",       type=float, default=1e-4,
                   help="Higher than DPO — reward head trained from scratch")
    p.add_argument("--lora-rank",type=int,   default=8,
                   help="Small rank — ORM just needs coarse features")
    p.add_argument("--max-len",  type=int,   default=768,
                   help="Max tokens for question + solution")
    return p.parse_args()

# ─── Dataset ──────────────────────────────────────────────────────────────────

class ORMDataset(Dataset):
    """
    Each item: (question, chosen_solution, rejected_solution)
    Loaded from dpo_pairs_raw.jsonl — already has the right format.
    Also creates augmented items: swap chosen/rejected with inverted labels
    to balance the dataset and prevent the model from learning position bias.
    """

    def __init__(self, tokenizer, max_len: int, sanity: bool = False):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.items     = []

        print(f"\nLoading ORM data from: {DATA_FILE}")
        if not DATA_FILE.exists():
            raise FileNotFoundError(f"DPO pairs not found: {DATA_FILE}")

        raw_pairs = []
        with open(DATA_FILE) as f:
            for line in f:
                obj = json.loads(line.strip())
                if obj.get("status") != "pair":
                    continue
                q  = obj.get("question", "")
                ch = obj.get("chosen", "")
                rj = obj.get("rejected", "")
                if not (q and ch and rj):
                    continue
                # Quality filter — chosen must have solution tag
                if "</solution>" not in ch.lower():
                    continue
                raw_pairs.append((q, ch, rj))

        if sanity:
            raw_pairs = raw_pairs[:32]
            print(f"  SANITY MODE — {len(raw_pairs)} pairs")
        else:
            print(f"  Loaded {len(raw_pairs)} valid pairs")

        # Build items: (question, solution, label)
        # label=1 → chosen (correct), label=0 → rejected (wrong)
        for q, ch, rj in raw_pairs:
            self.items.append({"question": q, "solution": ch, "label": 1.0})
            self.items.append({"question": q, "solution": rj, "label": 0.0})

        random.shuffle(self.items)
        print(f"  Total samples (pos+neg): {len(self.items)}")
        print(f"  Positive (correct): {sum(1 for x in self.items if x['label']==1.0)}")
        print(f"  Negative (wrong):   {sum(1 for x in self.items if x['label']==0.0)}")

    def __len__(self):
        return len(self.items)

    def _format(self, question: str, solution: str) -> str:
        """
        Format input for ORM scoring.
        Flat text — no chat template needed for discriminative scoring.
        """
        return (
            f"<|problem|>\n{question.strip()}\n"
            f"<|solution|>\n{solution.strip()}\n"
            f"<|score|>"
        )

    def __getitem__(self, idx):
        item = self.items[idx]
        text = self._format(item["question"], item["solution"])
        enc  = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(item["label"], dtype=torch.float32),
        }


class PairwiseORMDataset(Dataset):
    """
    Pairwise version — each item is (chosen, rejected) for ranking loss.
    This gives stronger gradient signal than binary classification.
    """
    def __init__(self, tokenizer, max_len: int, sanity: bool = False):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.pairs     = []

        print(f"\nLoading pairwise ORM data from: {DATA_FILE}")
        with open(DATA_FILE) as f:
            for line in f:
                obj = json.loads(line.strip())
                if obj.get("status") != "pair":
                    continue
                q  = obj.get("question", "")
                ch = obj.get("chosen", "")
                rj = obj.get("rejected", "")
                if not (q and ch and rj):
                    continue
                if "</solution>" not in ch.lower():
                    continue
                self.pairs.append((q, ch, rj))

        if sanity:
            self.pairs = self.pairs[:32]

        print(f"  Loaded {len(self.pairs)} pairs")
        random.shuffle(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def _encode(self, question, solution):
        text = (
            f"<|problem|>\n{question.strip()}\n"
            f"<|solution|>\n{solution.strip()}\n"
            f"<|score|>"
        )
        return self.tokenizer(
            text, truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt",
        )

    def __getitem__(self, idx):
        q, ch, rj = self.pairs[idx]
        enc_ch = self._encode(q, ch)
        enc_rj = self._encode(q, rj)
        return {
            "chosen_input_ids":       enc_ch["input_ids"].squeeze(0),
            "chosen_attention_mask":  enc_ch["attention_mask"].squeeze(0),
            "rejected_input_ids":     enc_rj["input_ids"].squeeze(0),
            "rejected_attention_mask":enc_rj["attention_mask"].squeeze(0),
        }


# ─── ORM Model ────────────────────────────────────────────────────────────────

class ORM(nn.Module):
    """
    Qwen2.5-0.5B backbone + scalar reward head.

    Backbone: frozen except LoRA adapters on attention layers
    Head:     linear(hidden_size → 1) trained from scratch

    Scoring:  mean-pool last N tokens of backbone hidden states → head → scalar
    """

    def __init__(self, base_model_path: str, lora_rank: int = 16):
        super().__init__()

        print(f"\nLoading backbone: {base_model_path}")
        self.backbone = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": 0},
            trust_remote_code=True,
        )

        # Apply LoRA to backbone — only attention layers
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)
        self.backbone.print_trainable_parameters()
        # Gradient checkpointing — trades compute for memory
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.backbone.enable_input_require_grads()

        hidden_size = self.backbone.config.hidden_size

        # Reward head — trained from scratch
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1, dtype=torch.bfloat16),
        )

        # Init head weights
        for m in self.reward_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask):
        """
        Returns scalar reward for each item in batch.
        Uses mean-pool of last 8 non-padding hidden states.
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Last hidden state: (batch, seq_len, hidden)
        hidden = outputs.hidden_states[-1]

        # Mean pool over last 8 non-padding tokens
        # These are the <|score|> region — most informative for scoring
        last_8_mask = torch.zeros_like(attention_mask)
        for b in range(attention_mask.shape[0]):
            # Find last valid token positions
            valid_positions = attention_mask[b].nonzero().squeeze(-1)
            if len(valid_positions) >= 8:
                last_8_mask[b, valid_positions[-8:]] = 1
            else:
                last_8_mask[b, valid_positions] = 1

        # Weighted mean pool
        mask_expanded = last_8_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask_expanded).sum(dim=1)
        pooled = pooled / last_8_mask.sum(dim=1, keepdim=True).clamp(min=1).to(hidden.dtype)

        # Reward scalar
        reward = self.reward_head(pooled).squeeze(-1)  # (batch,)
        return reward

    def score(self, input_ids, attention_mask):
        """Alias for inference."""
        return self.forward(input_ids, attention_mask)


# ─── Loss ─────────────────────────────────────────────────────────────────────

def pairwise_ranking_loss(score_chosen, score_rejected):
    """
    Bradley-Terry pairwise ranking loss.
    -log σ(s_chosen - s_rejected)
    Pushes score_chosen > score_rejected.
    Same mathematical form as DPO but applied to reward scalars.
    """
    margin = score_chosen - score_rejected
    loss   = -F.logsigmoid(margin).mean()
    # Track accuracy: fraction of pairs where chosen > rejected
    accuracy = (margin > 0).float().mean()
    return loss, accuracy


# ─── Training ─────────────────────────────────────────────────────────────────

def train(args):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"ORM Training — MathReason-1.5B")
    print(f"Run    : {args.run_name}")
    print(f"Device : {device}")
    print(f"Mode   : {'SANITY' if args.sanity else 'FULL'}")
    print(f"{'='*60}")

    # Tokenizer
    print(f"\nLoading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Dataset — pairwise for ranking loss
    dataset = PairwiseORMDataset(tokenizer, max_len=args.max_len, sanity=args.sanity)

    # Train/val split — 90/10
    n_val   = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    batch_size = 1
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"\n  Train pairs: {len(train_ds)}")
    print(f"  Val pairs:   {len(val_ds)}")

    # Model
    orm = ORM(BASE_MODEL, lora_rank=args.lora_rank).to(device)

    # Optimizer — reward head gets higher LR than LoRA backbone
    head_params    = list(orm.reward_head.parameters())
    lora_params    = [p for n, p in orm.named_parameters()
                      if "lora_" in n and p.requires_grad]
    other_params   = [p for n, p in orm.named_parameters()
                      if p.requires_grad and "lora_" not in n
                      and not any(p is hp for hp in head_params)]

    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": args.lr,         "weight_decay": 0.01},
        {"params": lora_params, "lr": args.lr * 0.1,   "weight_decay": 0.01},
    ], betas=(0.9, 0.999))

    total_steps   = len(train_loader) * args.epochs
    warmup_steps  = min(50, total_steps // 10)
    scheduler     = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    if args.wandb:
        import wandb
        wandb.init(project="mathreason-orm", name=args.run_name)

    print(f"\n  Training config:")
    print(f"    base model  : {BASE_MODEL}")
    print(f"    lora rank   : {args.lora_rank}")
    print(f"    epochs      : {args.epochs}")
    print(f"    lr (head)   : {args.lr}")
    print(f"    lr (lora)   : {args.lr * 0.1}")
    print(f"    batch size  : {batch_size}")
    print(f"    total steps : {total_steps}")
    print(f"    loss        : pairwise ranking (Bradley-Terry)")
    print()

    best_val_acc = 0.0
    log_entries  = []

    for epoch in range(args.epochs):
        # ── Train ─────────────────────────────────────────────────────────────
        orm.train()
        train_loss, train_acc, n_steps = 0.0, 0.0, 0

        for step, batch in enumerate(train_loader):
            if args.sanity and step >= 10:
                break

            ch_ids  = batch["chosen_input_ids"].to(device)
            ch_mask = batch["chosen_attention_mask"].to(device)
            rj_ids  = batch["rejected_input_ids"].to(device)
            rj_mask = batch["rejected_attention_mask"].to(device)

            score_ch = orm(ch_ids, ch_mask)
            score_rj = orm(rj_ids, rj_mask)

            loss, acc = pairwise_ranking_loss(score_ch, score_rj)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(orm.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_acc  += acc.item()
            n_steps    += 1

            if step % 20 == 0:
                avg_loss = train_loss / n_steps
                avg_acc  = train_acc  / n_steps
                lr_now   = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1}/{args.epochs} | "
                      f"Step {step:>4}/{len(train_loader)} | "
                      f"loss={avg_loss:.4f} | "
                      f"rank_acc={avg_acc*100:.1f}% | "
                      f"lr={lr_now:.2e} | "
                      f"sc={score_ch.mean().item():.3f} "
                      f"sr={score_rj.mean().item():.3f}")

        # ── Validation ────────────────────────────────────────────────────────
        orm.eval()
        val_loss, val_acc, v_steps = 0.0, 0.0, 0

        with torch.no_grad():
            for batch in val_loader:
                ch_ids  = batch["chosen_input_ids"].to(device)
                ch_mask = batch["chosen_attention_mask"].to(device)
                rj_ids  = batch["rejected_input_ids"].to(device)
                rj_mask = batch["rejected_attention_mask"].to(device)

                score_ch = orm(ch_ids, ch_mask)
                score_rj = orm(rj_ids, rj_mask)

                loss, acc = pairwise_ranking_loss(score_ch, score_rj)
                val_loss += loss.item()
                val_acc  += acc.item()
                v_steps  += 1

        avg_val_loss = val_loss / max(v_steps, 1)
        avg_val_acc  = val_acc  / max(v_steps, 1)
        avg_tr_loss  = train_loss / max(n_steps, 1)
        avg_tr_acc   = train_acc  / max(n_steps, 1)

        print(f"\n  ── Epoch {epoch+1} Summary ──")
        print(f"     train loss={avg_tr_loss:.4f}  rank_acc={avg_tr_acc*100:.1f}%")
        print(f"     val   loss={avg_val_loss:.4f}  rank_acc={avg_val_acc*100:.1f}%")

        # Save best checkpoint by val ranking accuracy
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            ckpt_path = CKPT_DIR / "best_orm.pt"
            torch.save({
                "epoch":         epoch + 1,
                "model_state":   orm.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "val_rank_acc":  avg_val_acc,
                "val_loss":      avg_val_loss,
                "args":          vars(args),
            }, ckpt_path)
            print(f"     ⭐ New best val rank_acc={avg_val_acc*100:.1f}% → saved")

        if args.wandb:
            import wandb
            wandb.log({
                "train/loss": avg_tr_loss, "train/rank_acc": avg_tr_acc,
                "val/loss": avg_val_loss,  "val/rank_acc": avg_val_acc,
                "epoch": epoch + 1,
            })

        log_entries.append({
            "epoch": epoch+1, "train_loss": avg_tr_loss,
            "train_rank_acc": avg_tr_acc, "val_loss": avg_val_loss,
            "val_rank_acc": avg_val_acc,
        })
        print()

    # ── Save final + inference wrapper ────────────────────────────────────────
    final_path = CKPT_DIR / "final_orm.pt"
    torch.save(orm.state_dict(), final_path)
    # Also save tokenizer alongside for inference
    tokenizer.save_pretrained(str(CKPT_DIR))

    log_path = LOG_DIR / "orm_training_log.jsonl"
    with open(log_path, "a") as f:
        for entry in log_entries:
            f.write(json.dumps({"run": args.run_name, **entry}) + "\n")

    print(f"{'='*60}")
    print(f"ORM Training Complete")
    print(f"Best val ranking accuracy : {best_val_acc*100:.1f}%")
    print(f"Best checkpoint           : {CKPT_DIR}/best_orm.pt")
    print(f"Log                       : {log_path}")
    print(f"{'='*60}")
    print()
    print("Interpretation:")
    print("  rank_acc > 85% → ORM reliably picks correct over wrong")
    print("  rank_acc > 90% → strong selector, use for Maj@8 reranking")
    print("  rank_acc < 75% → ORM undertrained, try more epochs or lower LR")
    print()
    print("Next step:")
    print("  Use best_orm.pt in eval_prm_maj8.py as the scorer")
    print("  Replace score_with_orm() with load_and_score_orm()")

    if args.sanity:
        print("\n✅ Sanity run passed — re-run without --sanity for full training")

    return best_val_acc


# ─── Inference helper (for use in eval scripts) ───────────────────────────────

def load_orm(ckpt_path: str, device: str = "cuda:0"):
    """
    Load trained ORM for inference.

    Usage in eval script:
        orm, orm_tok = load_orm("~/nlp/checkpoints/orm_v1/best_orm.pt")
        score = inference_score(orm, orm_tok, question, solution, device)
    """
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(ckpt_path).parent), trust_remote_code=True
    )
    orm = ORM(BASE_MODEL, lora_rank=16)
    state = torch.load(ckpt_path, map_location="cpu")
    if "model_state" in state:
        state = state["model_state"]
    orm.load_state_dict(state)
    orm = orm.to(device)
    orm.eval()
    return orm, tokenizer


def inference_score(orm, tokenizer, question: str, solution: str,
                    device: str = "cuda:0", max_len: int = 1536) -> float:
    """
    Score a single (question, solution) pair.
    Returns a scalar — higher = more likely correct.

    Drop-in replacement for score_with_orm() in eval scripts.
    """
    text = (
        f"<|problem|>\n{question.strip()}\n"
        f"<|solution|>\n{solution.strip()}\n"
        f"<|score|>"
    )
    enc = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=max_len,
        padding="max_length",
    )
    with torch.inference_mode():
        score = orm(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
        )
    return score.item()


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    train(args)
