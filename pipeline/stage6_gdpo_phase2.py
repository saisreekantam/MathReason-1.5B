# ══════════════════════════════════════════════════════════════════════════════
# stage6_gdpo_phase2.py
# ══════════════════════════════════════════════════════════════════════════════
# Second GDPO — built on stage5_gap_sft_merged
#
# What changed vs Phase 1 (stage4c_fullgrpo.py):
#   1. Input: stage5_gap_sft_merged (tag rate 100%, but thinking compressed)
#   2. Reward: correctness + termination + DEPTH (not efficiency)
#              Phase 1 rewarded brevity → compressed thinking
#              Phase 2 rewards adequate depth → restores 300-700w reasoning
#   3. Dataset: harder problems (model improved) + state tracking emphasis
#   4. max_comp_len: 3072 (was 1500 → caused clipped_ratio=0.89 bug)
#   5. Steps: 600 (Phase 1 was 1500, but starting from stronger base)
#
# Reward design (3 components, normalize_then_sum):
#   R1: correctness   — 1.0 correct / 0.0 wrong (primary)
#   R2: termination   — rewards <solution> present + clean stop after
#   R3: depth quality — rewards 300-700w thinking, punishes <80w and >1200w
#
# Run:
#   CUDA_VISIBLE_DEVICES=1 python stage6_gdpo_phase2.py --sanity
#   CUDA_VISIBLE_DEVICES=1 python stage6_gdpo_phase2.py --run-name gdpo_p2_v1
# ══════════════════════════════════════════════════════════════════════════════

import argparse
import json
import math
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from trl import GRPOConfig, GRPOTrainer

# ─── Paths ────────────────────────────────────────────────────────────────────
WORK_DIR   = Path("~/nlp").expanduser()
INPUT_MODEL= WORK_DIR / "checkpoints" / "stage5_gap_sft_merged"
LOG_DIR    = WORK_DIR / "logs"
DATA_CACHE = WORK_DIR / "data" / "cache"

SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Think step by step inside <think> tags, "
    "then give your final answer inside <solution> tags."
)

# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Stage 6 — GDPO Phase 2")
    p.add_argument("--sanity",       action="store_true")
    p.add_argument("--resume",       action="store_true")
    p.add_argument("--wandb",        action="store_true")
    p.add_argument("--run-name",     type=str, default="gdpo_p2_v1")
    p.add_argument("--model",        type=str, default=str(INPUT_MODEL))
    p.add_argument("--max-steps",    type=int, default=600)
    p.add_argument("--lr",           type=float, default=5e-7)
    p.add_argument("--beta",         type=float, default=0.04)
    p.add_argument("--num-gen",      type=int, default=8)
    p.add_argument("--max-comp-len", type=int, default=3072)
    p.add_argument("--batch-size",   type=int, default=1)
    return p.parse_args()

# ─── Answer extraction ────────────────────────────────────────────────────────
def extract_boxed(text: str) -> Optional[str]:
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    return m.group(1).strip() if m else None

def normalize_answer(s: str) -> str:
    if not s:
        return ""
    s = str(s).strip()
    s = re.sub(r"\\%|%", "", s)
    s = re.sub(r"\$|,", "", s)
    s = s.strip()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else f"{f:.4f}".rstrip("0").rstrip(".")
    except:
        return s.lower().strip()

def try_numeric_equal(a: str, b: str) -> Optional[bool]:
    try:
        return abs(float(a) - float(b)) < 1e-4
    except:
        return None

def verify_answer(predicted: Optional[str], ground_truth: str) -> bool:
    if not predicted or not ground_truth:
        return False
    gt_parts = [g.strip() for g in re.split(r"(?:\s+and\s+|,\s*)", ground_truth)]
    pred_norm = normalize_answer(predicted)
    if not pred_norm:
        return False
    for gt_part in gt_parts:
        gt_norm = normalize_answer(gt_part)
        if not gt_norm:
            continue
        if pred_norm == gt_norm:
            return True
        num_eq = try_numeric_equal(pred_norm, gt_norm)
        if num_eq:
            return True
    return False

def extract_predicted_answer(response: str) -> Optional[str]:
    if isinstance(response, list):
        response = " ".join(
            m.get("content", "") if isinstance(m, dict) else str(m)
            for m in response
        )
    elif not isinstance(response, str):
        response = str(response)

    # 1. Solution tag first
    sol_m = re.search(r"<solution>(.*?)</solution>", response, re.DOTALL)
    if sol_m:
        content = sol_m.group(1).strip()
        boxed = extract_boxed(content)
        return boxed if boxed else content

    # 2. Boxed anywhere
    boxed = extract_boxed(response)
    if boxed:
        return boxed

    # 3. After </think> tag — model ends thinking, answer follows
    if "</think>" in response:
        after_think = response.split("</think>", 1)[1].strip()
        nums = re.findall(r"-?[\d,]+(?:\.\d+)?", after_think[:300])
        if nums:
            return nums[0].replace(",", "")

    # 4. First 60% scan — avoid loop region
    cutoff = int(len(response) * 0.6)
    early  = response[:cutoff]
    m = re.search(
        r"(?:so|therefore|thus|answer is|equals?|result is)[^.]*?=\s*(-?[\d,./]+)",
        early, re.IGNORECASE
    )
    if m:
        return m.group(1).replace(",", "")

    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", early)
    return nums[-1].replace(",", "") if nums else None

def extract_ground_truth(raw_answer: str, source: str) -> str:
    if source in ("math", "math_l3l4", "deepmath"):
        boxed = extract_boxed(raw_answer)
        return boxed if boxed else raw_answer.strip()
    if source == "metamath":
        if "####" in raw_answer:
            return raw_answer.split("####")[-1].strip()
        m = re.search(r"[Tt]he answer is[:\s]+([^\n.]+)", raw_answer)
        if m:
            return m.group(1).strip()
        nums = re.findall(r"-?[\d,]+(?:\.\d+)?", raw_answer)
        return nums[-1].replace(",", "") if nums else raw_answer.strip()
    return str(raw_answer).strip()

# ─── Reward System ────────────────────────────────────────────────────────────
# Three components, each normalized to [-1, 1] range before combining.
#
# R1 — CORRECTNESS (weight 1.0, primary signal)
#   1.0 if correct, 0.0 if wrong
#   Binary. Clean gradient signal.
#
# R2 — TERMINATION QUALITY (weight 0.15)
#   +0.15 : has <solution>...</solution> AND clean stop (nothing after)
#   +0.05 : has tag but continues writing after (partial credit)
#   -0.10 : no solution tag at all
#   Note: tag rate is already 100% after Stage 5, so this mainly
#         reinforces the "stop after tag" behavior
#
# R3 — REASONING DEPTH (weight 0.12)
#   Gaussian-shaped reward centered at 400 words.
#   Peak +0.12 at 400 words.
#   Decays toward 0 at 100 and 900 words.
#   Negative for <80 words (too shallow) and >1200 words (loop territory).
#   Only applied on CORRECT completions (no depth reward on wrong answers).
#
#   Why this shape?
#   Stage 5 SFT compressed thinking to 67-200 words. We need to push it
#   back to 300-700 words for adequate multi-step reasoning.
#   We do NOT want to penalize longer thinking on hard problems,
#   so the decay is gentle — only heavily penalizes >1200 (loop territory).

class MultiReward:
    def __init__(self,
                 w_correct: float = 1.0,
                 w_term:    float = 0.15,
                 w_depth:   float = 0.12):
        self.w_correct = w_correct
        self.w_term    = w_term
        self.w_depth   = w_depth
        self.stats     = defaultdict(list)
        self.n_calls   = 0

    def _r_correct(self, response: str, ground_truth: str) -> Tuple[float, bool]:
        pred       = extract_predicted_answer(response)
        is_correct = verify_answer(pred, ground_truth)
        return (self.w_correct if is_correct else 0.0), is_correct

    def _r_termination(self, response: str) -> float:
        has_tag  = "<solution>" in response and "</solution>" in response
        if not has_tag:
            return -0.10 * self.w_term

        # Check if model stops cleanly after </solution>
        after_sol = response.split("</solution>")[-1].strip()
        clean_stop = len(after_sol) < 30
        if clean_stop:
            return 1.0 * self.w_term     # perfect: tag + clean stop
        else:
            return 0.3 * self.w_term     # tag present but keeps writing

    def _r_depth(self, response: str, is_correct: bool) -> float:
        # Only apply depth reward on correct completions
        # Wrong completions shouldn't be rewarded for any thinking length
        if not is_correct:
            return 0.0

        # Count think words — handle both R1 embedded and tagged formats
        think_m = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        if think_m:
            words = len(think_m.group(1).split())
        elif "</think>" in response:
            # R1-Distill format: thinking before </think>
            words = len(response.split("</think>")[0].split())
        else:
            words = len(response.split())

        # Gaussian centered at 400 words, sigma=300
        # Peak reward = +w_depth at 400 words
        # At 100 words: +0.35 * w_depth
        # At 800 words: +0.35 * w_depth
        # At 1200 words: +0.01 * w_depth (near zero)
        gaussian = math.exp(-((words - 400) ** 2) / (2 * 300 ** 2))

        # Extra penalty for very shallow (<80 words) — Stage 5 regression
        if words < 80:
            shallow_penalty = -0.5 * self.w_depth
            return shallow_penalty

        # Extra penalty for loop territory (>1200 words)
        if words > 1200:
            loop_penalty = -0.3 * self.w_depth * min((words - 1200) / 600, 1.0)
            return loop_penalty

        return gaussian * self.w_depth

    def compute(self, response: str, ground_truth: str) -> float:
        if isinstance(response, list):
            response = " ".join(
                m.get("content", "") if isinstance(m, dict) else str(m)
                for m in response
            )

        r1, is_correct = self._r_correct(response, ground_truth)
        r2             = self._r_termination(response)
        r3             = self._r_depth(response, is_correct)

        total = r1 + r2 + r3

        # Track stats
        self.stats["correct"].append(float(is_correct))
        self.stats["r_correct"].append(r1)
        self.stats["r_term"].append(r2)
        self.stats["r_depth"].append(r3)
        self.stats["total"].append(total)
        self.stats["has_sol"].append(
            float("<solution>" in response and "</solution>" in response)
        )
        think_m = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        if think_m:
            tw = len(think_m.group(1).split())
        elif "</think>" in response:
            tw = len(response.split("</think>")[0].split())
        else:
            tw = len(response.split())
        self.stats["think_words"].append(tw)
        self.n_calls += 1

        return total

    def get_stats(self, last_n: int = 200) -> Dict:
        def avg(lst):
            tail = lst[-last_n:] if len(lst) >= last_n else lst
            return sum(tail) / max(len(tail), 1)
        return {
            "pass_rate":   avg(self.stats["correct"]),
            "tag_rate":    avg(self.stats["has_sol"]),
            "r_correct":   avg(self.stats["r_correct"]),
            "r_term":      avg(self.stats["r_term"]),
            "r_depth":     avg(self.stats["r_depth"]),
            "mean_reward": avg(self.stats["total"]),
            "think_words": avg(self.stats["think_words"]),
            "n_total":     self.n_calls,
        }


_reward: Optional[MultiReward] = None


def make_reward_fn(reward: MultiReward):
    def reward_fn(completions, ground_truths=None, **kwargs):
        # Handle both old and new TRL reward fn signatures
        if ground_truths is None:
            ground_truths = kwargs.get("solution", [""] * len(completions))

        rewards = []
        for comp, gt in zip(completions, ground_truths):
            r = reward.compute(comp, str(gt))
            rewards.append(r)

        # Dynamic batch skip: if all rewards identical (std≈0), return zeros
        # This avoids degenerate updates when all completions are equally wrong
        if len(rewards) > 1:
            mean_r = sum(rewards) / len(rewards)
            std_r  = (sum((r - mean_r)**2 for r in rewards) / len(rewards)) ** 0.5
            if std_r < 1e-6:
                return [0.0] * len(rewards)

        return rewards

    return reward_fn

# ─── Dataset ──────────────────────────────────────────────────────────────────
# Target pass rate: 25-55% for healthy RL signal
# After Stage 5, model improved on word problems.
# Need harder problems than Phase 1 used.
#
# Sources (in order of preference):
#   1. DeepMath-103K difficulty 3.5-5.5  — hard arithmetic, proofs
#   2. MATH Level 3-4                    — algebra, counting, geometry
#   3. MetaMath MATH variants            — hard word problems
#
# Excluded:
#   - GSM8K easy     : model at >90% → dead gradient
#   - AIME/olympiad  : model at <5% → dead gradient
#   - Difficulty ≥6  : <10% pass rate on 1.5B → dead gradient

def load_deepmath(n: int = 6000) -> Dataset:
    print("  Loading DeepMath-103K (difficulty 3.5-5.5)...")
    try:
        ds = load_dataset(
            "zwhe99/DeepMath-103K",
            split="train",
            cache_dir=str(DATA_CACHE),
        )
        # Filter difficulty 3.5 to 5.5
        def is_valid(ex):
            try:
                d = float(ex.get("difficulty", 0) or 0)
                return 3.5 <= d <= 5.5 and ex.get("answer") and ex.get("problem")
            except:
                return False
        ds = ds.filter(is_valid, num_proc=4)
        if len(ds) > n:
            ds = ds.shuffle(seed=42).select(range(n))
        print(f"    DeepMath: {len(ds):,} problems")
        return ds.map(lambda ex: {
            "problem": ex["problem"],
            "answer":  str(ex["answer"]),
            "source":  "deepmath",
        }, remove_columns=ds.column_names)
    except Exception as e:
        print(f"    DeepMath failed ({e}), skipping")
        return Dataset.from_list([])

def load_math_l3l4(n: int = 4000) -> Dataset:
    print("  Loading MATH Level 3-4...")
    try:
        ds = load_dataset(
            "lighteval/MATH",
            "all",
            split="train",
            cache_dir=str(DATA_CACHE),
        )
        def is_valid(ex):
            try:
                lvl = int(str(ex.get("level", "0")).replace("Level ", "").strip())
                return lvl in (3, 4) and ex.get("problem") and ex.get("solution")
            except:
                return False
        ds = ds.filter(is_valid, num_proc=4)
        if len(ds) > n:
            ds = ds.shuffle(seed=42).select(range(n))
        print(f"    MATH L3-4: {len(ds):,} problems")
        return ds.map(lambda ex: {
            "problem": ex["problem"],
            "answer":  extract_ground_truth(ex["solution"], "math"),
            "source":  "math_l3l4",
        }, remove_columns=ds.column_names)
    except Exception as e:
        print(f"    MATH failed ({e}), skipping")
        return Dataset.from_list([])

def load_metamath_hard(n: int = 2000) -> Dataset:
    """MetaMath MATH variants — harder word problems with multi-step reasoning."""
    print("  Loading MetaMath MATH variants (hard subset)...")
    try:
        ds = load_dataset(
            "meta-math/MetaMathQA",
            split="train",
            cache_dir=str(DATA_CACHE),
        )
        def is_hard_math(ex):
            q = ex.get("query", "") or ex.get("input", "") or ""
            a = ex.get("response", "") or ex.get("output", "") or ""
            # Only MATH variants (harder), not GSM8K rephrases
            orig = ex.get("original_question", "") or ex.get("query_source", "") or ""
            is_math_src = ("MATH" in str(ex.get("type", "")) or
                           "math" in orig.lower() or
                           "algebra" in q.lower() or
                           "probability" in q.lower() or
                           "geometry" in q.lower())
            has_answer = "####" in a or "The answer is" in a
            return is_math_src and has_answer and len(q) > 50
        ds = ds.filter(is_hard_math, num_proc=4)
        if len(ds) > n:
            ds = ds.shuffle(seed=42).select(range(n))
        print(f"    MetaMath hard: {len(ds):,} problems")
        return ds.map(lambda ex: {
            "problem": ex.get("query") or ex.get("input", ""),
            "answer":  extract_ground_truth(
                ex.get("response") or ex.get("output", ""), "metamath"
            ),
            "source":  "metamath",
        }, remove_columns=ds.column_names)
    except Exception as e:
        print(f"    MetaMath failed ({e}), skipping")
        return Dataset.from_list([])

def build_dataset(args) -> Dataset:
    print("\nBuilding dataset...")

    deepmath  = load_deepmath(n=6000)
    math_l3l4 = load_math_l3l4(n=4000)
    metamath  = load_metamath_hard(n=2000)

    parts = [d for d in [deepmath, math_l3l4, metamath] if len(d) > 0]
    if not parts:
        raise RuntimeError("No dataset loaded — check data sources")

    combined = concatenate_datasets(parts)
    combined = combined.shuffle(seed=42)

    # Remove examples with empty answers
    combined = combined.filter(
        lambda ex: bool(ex.get("answer", "").strip()),
        num_proc=4
    )

    print(f"\n  Total problems: {len(combined):,}")

    # Format as prompt-response pairs for GRPO
    def format_example(ex):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": ex["problem"]},
        ]
        return {
            "prompt":   messages,
            "solution": ex["answer"],
        }

    combined = combined.map(format_example, num_proc=4,
                            remove_columns=combined.column_names)
    return combined

# ─── Monitor callback ─────────────────────────────────────────────────────────
class GDPO2Monitor(TrainerCallback):
    def __init__(self, reward: MultiReward, log_path: Path,
                 check_every: int = 25):
        self.reward      = reward
        self.log_path    = log_path
        self.check_every = check_every
        self.history     = []
        self.best_pass   = 0.0
        self.best_step   = 0

    def on_step_end(self, args, state: TrainerState,
                    control: TrainerControl, **kwargs):
        step = state.global_step
        if step % self.check_every != 0 or step == 0:
            return control

        s = self.reward.get_stats(last_n=self.check_every * 8)

        entry = {
            "step":       step,
            "pass_rate":  round(s["pass_rate"],  4),
            "tag_rate":   round(s["tag_rate"],   4),
            "r_correct":  round(s["r_correct"],  4),
            "r_term":     round(s["r_term"],     4),
            "r_depth":    round(s["r_depth"],    4),
            "mean_reward":round(s["mean_reward"],4),
            "think_words":round(s["think_words"],1),
            "timestamp":  datetime.now().isoformat(),
        }
        self.history.append(entry)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"\n  ── Step {step} Monitor ──")
        print(f"    Pass@1       : {s['pass_rate']*100:.1f}%")
        print(f"    Tag rate     : {s['tag_rate']*100:.1f}%")
        print(f"    Avg think    : {s['think_words']:.0f} words  ← target 300-700")
        print(f"    R_correct    : {s['r_correct']:+.3f}")
        print(f"    R_term       : {s['r_term']:+.3f}")
        print(f"    R_depth      : {s['r_depth']:+.3f}")
        print(f"    Mean reward  : {s['mean_reward']:+.3f}")

        if s["pass_rate"] > self.best_pass:
            self.best_pass = s["pass_rate"]
            self.best_step = step
            print(f"    ⭐ New best pass@1: {self.best_pass*100:.1f}%")

        # Health warnings
        if step >= 100:
            if s["think_words"] < 150:
                print(f"    ⚠️  Thinking still too shallow ({s['think_words']:.0f}w) "
                      f"— depth reward should push it up")
            if s["think_words"] > 900:
                print(f"    ⚠️  Thinking getting long ({s['think_words']:.0f}w) "
                      f"— watch for loop regression")
            if s["tag_rate"] < 0.80:
                print(f"    ⚠️  Tag rate dropped to {s['tag_rate']*100:.0f}% "
                      f"— was 100% after Stage 5")
            if s["pass_rate"] < 0.15:
                print(f"    ⚠️  Pass rate very low — dataset may be too hard")

        return control

    def on_train_end(self, args, state, control, **kwargs):
        print(f"\n{'═'*60}")
        print(f"  GDPO PHASE 2 COMPLETE")
        print(f"  Best pass@1: {self.best_pass*100:.1f}% at step {self.best_step}")
        print(f"{'═'*60}")
        return control

# ─── Model ────────────────────────────────────────────────────────────────────
def load_model(model_path: str):
    print(f"\nLoading model from {model_path} ...")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    print(f"  Device: {next(model.parameters()).device}")
    return model, tok

# ─── GRPO Config ──────────────────────────────────────────────────────────────
def build_grpo_config(args, output_dir: Path) -> GRPOConfig:
    # generation_batch_size must be divisible by num_generations
    gen_batch = args.num_gen  # 8 completions, batch=8

    return GRPOConfig(
        # Paths
        output_dir   = str(output_dir),
        run_name     = args.run_name,

        # Steps
        max_steps    = args.max_steps,

        # Batch — full param needs small batch
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = 8,
        num_generations             = args.num_gen,
        generation_batch_size       = gen_batch,

        # Generation
        max_completion_length = args.max_comp_len,  # 3072 — critical fix
        temperature           = 0.8,
        top_p                 = 0.9,

        # LR
        learning_rate         = args.lr,
        lr_scheduler_type     = "cosine",
        warmup_steps          = 20,

        # GRPO
        beta                  = args.beta,
        loss_type             = "dr_grpo",
        # DAPO clip_higher — preserves reasoning tokens ("wait", "hmm")
        # These tokens have low probability but are crucial for backtracking
        epsilon               = 0.20,
        epsilon_high          = 0.28,

        # Multi-reward aggregation — preserves distinct gradient signals
        # normalize_then_sum: each reward normalized independently,
        # then summed. Prevents one reward dominating.
        reward_weights = [1.0],  # single reward_fn, weights handled inside

        # Precision
        bf16                  = True,
        fp16                  = False,

        # Save
        save_strategy         = "steps",
        save_steps            = 100,
        save_total_limit      = 3,

        # Logging
        logging_steps         = 10,
        report_to             = "wandb" if args.wandb else "none",

        # Gradient
        max_grad_norm         = 1.0,
        gradient_checkpointing= True,

        # Dataset
            )

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    args    = parse_args()
    sanity  = args.sanity

    run_name   = args.run_name if not sanity else "gdpo_p2_sanity"
    output_dir = WORK_DIR / "checkpoints" / run_name
    merged_dir = WORK_DIR / "checkpoints" / f"{run_name}_merged"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*65}")
    print(f"  GDPO Phase 2 — Stage 6")
    print(f"  Input  : {args.model}")
    print(f"  Output : {output_dir}")
    print(f"  Sanity : {sanity}")
    print(f"  Steps  : {args.max_steps}")
    print(f"  max_comp_len: {args.max_comp_len}")
    print(f"{'═'*65}")

    # Load
    model, tok = load_model(args.model)

    # Dataset
    dataset = build_dataset(args)
    if sanity:
        dataset = dataset.select(range(min(80, len(dataset))))
    print(f"\n  Training on {len(dataset):,} problems")

    # Reward
    global _reward
    _reward    = MultiReward(w_correct=1.0, w_term=0.15, w_depth=0.12)
    reward_fn  = make_reward_fn(_reward)

    # Config
    if sanity:
        args.max_steps    = 25
        args.max_comp_len = 1024
    grpo_cfg = build_grpo_config(args, output_dir)

    # Monitor
    monitor = GDPO2Monitor(
        reward      = _reward,
        log_path    = LOG_DIR / f"{run_name}_monitor.json",
        check_every = 5 if sanity else 25,
    )

    # Trainer — full parameter, no LoRA
    trainer = GRPOTrainer(
        model            = model,
        args             = grpo_cfg,
        reward_funcs     = [reward_fn],
        train_dataset    = dataset,
        processing_class = tok,
    )
    trainer.add_callback(monitor)

    print(f"\n  Starting GDPO Phase 2 ...")
    print(f"  Monitor: tail -f {LOG_DIR}/{run_name}_monitor.json\n")

    trainer.train(
        resume_from_checkpoint=str(output_dir) if args.resume else None
    )

    # Save stats
    s = _reward.get_stats()
    print(f"\n  Final Stats:")
    print(f"    Pass@1     : {s['pass_rate']*100:.1f}%")
    print(f"    Tag rate   : {s['tag_rate']*100:.1f}%")
    print(f"    Think words: {s['think_words']:.0f}w")
    print(f"    R_depth avg: {s['r_depth']:+.3f}")

    # Merge
    if not sanity:
        print(f"\n  Merging to {merged_dir} ...")
        from transformers import AutoModelForCausalLM as AMCL
        merged_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(merged_dir))
        tok.save_pretrained(str(merged_dir))
        print(f"  ✅ Saved to {merged_dir}")

        print(f"\n{'═'*65}")
        print(f"  NEXT: push to HF then run DPO")
        print(f"  python stage7_dpo.py --model {merged_dir}")
        print(f"{'═'*65}")
    else:
        print("\n  Sanity done. Check:")
        print("  1. Loss not NaN")
        print("  2. Pass@1 > 10%")
        print("  3. reward_std > 0.05")
        print("  4. Think words trending UP from baseline (was ~150w)")
        print(f"\n  Full run: CUDA_VISIBLE_DEVICES=1 python "
              f"stage6_gdpo_phase2.py --run-name gdpo_p2_v1")

if __name__ == "__main__":
    main()
