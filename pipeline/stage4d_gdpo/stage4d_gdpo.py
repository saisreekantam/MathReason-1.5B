"""
Stage 4D — GDPO Multi-Reward Training (Phase D: Discipline)
════════════════════════════════════════════════════════════════════════════
Server  : revanth@172.16.192.168
GPU     : CUDA_VISIBLE_DEVICES=0

Input   : ~/nlp/checkpoints/stage4c_fullgrpo_merged
Output  : ~/nlp/checkpoints/stage4d_gdpo

WHY GDPO OVER GRPO:
  With multiple reward functions, standard GRPO normalizes the SUM of rewards.
  This collapses distinct reward combinations into identical advantages:
    correct=1, no_tag=0  → sum=1 → normalized to same value as
    correct=0, tagged=1  → sum=1 → ZERO gradient to distinguish them!

  GDPO normalizes EACH reward separately before summing.
  Result: each reward contributes independently to the advantage.
  One TRL config line: multi_objective_aggregation="normalize_then_sum"

THREE REWARD FUNCTIONS (separate, not summed first):

  1. correctness_reward  — is the answer right? (primary signal)
     Range: [0.0, 1.0]
     Wrong answers: always 0.0

  2. termination_reward  — did you write </solution> and STOP?
     Range: [0.0, 1.0]
     1.0 = clean stop, 0.5 = tagged but kept going, 0.0 = no tag
     Weight: 0.5 (half importance of correctness)

  3. efficiency_reward   — was the response concise? (anti-loop signal)
     Range: [0.0, 1.0]
     1.0 = ideal length, 0.0 = severe repetition or extreme length
     Weight: 0.3 (supporting signal)

  With GDPO, each reward is normalized within its own range.
  No reward drowns out another. No gradient conflict.

DATASET STRATEGY (why NOT DeepMath L5-9):
  DeepMath L5-9 → our 1.5B model gets <5% pass rate → reward std≈0 → dead gradient
  
  Right difficulty window for 1.5B at our stage:
    MATH L3-4        : ~35-50% pass rate ← primary signal
    MetaMath MATH    : ~45-55% pass rate ← stability anchor
    DeepMath L3-5    : ~25-40% pass rate ← harder edge, topic diversity

Usage:
  CUDA_VISIBLE_DEVICES=0 python stage4d_gdpo.py --sanity
  CUDA_VISIBLE_DEVICES=0 python stage4d_gdpo.py --run-name gdpo_v1 --wandb
  CUDA_VISIBLE_DEVICES=0 python stage4d_gdpo.py --resume --run-name gdpo_v1
════════════════════════════════════════════════════════════════════════════
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
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


# ─── Paths ───────────────────────────────────────────────────────────────────

WORK_DIR    = Path("~/nlp").expanduser()
INPUT_MODEL = WORK_DIR / "checkpoints" / "stage4c_fullgrpo_merged"
CKPT_DIR    = WORK_DIR / "checkpoints" / "stage4d_gdpo"
MERGED_DIR  = WORK_DIR / "checkpoints" / "stage4d_gdpo_merged"
LOG_DIR     = WORK_DIR / "logs"
DATA_CACHE  = WORK_DIR / "data" / "cache"

SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Think step by step inside <think> tags, "
    "then give your final answer inside <solution> tags."
)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 4D — GDPO Multi-Reward")

    p.add_argument("--sanity",       action="store_true")
    p.add_argument("--resume",       action="store_true")
    p.add_argument("--wandb",        action="store_true")
    p.add_argument("--run-name",     type=str,   default="gdpo_v1")
    p.add_argument("--model",        type=str,   default=str(INPUT_MODEL))
    p.add_argument("--output-dir",   type=str,   default=str(CKPT_DIR))
    p.add_argument("--merged-dir",   type=str,   default=str(MERGED_DIR))

    # GRPO / GDPO core
    p.add_argument("--num-gen",      type=int,   default=8)
    p.add_argument("--max-steps",    type=int,   default=500)
    p.add_argument("--lr",           type=float, default=5e-7)
    p.add_argument("--beta",         type=float, default=0.04)
    p.add_argument("--temperature",  type=float, default=0.8)
    p.add_argument("--max-comp-len", type=int,   default=1500)

    # DAPO clip_higher (already proven working)
    p.add_argument("--epsilon-low",  type=float, default=0.20)
    p.add_argument("--epsilon-high", type=float, default=0.28)

    # Dataset sizes
    p.add_argument("--n-math",       type=int,   default=4000)
    p.add_argument("--n-metamath",   type=int,   default=4000)
    p.add_argument("--n-deepmath",   type=int,   default=6000)

    p.add_argument("--skip-merge",   action="store_true")
    p.add_argument("--attn-impl",    type=str,   default="sdpa")

    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════════
# ANSWER VERIFICATION — same proven extractor from Phase C
# ════════════════════════════════════════════════════════════════════════════

def extract_boxed(text: str) -> Optional[str]:
    idx = text.rfind("\\boxed{")
    if idx == -1:
        idx = text.rfind("\\boxed {")
        if idx == -1:
            return None
    start = text.find("{", idx) + 1
    depth, pos = 1, start
    while pos < len(text) and depth > 0:
        if text[pos] == "{":   depth += 1
        elif text[pos] == "}": depth -= 1
        pos += 1
    return text[start:pos-1].strip() if depth == 0 else None


def normalize_answer(ans: str) -> str:
    if ans is None:
        return ""
    ans = str(ans).strip()
    ans = re.sub(r"\\d?frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", ans)
    ans = re.sub(r"\\text\{([^}]*)\}",    r"\1", ans)
    ans = re.sub(r"\\mathrm\{([^}]*)\}",  r"\1", ans)
    ans = re.sub(r"\\left|\\right",        "",    ans)
    ans = re.sub(r"\\%",                   "%",   ans)
    ans = ans.replace("$", "").strip()
    ans = re.sub(
        r"\s*(dollars?|cents?|meters?|km|kg|cm|miles?|feet|inches?|hours?|"
        r"minutes?|seconds?|%|sq\.?\s*\w*)\s*$",
        "", ans, flags=re.IGNORECASE
    ).strip()
    ans = re.sub(r"(\d),(\d)", r"\1\2", ans)
    try:
        f = float(ans)
        if f == int(f):
            return str(int(f))
        return f"{f:.6f}".rstrip("0")
    except (ValueError, OverflowError):
        pass
    return ans.lower().strip()


def try_numeric_equal(a: str, b: str) -> Optional[bool]:
    try:
        fa, fb = float(a), float(b)
        return abs(fa - fb) < max(1e-6, 1e-4 * max(abs(fa), abs(fb)))
    except:
        return None


def try_fraction_equal(a: str, b: str) -> Optional[bool]:
    def to_float(s):
        m = re.match(r"^(-?\d+)\s*/\s*(-?\d+)$", s.strip())
        if m:
            num, den = int(m.group(1)), int(m.group(2))
            return num / den if den != 0 else None
        try:
            return float(s)
        except:
            return None
    fa, fb = to_float(a), to_float(b)
    if fa is not None and fb is not None:
        return abs(fa - fb) < 1e-5
    return None


def verify_answer(predicted: Optional[str], ground_truth: str) -> bool:
    if predicted is None or ground_truth is None:
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
        if try_numeric_equal(pred_norm, gt_norm):
            return True
        if try_fraction_equal(pred_norm, gt_norm):
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

    # Strip system prompt echo
    echo = response.find("You are a mathematical reasoning assistant")
    if echo > 50:
        response = response[:echo]

    # 1. Solution tag
    sol_m = re.search(r"<solution>(.*?)</solution>", response, re.DOTALL)
    if sol_m:
        content = sol_m.group(1).strip()
        boxed = extract_boxed(content)
        return boxed if boxed else content

    # 2. Boxed anywhere
    boxed = extract_boxed(response)
    if boxed:
        return boxed

    # 3. First 60% scan
    cutoff = int(len(response) * 0.6)
    early  = response[:cutoff]

    # Try fraction first
    frac_m = re.findall(r'\b(\d+)\s*/\s*(\d+)\b', early)
    if frac_m:
        num, den = int(frac_m[-1][0]), int(frac_m[-1][1])
        if den != 0:
            val = num / den
            return str(int(val)) if val == int(val) else str(round(val, 4))

    m = re.search(
        r"(?:so|therefore|thus|answer is|equals?|result is)"
        r"[^.]*?=\s*(-?[\d,./]+)",
        early, re.IGNORECASE
    )
    if m:
        return m.group(1).replace(",", "")

    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", early)
    if nums:
        return nums[-1].replace(",", "")

    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", response)
    return nums[-1].replace(",", "") if nums else None


def extract_ground_truth(raw_answer: str, source: str) -> str:
    if source == "math":
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
    if source == "deepmath":
        return str(raw_answer).strip()
    return raw_answer.strip()


# ════════════════════════════════════════════════════════════════════════════
# REWARD FUNCTIONS — THREE SEPARATE FUNCTIONS FOR GDPO
#
# GDPO key: pass as list to reward_funcs, set multi_objective_aggregation
# Each function normalized independently → equal gradient influence
# ════════════════════════════════════════════════════════════════════════════

# ── Global stats tracker ──────────────────────────────────────────────────
_stats = defaultdict(list)
_n_calls = 0


def _get_text(completion) -> str:
    """Extract text from completion (handles trl dict format)."""
    if isinstance(completion, list):
        return " ".join(
            m.get("content", "") if isinstance(m, dict) else str(m)
            for m in completion
        )
    return str(completion) if not isinstance(completion, str) else completion


def _n_gram_repeat_ratio(text: str, n: int = 8) -> float:
    """Returns 0.0 (no repetition) to 1.0 (all repeated)."""
    tokens = text.split()
    if len(tokens) < n * 3:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n)]
    if not ngrams:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)


# ── Reward 1: Correctness ────────────────────────────────────────────────

def correctness_reward(
    prompts: List[str],
    completions: List[str],
    ground_truth: List[str] = None,
    **kwargs
) -> List[float]:
    """
    Primary signal. Binary correct/wrong.
    Range: [0.0, 1.0]
    Weight in GDPO: 1.0 (highest priority)
    TRL passes dataset columns as named kwargs matching column names.
    """
    ground_truths = ground_truth if ground_truth is not None else [""] * len(completions)
    rewards = []
    for comp, gt in zip(completions, ground_truths):
        text      = _get_text(comp)
        predicted = extract_predicted_answer(text)
        is_correct = verify_answer(predicted, str(gt))
        r = 1.0 if is_correct else 0.0
        rewards.append(r)
        _stats["correct"].append(float(is_correct))
    return rewards


# ── Reward 2: Termination ────────────────────────────────────────────────

def termination_reward(
    prompts: List[str],
    completions: List[str],
    ground_truth: List[str] = None,
    **kwargs
) -> List[float]:
    """
    Did the model write </solution> and STOP?
    Range: [0.0, 1.0]
    Weight in GDPO: 0.5

    Design:
      1.0 = <solution> written, clean stop (<30 chars after)
      0.5 = <solution> written, kept going (loops after tag)
      0.0 = no <solution> tag at all

    GDPO ensures this has equal normalized influence as correctness.
    This creates gradient signal that distinguishes:
      "correct + no tag" from "correct + clean tag"
    Which binary GRPO could NOT do (both got reward=1.0).
    """
    rewards = []
    for comp in completions:
        text     = _get_text(comp)
        has_sol  = "<solution>" in text and "</solution>" in text
        if has_sol:
            after_sol = text.split("</solution>")[-1].strip()
            clean_end = len(after_sol) < 30
            r = 1.0 if clean_end else 0.5
        else:
            r = 0.0
        rewards.append(r)
        _stats["has_tag"].append(float(has_sol))
    return rewards


# ── Reward 3: Efficiency ─────────────────────────────────────────────────

def efficiency_reward(
    prompts: List[str],
    completions: List[str],
    ground_truth: List[str] = None,
    **kwargs
) -> List[float]:
    """
    Was the response reasonably concise?
    Range: [0.0, 1.0]
    Weight in GDPO: 0.3

    Design — continuous Gaussian-style, no hard step functions:
      ~400 words : 1.0 (ideal)
      ~700 words : 0.75 (good)
      ~1000 words: 0.50 (acceptable)
      ~1300 words: 0.30 (long)
      >1500 words: 0.15 (very long)
      severe loop: 0.0  (n-gram ratio > 0.35)

    Continuous to avoid step-function gradient issues.
    """
    rewards = []
    for comp in completions:
        text  = _get_text(comp)
        words = len(text.split())

        # Severe repetition check first
        repeat = _n_gram_repeat_ratio(text)
        if repeat > 0.35:
            rewards.append(0.0)
            _stats["repeat_penalty"].append(-1.0)
            _stats["word_count"].append(words)
            continue

        # Continuous length reward — smooth Gaussian decay
        # Peak at 400 words, half-width ~600 words
        ideal   = 400.0
        width   = 700.0
        r = math.exp(-0.5 * ((words - ideal) / width) ** 2)

        # But don't reward TOO short (< 100 words) — might be cutting corners
        if words < 100:
            r = r * 0.3

        rewards.append(max(0.05, r))  # floor at 0.05 (never fully zero for length)
        _stats["repeat_penalty"].append(0.0)
        _stats["word_count"].append(words)
    return rewards


# ════════════════════════════════════════════════════════════════════════════
# DATASET — three sources, difficulty-calibrated for our 1.5B model
# Target pass rate: 25-55% per problem (ensures reward variance)
# ════════════════════════════════════════════════════════════════════════════

def make_prompt(problem: str) -> List[Dict]:
    return [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": problem.strip()},
    ]


def load_math_l3l4(n: int, sanity: bool) -> Dataset:
    """
    MATH Level 3-4. Our model pass rate: ~35-50%. Primary MATH500 signal.
    Directly aligned with our benchmark target.
    """
    print("  Loading MATH L3-4...")
    try:
        ds = load_dataset(
            "DigitalLearningGmbH/MATH-lighteval",
            split="train",
            cache_dir=str(DATA_CACHE)
        )
        ds = ds.filter(lambda x: x.get("level", "") in ["Level 3", "Level 4"])
    except Exception as e:
        print(f"  ⚠️  MATH load failed: {e}")
        return Dataset.from_dict(
            {"prompt": [], "ground_truth": [], "problem_id": [], "source": []}
        )

    ds = ds.select(range(min(20 if sanity else n, len(ds))))

    def process(ex):
        answer = extract_boxed(ex.get("solution", "")) or ex.get("solution", "")
        return {
            "prompt":     make_prompt(ex["problem"]),
            "ground_truth": answer,
            "problem_id": hashlib.md5(ex["problem"].encode()).hexdigest()[:12],
            "source":     "math",
        }

    ds = ds.map(process, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: x["ground_truth"] and len(x["ground_truth"]) > 0)
    print(f"  ✅ MATH L3-4: {len(ds):,}")
    return ds


def load_metamath_math(n: int, sanity: bool) -> Dataset:
    """
    MetaMath MATH-style (NOT GSM8K). Our model pass rate: ~45-55%.
    Stability anchor — ensures gradient doesn't spike from all-hard batches.
    """
    print("  Loading MetaMath MATH-style...")
    try:
        ds = load_dataset(
            "meta-math/MetaMathQA",
            split="train",
            cache_dir=str(DATA_CACHE)
        )
        # Keep only MATH-sourced problems, exclude GSM8K (model 80% = dead signal)
        ds = ds.filter(
            lambda x: "MATH" in x.get("type", "") and
                      "GSM" not in x.get("type", "")
        )
    except Exception as e:
        print(f"  ⚠️  MetaMath load failed: {e}")
        return Dataset.from_dict(
            {"prompt": [], "ground_truth": [], "problem_id": [], "source": []}
        )

    ds = ds.shuffle(seed=42).select(range(min(20 if sanity else n, len(ds))))

    def process(ex):
        q = ex.get("query", "")
        a = ex.get("response", "")
        # Extract final answer
        if "The answer is" in a:
            ans = a.split("The answer is")[-1].strip().rstrip(".").strip()
        else:
            nums = re.findall(r"-?[\d,]+(?:\.\d+)?", a[-200:])
            ans  = nums[-1].replace(",", "") if nums else a[-50:]
        return {
            "prompt":       make_prompt(q),
            "ground_truth": ans,
            "problem_id":   hashlib.md5(q.encode()).hexdigest()[:12],
            "source":       "metamath",
        }

    ds = ds.map(process, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: x["ground_truth"] and len(x["ground_truth"]) > 0)
    print(f"  ✅ MetaMath MATH: {len(ds):,}")
    return ds


def load_deepmath_easy(n: int, sanity: bool) -> Dataset:
    """
    DeepMath-103K filtered to difficulty 3.0-5.5 only.
    Full DeepMath (Level 5-9) is too hard for our 1.5B model → dead gradient.
    Level 3-5 subset: our model pass rate ~25-40% → good difficulty window.
    Has topic diversity advantage (Algebra, Geometry, NT, Combinatorics, etc.)
    Decontaminated against MATH500/GSM8K benchmarks.
    """
    print("  Loading DeepMath-103K (difficulty 3.0-5.5 only)...")
    try:
        ds = load_dataset(
            "zwhe99/DeepMath-103K",
            split="train",
            cache_dir=str(DATA_CACHE)
        )
        # Filter to difficulty ≤ 5.5 — this is the sweet spot for 1.5B models
        # Above 5.5 → our model gets <10% → reward sparsity → dead gradient
        ds = ds.filter(
            lambda x: x.get("difficulty", 10.0) is not None and
                      float(x.get("difficulty", 10.0)) <= 5.5
        )
        print(f"  DeepMath after difficulty filter: {len(ds):,}")
    except Exception as e:
        print(f"  ⚠️  DeepMath load failed: {e}")
        return Dataset.from_dict(
            {"prompt": [], "ground_truth": [], "problem_id": [], "source": []}
        )

    ds = ds.shuffle(seed=42).select(range(min(20 if sanity else n, len(ds))))

    def process(ex):
        q   = ex.get("question", "")
        ans = str(ex.get("final_answer", "")).strip()
        return {
            "prompt":       make_prompt(q),
            "ground_truth": ans,
            "problem_id":   hashlib.md5(q.encode()).hexdigest()[:12],
            "source":       "deepmath",
        }

    ds = ds.map(process, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: x["ground_truth"] and
                              x["ground_truth"] not in ["", "None", "True", "False"])
    print(f"  ✅ DeepMath easy: {len(ds):,}")
    return ds


def build_dataset(args) -> Dataset:
    """Combine all sources, shuffle, print stats."""
    sanity = args.sanity
    print("\nBuilding Phase D dataset...")

    math_ds     = load_math_l3l4(args.n_math, sanity)
    metamath_ds = load_metamath_math(args.n_metamath, sanity)
    deepmath_ds = load_deepmath_easy(args.n_deepmath, sanity)

    all_ds = concatenate_datasets([math_ds, metamath_ds, deepmath_ds])
    all_ds = all_ds.shuffle(seed=42)

    # Print distribution
    sources = defaultdict(int)
    for ex in all_ds:
        sources[ex["source"]] += 1

    print(f"\n  ── Phase D Dataset ──")
    total = len(all_ds)
    for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {src:20}: {cnt:5,}  ({cnt/total*100:.1f}%)")
    print(f"  {'TOTAL':20}: {total:5,}")
    print(f"\n  Pass rate target : 25-55% per problem")
    print(f"  Why not DeepMath L5-9: <5% pass rate for 1.5B → dead gradient")
    print(f"  Why not GSM8K        : ~80% pass rate → dead gradient (too easy)")
    return all_ds


# ════════════════════════════════════════════════════════════════════════════
# MONITOR CALLBACK
# ════════════════════════════════════════════════════════════════════════════

class GDPOMonitor(TrainerCallback):
    def __init__(self, log_path: Path, check_every: int = 25):
        self.log_path    = log_path
        self.check_every = check_every
        self.history     = []

    def on_step_end(
        self,
        args:    TrainerControl,
        state:   TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        step = state.global_step
        if step % self.check_every != 0 or step == 0:
            return

        n = 200
        def avg(lst):
            tail = lst[-n:] if len(lst) >= n else lst
            return sum(tail) / max(len(tail), 1)

        pass_rate   = avg(_stats["correct"])
        tag_rate    = avg(_stats["has_tag"])
        word_avg    = avg(_stats["word_count"])
        loop_rate   = sum(1 for x in _stats["repeat_penalty"][-n:] if x < 0) / max(n, 1)

        entry = {
            "step":       step,
            "pass_rate":  pass_rate,
            "tag_rate":   tag_rate,
            "mean_len":   word_avg,
            "loop_rate":  loop_rate,
        }
        self.history.append(entry)

        # Print monitor line
        new_best = (len(self.history) == 1 or
                    pass_rate > max(e["pass_rate"] for e in self.history[:-1]))
        best_str = "  ⭐ New best pass@1" if new_best else ""

        print(f"\n  — Step {step} Monitor —")
        print(f"    Pass@1    : {pass_rate*100:.1f}%{best_str}")
        print(f"    Tag rate  : {tag_rate*100:.1f}%  ← termination reward signal")
        print(f"    Loop rate : {loop_rate*100:.1f}%  ← efficiency reward signal")
        print(f"    Avg words : {word_avg:.0f}")

        # Save to json
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump(self.history, f, indent=2)


class MATH500EvalCallback(TrainerCallback):
    """Eval at specific steps on 100 MATH500 problems."""

    def __init__(self, tokenizer, log_path: Path,
                 eval_at: List[int], n_eval: int = 100):
        self.tokenizer = tokenizer
        self.log_path  = log_path
        self.eval_at   = set(eval_at)
        self.n_eval    = n_eval
        self.history   = []
        self._ds       = None

    def _get_eval_ds(self):
        if self._ds is not None:
            return self._ds
        try:
            ds = load_dataset(
                "DigitalLearningGmbH/MATH-lighteval",
                split="test",
                cache_dir=str(DATA_CACHE)
            )
            # Stratified sample: mix of L3-5
            ds = ds.filter(lambda x: x.get("level", "") in
                           ["Level 3", "Level 4", "Level 5"])
            ds = ds.shuffle(seed=99).select(range(min(self.n_eval, len(ds))))
            self._ds = ds
            return ds
        except Exception as e:
            print(f"  ⚠️  MATH500 eval dataset failed: {e}")
            return None

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step not in self.eval_at or model is None:
            return

        ds = self._get_eval_ds()
        if ds is None:
            return

        print(f"\n  ── MATH500 Eval @ step {step} ──")
        model.eval()
        correct = 0
        tok     = self.tokenizer

        for ex in ds:
            answer = extract_boxed(ex.get("solution", "")) or ""
            if not answer:
                continue

            prompt_text = (
                f"<|im_start|>user\n{ex['problem']}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n"
            )
            inputs = tok(prompt_text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens       = 2048,
                    temperature          = 0.1,
                    do_sample            = True,
                    repetition_penalty   = 1.05,
                    pad_token_id         = tok.eos_token_id,
                )
            resp = tok.decode(
                out[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            pred = extract_predicted_answer(resp)
            if verify_answer(pred, answer):
                correct += 1

        acc = correct / len(ds) * 100
        entry = {"step": step, "math500_acc": acc, "n_eval": len(ds)}
        self.history.append(entry)

        print(f"  MATH500 @ step {step}: {acc:.1f}%  ({correct}/{len(ds)})")

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump(self.history, f, indent=2)

        model.train()


# ════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(model_path: str, attn_impl: str):
    model_path = str(Path(model_path).expanduser())
    print(f"\n  Loading : {model_path}")
    print(f"  Attn    : {attn_impl}")
    print(f"  LoRA    : NONE — full parameter (required for termination learning)")

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(Path(model_path).expanduser()),
        dtype         = torch.bfloat16,
        device_map    = {"": 0},
        trust_remote_code = True,
        attn_implementation = attn_impl,
        local_files_only = True,
    )

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params    : {n_total/1e9:.2f}B total")
    print(f"  Trainable : {n_trainable/1e9:.2f}B  (ALL — no LoRA)")

    # Memory estimate
    model_gb  = n_total * 2 / 1e9  # bfloat16
    ref_gb    = model_gb
    opt_gb    = n_trainable * 4 * 3 / 1e9  # AdamW: 2 states × 4 bytes × 3
    act_gb    = 8.0
    total_gb  = model_gb + ref_gb + opt_gb + act_gb
    print(f"\n  Memory estimate:")
    print(f"    Model     : {model_gb:.1f} GB")
    print(f"    Reference : {ref_gb:.1f} GB")
    print(f"    Optimizer : {opt_gb:.1f} GB")
    print(f"    Activations: ~{act_gb:.0f} GB")
    print(f"    Total est : {total_gb:.1f} GB  (available: 48 GB)")

    return model, tok


# ════════════════════════════════════════════════════════════════════════════
# GDPO CONFIG
# ════════════════════════════════════════════════════════════════════════════

def build_gdpo_config(args, output_dir: Path) -> GRPOConfig:
    max_steps = 10 if args.sanity else args.max_steps

    print(f"\n  Phase D GDPO Config:")
    print(f"    Algorithm     : GDPO (normalize_then_sum)")
    print(f"    max_steps     : {max_steps}")
    print(f"    num_gen       : {args.num_gen}")
    print(f"    max_comp_len  : {args.max_comp_len}")
    print(f"    lr            : {args.lr}")
    print(f"    beta (kl_coef): {args.beta}")
    print(f"    Rewards       : [correctness×1.0] + [termination×0.5] + [efficiency×0.3]")
    print(f"    LoRA          : NONE")
    print(f"    clip_higher   : {args.epsilon_high} (DAPO)")

    base_kwargs = dict(
        output_dir       = str(output_dir),
        run_name         = args.run_name,
        max_steps        = max_steps,
        num_train_epochs = 1,

        num_generations           = args.num_gen,
        max_completion_length     = args.max_comp_len,

        learning_rate             = args.lr,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        optim                     = "adamw_torch_fused",
        weight_decay              = 0.01,
        max_grad_norm             = 1.0,
        warmup_steps              = 20,
        lr_scheduler_type         = "cosine",

        beta                      = args.beta,
        temperature               = args.temperature,

        # ── GDPO: normalize each reward separately ──────────────────────
        multi_objective_aggregation = "normalize_then_sum",

        # Reward weights: correctness dominates, others supporting
        # Passed as list matching reward_funcs order
        # [correctness, termination, efficiency]

        bf16                      = True,
        use_vllm                  = False,

        save_strategy             = "steps",
        save_steps                = 5 if args.sanity else 100,
        save_total_limit          = 3,  # 200, 400, 500 — 3×9GB=27GB

        logging_steps             = 2 if args.sanity else 10,
        report_to                 = "wandb" if args.wandb else "none",

        remove_unused_columns     = False,
        seed                      = 42,
        generation_batch_size      = 8,
        reward_weights             = [1.0, 0.5, 0.3],
    )

    # DAPO clip_higher
    import inspect
    def _supported(key):
        return key in inspect.signature(GRPOConfig.__init__).parameters

    for high_key, low_key in [
        ("epsilon_high", "epsilon"),
        ("clip_higher",  None),
    ]:
        if _supported(high_key):
            base_kwargs[high_key] = args.epsilon_high
            if low_key and _supported(low_key):
                base_kwargs[low_key] = args.epsilon_low
            elif _supported("epsilon"):
                base_kwargs["epsilon"] = args.epsilon_low
            print(f"    clip_higher   : {args.epsilon_high} ✅ DAPO")
            break
    else:
        if _supported("epsilon"):
            base_kwargs["epsilon"] = args.epsilon_low

    return GRPOConfig(**base_kwargs)


# ════════════════════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════════════════════

def save_model(model, tokenizer, merged_dir: Path):
    print(f"\n{'─'*60}")
    print("  Saving full parameter model...")
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    print(f"  ✅ Saved → {merged_dir}")
    print(f"     Next: Stage 5 targeted distillation or RFT")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    sanity = args.sanity

    suffix     = "_sanity" if sanity else ""
    output_dir = Path(args.output_dir) / f"{args.run_name}{suffix}"
    merged_dir = Path(args.merged_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "█"*65)
    print("  STAGE 4D — GDPO MULTI-REWARD TRAINING")
    print("  Making the model disciplined, not just correct")
    if sanity:
        print("  *** SANITY RUN — 10 steps ***")
    print("█"*65)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n  Key improvements over Phase C (binary GRPO):")
    print(f"    Phase C: 1 reward (correct/wrong) → model learned to be RIGHT")
    print(f"    Phase D: 3 rewards (GDPO) → model learns to TERMINATE CLEANLY")
    print(f"\n  Why GDPO not GRPO for multiple rewards:")
    print(f"    GRPO: sum rewards THEN normalize → collapses distinct signals")
    print(f"    GDPO: normalize EACH reward THEN sum → preserves all signals")
    print(f"    One TRL config line: multi_objective_aggregation='normalize_then_sum'")
    print(f"\n  Why NOT DeepMath L5-9 for dataset:")
    print(f"    1.5B model pass rate on L5-9: ~2-5% → reward std≈0 → dead gradient")
    print(f"    Using L3-5 range: ~25-50% pass rate → rich gradient signal")

    # ── Load model ──
    model, tokenizer = load_model_and_tokenizer(args.model, args.attn_impl)

    # ── Dataset ──
    dataset = build_dataset(args)
    if sanity:
        dataset = dataset.select(range(min(40, len(dataset))))
    print(f"\n  Training on {len(dataset):,} problems")

    # ── GDPO config ──
    gdpo_cfg = build_gdpo_config(args, output_dir)

    # ── Callbacks ──
    monitor_cb = GDPOMonitor(
        log_path    = LOG_DIR / "stage4d_monitor.json",
        check_every = 2 if sanity else 25,
    )
    eval_cb = MATH500EvalCallback(
        tokenizer = tokenizer,
        log_path  = LOG_DIR / "stage4d_math500_evals.json",
        eval_at   = [] if sanity else [500],  # eval only at end
        n_eval    = 0 if sanity else 100,
    )

    # ── Trainer with 3 separate reward functions ──
    # GDPO requires passing reward functions as a list
    # multi_objective_aggregation="normalize_then_sum" handles the rest
    trainer = GRPOTrainer(
        model            = model,
        args             = gdpo_cfg,
        reward_funcs     = [
            correctness_reward,   # weight 1.0 — primary
            termination_reward,   # weight 0.5 — format/discipline
            efficiency_reward,    # weight 0.3 — anti-loop
        ],
        train_dataset    = dataset,
        processing_class = tokenizer,
        # NO peft_config — full parameter required for termination learning
    )
    trainer.add_callback(monitor_cb)
    trainer.add_callback(eval_cb)

    # ── Train ──
    print(f"\n{'─'*65}")
    print("  Starting GDPO training...")
    print(f"  Monitor: tail -f {LOG_DIR}/stage4d_monitor.json")
    print(f"{'─'*65}\n")

    resume_path = None
    if args.resume:
        # Find latest checkpoint
        ckpt_dir = output_dir
        checkpoints = sorted(
            [d for d in ckpt_dir.iterdir() if d.name.startswith("checkpoint-")],
            key=lambda x: int(x.name.split("-")[1])
        )
        if checkpoints:
            resume_path = str(checkpoints[-1])
            print(f"  Resuming from: {resume_path}")
        else:
            print(f"  ⚠️  No checkpoints found in {ckpt_dir}, starting fresh")

    result = trainer.train(resume_from_checkpoint=resume_path)

    # ── Save metrics ──
    metrics = result.metrics
    with open(LOG_DIR / "stage4d_train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Final stats ──
    n = 500
    def avg(lst):
        tail = lst[-n:] if len(lst) >= n else lst
        return sum(tail) / max(len(tail), 1)

    print(f"\n  Final Training Stats:")
    print(f"    Pass@1    : {avg(_stats['correct'])*100:.1f}%")
    print(f"    Tag rate  : {avg(_stats['has_tag'])*100:.1f}%")
    print(f"    Avg words : {avg(_stats['word_count']):.0f}")
    loop_ct = sum(1 for x in _stats["repeat_penalty"][-n:] if x < 0)
    print(f"    Loop rate : {loop_ct/max(n,1)*100:.1f}%")

    # ── MATH500 trajectory ──
    eval_log = LOG_DIR / "stage4d_math500_evals.json"
    if eval_log.exists():
        with open(eval_log) as f:
            evals = json.load(f)
        if evals:
            print(f"\n  MATH500 trajectory:")
            for e in evals:
                print(f"    Step {e['step']:>4}: {e['math500_acc']:.1f}%")
            best_e = max(evals, key=lambda x: x["math500_acc"])
            print(f"\n  Best: {best_e['math500_acc']:.1f}% @ step {best_e['step']}")
            print(f"  Phase C baseline: ~34% (greedy)")

    # ── Save model ──
    if not args.skip_merge and not sanity:
        save_model(model, tokenizer, merged_dir)
    elif sanity:
        print("\n  [Sanity] Skipping save.")

    # ── Summary ──
    print(f"\n{'█'*65}")
    if sanity:
        print("  SANITY COMPLETE — check:")
        print("  1. Loss not NaN")
        print("  2. All 3 rewards logging (correctness, termination, efficiency)")
        print("  3. Pass@1 > 5%")
        print("  4. reward_std > 0 on some batches")
        print("  5. No OOM  (~26GB needed)")
        print()
        print("  If all good → full run:")
        print(f"  CUDA_VISIBLE_DEVICES=0 python stage4d_gdpo.py "
              f"--run-name gdpo_v1 --wandb")
    else:
        print("  PHASE D COMPLETE")
        print(f"  Saved: {merged_dir}")
        print()
        print("  Expected improvements:")
        print("    Tag rate    : 0% → 60%+  (termination reward fired)")
        print("    Avg words   : 900 → 400-600  (efficiency reward fired)")
        print("    MATH500     : ~34% → ~55-60%  (discipline + correctness)")
        print("    Loop rate   : ~40% → <10%")
        print()
        print("  Next steps:")
        print("    1. Run MATH500 eval with stop_strings=['</solution>']")
        print("    2. If MATH500 > 55% → proceed to RFT (filter own best outputs)")
        print("    3. If MATH500 < 50% → run 200 more steps (extend to 800)")
    print(f"{'█'*65}\n")


if __name__ == "__main__":
    main()
