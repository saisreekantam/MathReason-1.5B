"""
Stage 4B — GRPO Phase B (Termination + Control)
════════════════════════════════════════════════════════════════════════════
Server  : revanth@172.16.192.168
Input   : ~/nlp/checkpoints/stage4a_grpo_merged   (89% reasoning, 56% clean output)
Output  : ~/nlp/checkpoints/stage4b_grpo

Goal    : Close the gap between reasoning capability (89%) and clean output (56%)
          by teaching the model to COMMIT and STOP after finding the answer.

What Phase A did:
  - Binary reward taught the model to find correct answers
  - Result: 89% reasoning capability, 14.5% reported pass@1 (extractor bug)
  - Remaining failures: re-verification spirals, no <solution> tag

What Phase B adds:
  - Termination reward: +0.15 for clean </solution>→STOP, -0.15 for no tag
  - Harder dataset: MATH L3-4, AMC, Olympiad (35-50% pass rate)
  - No length penalty (proven harmful in Phase A v1)
  - No complex multi-component reward (proven harmful in Phase A v2)

Reward design (two components only):
  r_correct : 1.0 (correct) / 0.0 (wrong)
  r_term    : +0.15 (clean stop) / 0.0 (tag present, kept going) / -0.15 (no tag)
  Total range: -0.15 to +1.15

  Four distinct reward levels:
    correct + clean stop  → 1.15  (best — what we want)
    correct + kept going  → 1.00  (ok — but train toward 1.15)
    wrong   + clean stop  → 0.15  (learns format even when wrong)
    wrong   + no tag      → -0.15 (worst — explicit loop penalty)

Dataset (15K problems, difficulty window 35-50% pass rate):
  MATH L3-4        : 6000   formula-heavy, not enumeration
  DeepScaleR AMC   : 5000   competition, good difficulty window
  NuminaMath olymp : 2500   hard end, pushes capability ceiling
  MetaMath MATH    : 1500   prevents regression on structured algebra

Usage:
  # ALWAYS sanity run first
  CUDA_VISIBLE_DEVICES=1 python stage4b_grpo.py --sanity

  # Full run
  CUDA_VISIBLE_DEVICES=1 python stage4b_grpo.py --run-name grpo_phaseB_v1

  # With wandb
  CUDA_VISIBLE_DEVICES=1 python stage4b_grpo.py --wandb --run-name grpo_phaseB_v1

  # Resume
  CUDA_VISIBLE_DEVICES=1 python stage4b_grpo.py --resume --run-name grpo_phaseB_v1
════════════════════════════════════════════════════════════════════════════
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
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
INPUT_MODEL = WORK_DIR / "checkpoints" / "stage4a_grpo_merged"
CKPT_DIR    = WORK_DIR / "checkpoints" / "stage4b_grpo"
MERGED_DIR  = WORK_DIR / "checkpoints" / "stage4b_grpo_merged"
LOG_DIR     = WORK_DIR / "logs"
DATA_CACHE  = WORK_DIR / "data" / "cache"

SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Think step by step inside <think> tags, "
    "then give your final answer inside <solution> tags."
)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 4B — GRPO Phase B")

    # Modes
    p.add_argument("--sanity",         action="store_true",
                   help="Quick sanity: 60 examples, 25 steps")
    p.add_argument("--resume",         action="store_true")
    p.add_argument("--wandb",          action="store_true")
    p.add_argument("--run-name",       type=str,  default="grpo_phaseB_v1")

    # Paths
    p.add_argument("--model",          type=str,  default=str(INPUT_MODEL))
    p.add_argument("--output-dir",     type=str,  default=str(CKPT_DIR))
    p.add_argument("--merged-dir",     type=str,  default=str(MERGED_DIR))

    # LoRA
    p.add_argument("--lora-rank",      type=int,  default=64)
    p.add_argument("--lora-alpha",     type=int,  default=128)

    # GRPO core
    p.add_argument("--num-gen",        type=int,  default=8,
                   help="Completions per prompt (G)")
    p.add_argument("--max-steps",      type=int,  default=600)
    p.add_argument("--lr",             type=float,default=5e-7)
    p.add_argument("--kl-coef",        type=float,default=0.10,
                   help="Slightly lower than Phase A (0.15) — model more stable now")
    p.add_argument("--temperature",    type=float,default=0.8)
    p.add_argument("--max-prompt-len", type=int,  default=512)
    p.add_argument("--max-comp-len",   type=int,  default=2048,
                   help="Increased from Phase A 1500 — harder problems need more room")

    # DAPO clip_higher
    p.add_argument("--epsilon-low",    type=float,default=0.20)
    p.add_argument("--epsilon-high",   type=float,default=0.28,
                   help="DAPO clip_higher — preserves reasoning tokens")

    # Reward weights
    p.add_argument("--w-correct",      type=float,default=1.0)
    p.add_argument("--w-term",         type=float,default=0.15,
                   help="Termination reward weight. Creates 0.30 differential "
                        "between looping (0.85) and clean stop (1.15)")

    # Dataset sizes
    p.add_argument("--n-math-l3l4",   type=int,  default=2000)
    p.add_argument("--n-deepscaler",  type=int,  default=2000)
    p.add_argument("--n-numina",      type=int,  default=0)
    p.add_argument("--n-metamath",    type=int,  default=10000)

    # Misc
    p.add_argument("--skip-eval",      action="store_true")
    p.add_argument("--skip-merge",     action="store_true")
    p.add_argument("--attn-impl",      type=str,  default="sdpa",
                   choices=["sdpa", "flash_attention_2", "eager"])

    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════════
# ANSWER VERIFICATION
# Learned lessons:
#   - \dfrac{5}{14} must normalize to 5/14  (Q9 failure)
#   - multi-value answers: "12 and 8"       (Q3/Q4 failure)
#   - LaTeX units and formatting            (various)
# ════════════════════════════════════════════════════════════════════════════

def extract_boxed(text: str) -> Optional[str]:
    """Extract content from \\boxed{...} handling nested braces."""
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
    """
    Normalize answer for comparison.
    Handles: LaTeX fractions, units, commas, decimals, simple expressions.
    """
    if ans is None:
        return ""
    ans = str(ans).strip()

    # Fix \dfrac{a}{b} and \frac{a}{b} → a/b  (the Q9 bug)
    ans = re.sub(r"\\d?frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", ans)

    # Strip LaTeX formatting
    ans = re.sub(r"\\text\{([^}]*)\}",   r"\1", ans)
    ans = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", ans)
    ans = re.sub(r"\\left|\\right",       "",    ans)
    ans = ans.replace("$", "").strip()

    # Remove common units
    ans = re.sub(
        r"\s*(dollars?|cents?|meters?|km|kg|cm|miles?|feet|inches?|hours?|"
        r"minutes?|seconds?|%|sq\.?\s*\w*)\s*$",
        "", ans, flags=re.IGNORECASE
    ).strip()

    # Remove commas in numbers (1,000 → 1000)
    ans = re.sub(r"(\d),(\d)", r"\1\2", ans)

    # Normalize decimals: trailing zeros
    try:
        f = float(ans)
        if f == int(f):
            return str(int(f))
        return f"{f:.6f}".rstrip("0")
    except (ValueError, OverflowError):
        pass

    return ans.lower().strip()


def try_numeric_equal(a: str, b: str) -> Optional[bool]:
    """Try float comparison."""
    try:
        fa, fb = float(a), float(b)
        return abs(fa - fb) < max(1e-6, 1e-4 * max(abs(fa), abs(fb)))
    except (ValueError, OverflowError):
        return None


def try_fraction_equal(a: str, b: str) -> Optional[bool]:
    """Compare fractions: 3/4 vs 0.75."""
    def to_float(s):
        m = re.match(r"^(-?\d+)\s*/\s*(-?\d+)$", s.strip())
        if m:
            num, den = int(m.group(1)), int(m.group(2))
            return num / den if den != 0 else None
        try:
            return float(s)
        except ValueError:
            return None

    fa, fb = to_float(a), to_float(b)
    if fa is not None and fb is not None:
        return abs(fa - fb) < 1e-5
    return None


def verify_answer(predicted: Optional[str], ground_truth: str) -> bool:
    """
    Robust math answer verification.
    Handles numeric, fraction, normalized string, multi-value answers.
    """
    if predicted is None or ground_truth is None:
        return False

    # Multi-value ground truth: "12 and 8" or "5, -2"
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
        if num_eq is not None and num_eq:
            return True

        frac_eq = try_fraction_equal(pred_norm, gt_norm)
        if frac_eq is not None and frac_eq:
            return True

    # Word answers
    word_map = {"yes": "true", "no": "false", "1": "true", "0": "false"}
    p_word = word_map.get(pred_norm, pred_norm)
    for gt_part in gt_parts:
        g_word = word_map.get(normalize_answer(gt_part), normalize_answer(gt_part))
        if p_word == g_word:
            return True

    return False


def extract_predicted_answer(response: str) -> Optional[str]:
    """
    Smart extractor v2 — priority chain.
    Lesson learned: always scan first 60% for loops that corrupt later text.

    Priority:
      1. <solution>...</solution> tag
      2. \\boxed{} last occurrence
      3. First "So/Therefore = X" in first 60%
      4. Last number in first 60%
      5. Last number in full response (fallback)
    """
    # 1. Solution tag
    sol_m = re.search(r"<solution>(.*?)</solution>", response, re.DOTALL)
    if sol_m:
        content = sol_m.group(1).strip()
        # Check for boxed inside solution
        boxed = extract_boxed(content)
        return boxed if boxed else content

    # 2. Boxed anywhere
    boxed = extract_boxed(response)
    if boxed:
        return boxed

    # 3. First 60% scan
    cutoff = int(len(response) * 0.6)
    early  = response[:cutoff]

    # "So/Therefore/Thus/answer is X = Y"
    m = re.search(
        r"(?:so|therefore|thus|answer is|equals?|result is)"
        r"[^.]*?=\s*(-?[\d,./]+)",
        early, re.IGNORECASE
    )
    if m:
        return m.group(1).replace(",", "")

    # Last number in first 60%
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", early)
    if nums:
        return nums[-1].replace(",", "")

    # 5. Full response fallback
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", response)
    return nums[-1].replace(",", "") if nums else None


def extract_ground_truth(raw_answer: str, source: str) -> str:
    """Extract clean GT from raw dataset answer field."""
    if source == "gsm8k":
        parts = raw_answer.split("####")
        return parts[-1].strip() if len(parts) > 1 else raw_answer.strip()

    if source in ("math", "math_l3l4", "deepscaler"):
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

    if source == "numina":
        boxed = extract_boxed(raw_answer)
        return boxed if boxed else raw_answer.strip()

    return raw_answer.strip()


# ════════════════════════════════════════════════════════════════════════════
# REWARD SYSTEM — Phase B
# Two components only. Correctness dominant.
# ════════════════════════════════════════════════════════════════════════════

class PhaseBreward:
    """
    Phase B reward: binary correctness + termination signal.

    Key design decisions vs Phase A:
      - No HAPO (proven harmful — caused 70%→64.5% regression)
      - No length penalty (proven harmful — shortens reasoning pathologically)
      - No anti-loop pattern matching (too brittle, doesn't generalize)
      - No format component (merged into termination)
      - Termination reward creates implicit loop penalty:
          loop → no </solution> → r_term = -0.15
          This is cleaner than detecting loop phrases explicitly

    Reward levels:
      correct + clean stop  → 1.15  (model learns: commit to answer, stop)
      correct + kept going  → 1.00  (ok, but gradient pushes toward 1.15)
      wrong   + clean stop  → 0.15  (learns format even on wrong answers)
      wrong   + no tag      → -0.15 (explicit penalty for loops/no termination)
    """

    def __init__(self, w_correct: float = 1.0, w_term: float = 0.15):
        self.w_correct = w_correct
        self.w_term    = w_term

        # Stats tracking
        self.stats = defaultdict(list)
        self.n_calls = 0

    def _r_correct(self, completion: str, ground_truth: str) -> Tuple[float, bool]:
        predicted  = extract_predicted_answer(completion)
        is_correct = verify_answer(predicted, ground_truth)
        return self.w_correct * (1.0 if is_correct else 0.0), is_correct

    def _r_termination(self, completion: str) -> Tuple[float, str]:
        """
        Termination reward — the core Phase B addition.

        Three states:
          CLEAN: has <solution>X</solution> AND nothing meaningful after it
                 → model committed to answer and stopped   → +w_term
          PARTIAL: has <solution> tag but kept generating after </solution>
                   → model answered but couldn't stop     → 0.0
          NONE: no <solution> tag at all
                → model looped or never reached answer     → -w_term
        """
        has_open  = "<solution>" in completion
        has_close = "</solution>" in completion

        if has_open and has_close:
            after_close = completion.split("</solution>")[-1].strip()
            if len(after_close) < 30:
                return self.w_term, "CLEAN"        # +0.15: committed and stopped
            else:
                return self.w_term * 0.33, "PARTIAL"  # +0.05: tag present, kept going
        else:
            return -self.w_term, "NONE"            # -0.15: looped, never terminated

    def compute(self, completion: str, ground_truth: str) -> float:
        """Compute total reward for a single (completion, ground_truth) pair."""
        r_correct, is_correct  = self._r_correct(completion, ground_truth)
        r_term,    term_status = self._r_termination(completion)
        total = r_correct + r_term

        # Track stats for monitoring
        self.stats["correct"].append(float(is_correct))
        self.stats["r_correct"].append(r_correct)
        self.stats["r_term"].append(r_term)
        self.stats["total"].append(total)
        self.stats["term_clean"].append(float(term_status == "CLEAN"))
        self.stats["term_partial"].append(float(term_status == "PARTIAL"))
        self.stats["term_none"].append(float(term_status == "NONE"))
        self.stats["resp_len"].append(len(completion.split()))
        self.n_calls += 1

        return total

    def get_stats(self, last_n: int = 200) -> Dict:
        def avg(lst):
            tail = lst[-last_n:] if len(lst) >= last_n else lst
            return sum(tail) / max(len(tail), 1)

        return {
            "pass_rate":        avg(self.stats["correct"]),
            "mean_reward":      avg(self.stats["total"]),
            "clean_term_rate":  avg(self.stats["term_clean"]),
            "partial_term_rate":avg(self.stats["term_partial"]),
            "no_term_rate":     avg(self.stats["term_none"]),
            "mean_len":         avg(self.stats["resp_len"]),
            "n_total":          self.n_calls,
        }

    def log_stats(self, step: int):
        s = self.get_stats()
        print(f"\n  [Reward @ step {step}]")
        print(f"    Pass@1           : {s['pass_rate']*100:.1f}%")
        print(f"    Mean reward      : {s['mean_reward']:+.3f}")
        print(f"    Clean term rate  : {s['clean_term_rate']*100:.1f}%  ← KEY METRIC")
        print(f"    Partial term rate: {s['partial_term_rate']*100:.1f}%")
        print(f"    No tag rate      : {s['no_term_rate']*100:.1f}%  ← should drop")
        print(f"    Mean resp words  : {s['mean_len']:.0f}")


# Global reward instance (persists across training steps for stat tracking)
_reward: Optional[PhaseBreward] = None


def make_reward_fn(reward: PhaseBreward):
    """Factory for GRPOTrainer reward function."""
    def reward_fn(
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> List[float]:
        ground_truths = kwargs.get("ground_truth", [""] * len(completions))
        rewards = []
        for comp, gt in zip(completions, ground_truths):
            # trl may pass completions as list of message dicts — extract text
            if isinstance(comp, list):
                # [{"role": "assistant", "content": "..."}]
                comp = " ".join(
                    m.get("content", "") if isinstance(m, dict) else str(m)
                    for m in comp
                )
            elif not isinstance(comp, str):
                comp = str(comp)
            r = reward.compute(comp, str(gt))
            rewards.append(r)

        # Dynamic batch skip: if all same reward (std≈0), zero out gradient
        import torch as _torch
        if len(rewards) > 1:
            std = _torch.tensor(rewards).float().std().item()
            if std < 0.05:
                return [0.0] * len(rewards)

        return rewards

    return reward_fn


# ════════════════════════════════════════════════════════════════════════════
# DATASET LOADING
# Phase B dataset design:
#   - Difficulty window: 35-50% pass rate (model at ~65% on Phase A data)
#   - Formula-heavy problems (not enumeration-prone)
#   - No GSM8K (model at 85%+, dead signal)
#   - No MATH L1-2 (model at 75%+, mostly dead)
# ════════════════════════════════════════════════════════════════════════════

def problem_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:16]


def make_prompt(problem: str) -> List[Dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": problem.strip()},
    ]


def load_math_l3l4(n: int, sanity: bool) -> Dataset:
    """MATH Level 3-4 — formula-heavy, ideal difficulty window."""
    print("  Loading MATH L3-4...")
    try:
        ds = load_dataset(
            "DigitalLearningGmbH/MATH-lighteval",
            split="train",
            cache_dir=str(DATA_CACHE)
        )
    except Exception as e:
        print(f"  ⚠️  MATH load failed: {e}")
        return Dataset.from_dict(
            {"prompt": [], "ground_truth": [], "problem_id": [], "source": []}
        )

    try:
        ds = ds.filter(lambda x: x.get("level", "") in ["Level 3", "Level 4"])
    except Exception:
        pass

    ds = ds.select(range(min(20 if sanity else n, len(ds))))

    def process(ex):
        prob = ex.get("problem", "")
        sol  = ex.get("solution", "")   # DigitalLearningGmbH: answer in oxed{} inside solution
        gt   = extract_ground_truth(sol, "math_l3l4")
        return {
            "prompt":       make_prompt(prob),
            "ground_truth": gt,
            "problem_id":   problem_hash(prob),
            "source":       "math_l3l4",
        }

    ds = ds.map(process, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: 0 < len(str(x["ground_truth"])) < 150)
    print(f"  ✅ MATH L3-4: {len(ds):,}")
    return ds


def load_deepscaler_amc(n: int, sanity: bool) -> Dataset:
    """DeepScaleR AMC10/AMC12 — competition problems, good difficulty window."""
    print("  Loading DeepScaleR AMC...")
    local = WORK_DIR / "data" / "deepscaler"
    try:
        if local.exists():
            ds = load_dataset(str(local), split="train")
        else:
            ds = load_dataset(
                "agentica-org/DeepScaleR-Preview-Dataset",
                cache_dir=str(DATA_CACHE)
            )
            if hasattr(ds, "keys"):
                ds = ds["train"] if "train" in ds else list(ds.values())[0]
    except Exception as e:
        print(f"  ⚠️  DeepScaleR load failed: {e}")
        print(f"     Download: huggingface-cli download agentica-org/DeepScaleR-Preview-Dataset "
              f"--repo-type dataset --local-dir ~/nlp/data/deepscaler")
        return Dataset.from_dict(
            {"prompt": [], "ground_truth": [], "problem_id": [], "source": []}
        )



    ds = ds.select(range(min(15 if sanity else n, len(ds))))

    def process(ex):
        prob   = ex.get("problem", "")
        # DeepScaleR: answer is the clean answer, solution is full working
        raw_gt = ex.get("answer", "")
        gt     = str(raw_gt).strip()
        return {
            "prompt":       make_prompt(prob),
            "ground_truth": gt,
            "problem_id":   problem_hash(prob),
            "source":       "deepscaler",
        }

    ds = ds.map(process, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: 0 < len(str(x["ground_truth"])) < 200)
    print(f"  ✅ DeepScaleR: {len(ds):,}")
    return ds


def load_numina_olympiad(n: int, sanity: bool) -> Dataset:
    """NuminaMath olympiad/AMC-AIME — hard end, pushes capability ceiling."""
    print("  Loading NuminaMath olympiad...")
    try:
        ds = load_dataset(
            "AI-MO/NuminaMath-CoT",
            split="train",
            cache_dir=str(DATA_CACHE)
        )
    except Exception as e:
        print(f"  ⚠️  NuminaMath load failed: {e}")
        return Dataset.from_dict(
            {"prompt": [], "ground_truth": [], "problem_id": [], "source": []}
        )

    try:
        ds = ds.filter(lambda x: x.get("source", "") in
                       ["olympiads", "amc_aime", "aops_forum"])
    except Exception:
        pass

    ds = ds.select(range(min(10 if sanity else n, len(ds))))

    def process(ex):
        prob = ex.get("problem", ex.get("question", ""))
        sol  = ex.get("solution", ex.get("answer", ""))
        gt   = extract_ground_truth(sol, "numina")
        return {
            "prompt":       make_prompt(prob),
            "ground_truth": gt,
            "problem_id":   problem_hash(prob),
            "source":       "numina",
        }

    ds = ds.map(process, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: 0 < len(str(x["ground_truth"])) < 200)
    print(f"  ✅ NuminaMath olympiad: {len(ds):,}")
    return ds


def load_metamath_math_style(n: int, sanity: bool) -> Dataset:
    """
    MetaMath MATH-style subset only — prevents regression on structured algebra.
    NOT GSM8K style (model at 85%+ on those, dead signal).
    """
    print("  Loading MetaMath MATH-style subset...")
    try:
        ds = load_dataset(
            "meta-math/MetaMathQA",
            split="train",
            cache_dir=str(DATA_CACHE)
        )
    except Exception as e:
        print(f"  ⚠️  MetaMath load failed: {e}")
        return Dataset.from_dict(
            {"prompt": [], "ground_truth": [], "problem_id": [], "source": []}
        )

    # MATH-style only — excludes GSM8K rephrased (too easy now)
    try:
        ds = ds.filter(lambda x: x.get("type", "") in [
            "MATH_AnsAug", "MATH_Rephrased", "MATH_FOBAR", "MATH_SV"
        ])
    except Exception:
        pass

    ds = ds.select(range(min(15 if sanity else n, len(ds))))

    def process(ex):
        prob   = ex.get("query", "")
        raw_gt = ex.get("response", "")
        gt     = extract_ground_truth(raw_gt, "metamath")
        return {
            "prompt":       make_prompt(prob),
            "ground_truth": gt,
            "problem_id":   problem_hash(prob),
            "source":       "metamath_math",
        }

    ds = ds.map(process, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: 0 < len(str(x["ground_truth"])) < 150)
    print(f"  ✅ MetaMath MATH-style: {len(ds):,}")
    return ds


def build_dataset(args) -> Dataset:
    """
    Build Phase B dataset.
    Sanity: 60 examples total
    Full:   ~15,000 examples

    Difficulty shift vs Phase A:
      Phase A: MetaMath 55%, MATH L2-3 17%, Numina 12%, gsm8k_hard 9%, MATH L3-4 7%
      Phase B: MATH L3-4 40%, DeepScaleR AMC 33%, Numina 17%, MetaMath MATH 10%

    Expected batch skip rate: <20% (down from Phase A's 44%)
    """
    print("\nBuilding Phase B dataset...")
    sanity = args.sanity

    parts = []

    math  = load_math_l3l4(args.n_math_l3l4, sanity)
    parts.append(math)

    deep  = load_deepscaler_amc(args.n_deepscaler, sanity)
    parts.append(deep)

    numi  = load_numina_olympiad(args.n_numina, sanity)
    parts.append(numi)

    meta  = load_metamath_math_style(args.n_metamath, sanity)
    parts.append(meta)

    parts = [p for p in parts if len(p) > 0]
    if not parts:
        raise ValueError("All datasets failed to load.")

    combined = concatenate_datasets(parts)
    combined = combined.filter(lambda x: 0 < len(str(x["ground_truth"])) < 200)
    combined = combined.shuffle(seed=42)

    print(f"\n  ── Phase B Dataset ──")
    print(f"  Total: {len(combined):,} problems")
    sources = Counter(combined["source"])
    for src, cnt in sources.most_common():
        print(f"  {src:<20}: {cnt:>5,}  ({cnt/len(combined)*100:.1f}%)")
    print(f"  Expected pass rate: 35-50%")
    print(f"  Expected batch skip: <20% (was 44% in Phase A)")

    return combined


# ════════════════════════════════════════════════════════════════════════════
# MONITORING CALLBACKS
# ════════════════════════════════════════════════════════════════════════════

class PhaseBMonitorCallback(TrainerCallback):
    """
    Phase B specific monitoring.
    Primary metric: clean_term_rate (should rise from ~40% to 80%+)
    Secondary: pass@1, batch_skip_rate, response length
    """

    def __init__(
        self,
        reward: PhaseBreward,
        log_path: Path,
        check_every: int = 25,
    ):
        self.reward      = reward
        self.log_path    = log_path
        self.check_every = check_every
        self.history     = []
        self.best_clean  = 0.0
        self.best_pass   = 0.0

    def on_step_end(
        self, args, state: TrainerState,
        control: TrainerControl, **kwargs
    ):
        step = state.global_step
        if step % self.check_every != 0 or step == 0:
            return control

        stats = self.reward.get_stats(last_n=self.check_every * 8)

        entry = {
            "step":             step,
            "pass_rate":        stats["pass_rate"],
            "clean_term_rate":  stats["clean_term_rate"],
            "no_term_rate":     stats["no_term_rate"],
            "mean_reward":      stats["mean_reward"],
            "mean_len":         stats["mean_len"],
            "timestamp":        datetime.now().isoformat(),
        }
        self.history.append(entry)

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump(self.history, f, indent=2)

        # ── Print summary ──
        print(f"\n  ── Step {step} Monitor ──")
        print(f"    Pass@1           : {stats['pass_rate']*100:.1f}%")
        print(f"    Clean term rate  : {stats['clean_term_rate']*100:.1f}%  "
              f"← RISING? (was ~40% Phase A end)")
        print(f"    No tag rate      : {stats['no_term_rate']*100:.1f}%  "
              f"← FALLING? (was ~40% Phase A end)")
        print(f"    Mean reward      : {stats['mean_reward']:+.3f}")
        print(f"    Mean resp words  : {stats['mean_len']:.0f}  "
              f"← FALLING? (was ~800 Phase A end)")

        if stats["clean_term_rate"] > self.best_clean:
            self.best_clean = stats["clean_term_rate"]
            print(f"    ⭐ New best clean_term: {self.best_clean*100:.1f}%")

        if stats["pass_rate"] > self.best_pass:
            self.best_pass = stats["pass_rate"]
            print(f"    ⭐ New best pass@1: {self.best_pass*100:.1f}%")

        # ── Health checks ──
        if step > 100 and stats["clean_term_rate"] < 0.45:
            print(f"    ⚠️  Clean term not improving after step 100")
            print(f"       Check: is termination reward firing?")
            print(f"       Action: run manual eval on current checkpoint")

        if step > 50 and stats["pass_rate"] < 0.08:
            print(f"    ⚠️  Pass@1 very low (<8%) — dataset may be too hard")
            print(f"       Action: increase --n-math-l3l4, reduce --n-numina")

        return control

    def on_train_end(self, args, state, control, **kwargs):
        print(f"\n{'═'*55}")
        print(f"  PHASE B TRAINING COMPLETE")
        print(f"  Best clean_term_rate : {self.best_clean*100:.1f}%")
        print(f"  Best pass@1          : {self.best_pass*100:.1f}%")
        print(f"  Target               : clean_term >80%, pass@1 >18%")
        return control


class MATH500EvalCallback(TrainerCallback):
    """
    Runs MATH500 eval at step 300 and 800 only.
    (Not every 100 steps — wastes GPU time, returns misleading numbers early)
    """

    def __init__(
        self,
        tokenizer,
        log_path: Path,
        eval_at: List[int] = None,
        n_eval: int = 100,
    ):
        self.tokenizer = tokenizer
        self.log_path  = log_path
        self.eval_at   = eval_at or [300, 600]
        self.n_eval    = n_eval
        self.history   = []
        self.best_acc  = 0.0

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step not in self.eval_at:
            return control
        print(f"\n{'─'*50}")
        print(f"  MATH500 eval @ step {step}...")
        acc = self._run_eval(model, step)
        if acc > self.best_acc:
            self.best_acc = acc
            print(f"  ⭐ New best MATH500: {acc:.1f}%")
        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):
        print(f"\n  FINAL MATH500 eval...")
        self._run_eval(model, state.global_step, n_override=150)
        return control

    def _run_eval(self, model, step: int, n_override: int = None) -> float:
        n = n_override or self.n_eval
        try:
            ds = load_dataset(
                "DigitalLearningGmbH/MATH-lighteval",
                split="test",
                cache_dir=str(DATA_CACHE)
            )
            ds = ds.select(range(min(n, len(ds))))
        except Exception as e:
            print(f"  ⚠️  MATH eval failed: {e}")
            return -1.0

        model.eval()
        device  = next(model.parameters()).device
        correct = 0

        for i, item in enumerate(ds):
            prob = item.get("problem", item.get("question", ""))
            sol  = item.get("solution", item.get("answer", ""))
            gt   = extract_ground_truth(sol, "math_l3l4")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prob},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            enc = self.tokenizer(
                prompt, return_tensors="pt",
                truncation=True, max_length=512
            ).to(device)

            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=2000,   # enough for smart extractor
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            resp      = self.tokenizer.decode(
                out[0][enc["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            predicted = extract_predicted_answer(resp)
            if verify_answer(predicted, gt):
                correct += 1

            if (i + 1) % 50 == 0:
                print(f"    [{i+1:>3}/{n}]  {correct}/{i+1} = {correct/(i+1)*100:.1f}%")

        acc = correct / n * 100
        print(f"  MATH500 @ step {step}: {correct}/{n} = {acc:.1f}%")
        print(f"  (Stage 3 baseline: ~55% | Phase A target: ~62%+ | Phase B target: ~68%+)")

        entry = {"step": step, "math500_acc": acc, "n": n,
                 "timestamp": datetime.now().isoformat()}
        self.history.append(entry)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump(self.history, f, indent=2)

        model.train()
        return acc


# ════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(model_path: str, attn_impl: str):
    print(f"\n{'─'*60}")
    print(f"  Loading: {model_path}")
    print(f"  Attn   : {attn_impl}")

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run the merge first: python merge_checkpoint.py"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",   # LEFT padding for generation
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map={"": 0},
        trust_remote_code=True,
    )

    params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Params : {params:.2f}B")
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
    return model


# ════════════════════════════════════════════════════════════════════════════
# GRPO CONFIG
# ════════════════════════════════════════════════════════════════════════════

def build_grpo_config(args, output_dir: Path) -> GRPOConfig:
    if args.wandb:
        os.environ["WANDB_PROJECT"] = "mathReason-1.5B"

    max_steps = 25 if args.sanity else args.max_steps

    print(f"\n  Phase B GRPO Config:")
    print(f"    max_steps          : {max_steps}")
    print(f"    num_generations    : {args.num_gen}")
    print(f"    max_comp_len       : {args.max_comp_len}  (↑ from Phase A 1500)")
    print(f"    lr                 : {args.lr}")
    print(f"    kl_coef            : {args.kl_coef}  (↓ from Phase A 0.15)")
    print(f"    epsilon_low/high   : {args.epsilon_low}/{args.epsilon_high}  (DAPO)")
    print(f"    reward             : r_correct(1.0) + r_term(±{args.w_term})")

    base_kwargs = dict(
        output_dir=str(output_dir),
        run_name=args.run_name,

        max_steps=max_steps,
        num_train_epochs=1,

        num_generations=args.num_gen,
        max_completion_length=args.max_comp_len,

        learning_rate=args.lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        optim="adamw_torch_fused",
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_steps=20,
        lr_scheduler_type="cosine",

        beta=args.kl_coef,
        temperature=args.temperature,

        bf16=True,
        use_vllm=False,

        save_strategy="steps",
        save_steps=50 if args.sanity else 100,
        save_total_limit=6,   # keep more checkpoints — Phase B is short

        logging_steps=5 if args.sanity else 10,
        report_to="wandb" if args.wandb else "none",

        remove_unused_columns=False,
        seed=42,
    )

    # Try DAPO clip_higher — graceful fallback
    def _supported(key):
        import inspect
        return key in inspect.signature(GRPOConfig.__init__).parameters

    clip_set = False
    for high_key, low_key in [
        ("clip_higher",    None),
        ("epsilon_high",   "epsilon_low"),
        ("ratio_clip_max", "ratio_clip_min"),
    ]:
        if _supported(high_key):
            base_kwargs[high_key] = args.epsilon_high
            if low_key and _supported(low_key):
                base_kwargs["epsilon"] = args.epsilon_low
            print(f"    clip_higher ({high_key}): {args.epsilon_high} ✅ DAPO")
            clip_set = True
            break

    if not clip_set:
        if _supported("epsilon"):
            base_kwargs["epsilon"] = args.epsilon_low
            print(f"    epsilon (symmetric)  : {args.epsilon_low}  "
                  f"(clip_higher not in this trl version)")

    cfg = GRPOConfig(**base_kwargs)
    print(f"    entropy_coef         : (not in trl 0.29.0 — OK)")

    return cfg


# ════════════════════════════════════════════════════════════════════════════
# MERGE
# ════════════════════════════════════════════════════════════════════════════

def merge_and_save(model, tokenizer, merged_dir: Path):
    print(f"\n{'─'*60}")
    print("  Merging LoRA → base model...")
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    print(f"  ✅ Merged → {merged_dir}")
    print(f"     Next: Stage 5 Gap-Filling SFT")
    return merged


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
    print("  STAGE 4B — GRPO PHASE B (TERMINATION + CONTROL)")
    if sanity:
        print("  *** SANITY RUN — 60 examples, 25 steps ***")
    print("█"*65)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Input  : {args.model}")
    print(f"  Output : {output_dir}")
    print(f"\n  What Phase B fixes:")
    print(f"    Re-verification spirals    → termination reward -0.15 for no tag")
    print(f"    Model loops after answer   → gradient pushes toward clean stop")
    print(f"    Dead gradient (44% skip)   → harder dataset, better signal")
    print(f"\n  What Phase B does NOT change:")
    print(f"    Reasoning capability (89%) → no additional capability training")
    print(f"    Binary correctness reward  → still the primary signal")
    print(f"    KL protection              → kl_coef=0.10, stays close to Phase A")

    # ── Load model ──
    model, tokenizer = load_model_and_tokenizer(args.model, args.attn_impl)

    # ── Apply LoRA ──
    model = apply_lora(model, args.lora_rank, args.lora_alpha)

    # ── Reward ──
    global _reward
    _reward   = PhaseBreward(w_correct=args.w_correct, w_term=args.w_term)
    reward_fn = make_reward_fn(_reward)

    # ── Dataset ──
    dataset = build_dataset(args)
    if sanity:
        dataset = dataset.select(range(min(60, len(dataset))))
    print(f"\n  Training on {len(dataset):,} problems")

    # ── GRPO config ──
    grpo_cfg = build_grpo_config(args, output_dir)

    # ── Callbacks ──
    monitor_cb = PhaseBMonitorCallback(
        reward     = _reward,
        log_path   = LOG_DIR / "stage4b_monitor.json",
        check_every= 5 if sanity else 25,
    )
    eval_cb = MATH500EvalCallback(
        tokenizer = tokenizer,
        log_path  = LOG_DIR / "stage4b_math500_evals.json",
        eval_at   = [10, 25] if sanity else [300, 600],
        n_eval    = 20 if sanity else 100,
    )
    callbacks = [monitor_cb, eval_cb]

    # ── GRPOTrainer ──
    trainer = GRPOTrainer(
        model            = model,
        args             = grpo_cfg,
        reward_funcs     = [reward_fn],
        train_dataset    = dataset,
        processing_class = tokenizer,
    )
    for cb in callbacks:
        trainer.add_callback(cb)

    # ── Train ──
    print(f"\n{'─'*65}")
    print("  Starting Phase B training...")
    print(f"  Monitor: tail -f {LOG_DIR}/stage4b_monitor.json")
    print(f"{'─'*65}\n")

    result = trainer.train(
        resume_from_checkpoint=str(output_dir) if args.resume else None
    )

    # ── Save ──
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = result.metrics
    with open(LOG_DIR / "stage4b_train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  ✅ LoRA adapter   : {output_dir}")
    print(f"  ✅ Train metrics  : {LOG_DIR}/stage4b_train_metrics.json")

    # ── Final reward stats ──
    _reward.log_stats(step=grpo_cfg.max_steps)

    # ── Merge ──
    if not args.skip_merge and not sanity:
        merged_model = merge_and_save(model, tokenizer, merged_dir)
    else:
        if sanity:
            print("\n  [Sanity] Skipping merge.")

    # ── Final summary ──
    print(f"\n{'█'*65}")
    if sanity:
        print("  SANITY COMPLETE — check:")
        print("  1. Loss decreasing (not NaN)")
        print("  2. clean_term_rate > 0% in monitor")
        print("  3. Reward not constant (std > 0.05 on some batches)")
        print("  4. No OOM errors")
        print()
        print("  If all good → launch full run:")
        print(f"  CUDA_VISIBLE_DEVICES=1 python stage4b_grpo.py "
              f"--run-name grpo_phaseB_v1")
    else:
        print("  PHASE B COMPLETE")
        print("  ─────────────────────────────────────────────")
        print(f"  LoRA adapter  : {output_dir}")
        print(f"  Merged model  : {merged_dir}")
        print()

        # Print eval history
        eval_log = LOG_DIR / "stage4b_math500_evals.json"
        if eval_log.exists():
            with open(eval_log) as f:
                evals = json.load(f)
            if evals:
                print(f"  MATH500 trajectory:")
                for e in evals:
                    print(f"    Step {e['step']:>3}: {e['math500_acc']:.1f}%")
                best_e = max(evals, key=lambda x: x["math500_acc"])
                print(f"  Best MATH500: {best_e['math500_acc']:.1f}% @ step {best_e['step']}")

        print()
        print("  Next steps:")
        print("    1. Run diag script on stage4b_grpo_merged")
        print("    2. Stage 5 — Gap-filling SFT (combinatorics + probability)")
        print("    3. Stage 6 — Elastic Reasoning")
        print("    4. Stage 7 — DPO Alignment")
        print("    5. Test-time: Maj@8 + PRM scoring")

    print(f"{'█'*65}\n")


if __name__ == "__main__":
    main()
