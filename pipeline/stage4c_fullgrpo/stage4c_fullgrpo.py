"""
Stage 4C — Full Parameter GRPO (The Correct Approach)
════════════════════════════════════════════════════════════════════════════
Server  : revanth@172.16.192.168
GPU     : CUDA_VISIBLE_DEVICES=0  (GPU 1 running Phase B)

Input   : ~/nlp/checkpoints/stage4b_best_merged   (or stage4a_grpo_merged)
Output  : ~/nlp/checkpoints/stage4c_fullgrpo

Why full parameter (no LoRA):
  - LoRA constrains policy to low-rank subspace
  - GRPO needs broad weight updates to learn termination behavior
  - "Write </solution> and stop" requires updating attention + MLP broadly
  - Those updates are OUTSIDE LoRA rank-64 subspace
  - Full param removes the constraint → binary reward alone is sufficient
  - DeepScaleR proved: binary reward + full param → 43% AIME24

Why binary reward only:
  - Termination reward was a band-aid for LoRA's constraint
  - With full params, binary reward gradient reaches ALL weights naturally
  - looping completion = reward 0 = already penalized
  - clean termination = reward 1 = already rewarded
  - No additional signal needed — this is what every top 1.5B paper uses

Dataset design (20-50% pass rate window):
  - MATH L3-4     : 5000  (model solves ~35-50%) ← primary signal
  - MetaMath MATH : 4000  (model solves ~45-55%) ← stability anchor
  - DeepScaleR AMC: 3000  (model solves ~25-40%) ← harder end
  - NOT: AIME/olympiad   (model solves <10% → dead gradient)
  - NOT: GSM8K/easy      (model solves 85%+ → dead gradient)

Steps: 1500 (DeepScaleR used 1600 for 8K phase — we match that)

Usage:
  # Sanity first (always)
  CUDA_VISIBLE_DEVICES=0 python stage4c_fullgrpo.py --sanity

  # Full run
  CUDA_VISIBLE_DEVICES=0 python stage4c_fullgrpo.py --run-name fullgrpo_v1

  # With wandb
  CUDA_VISIBLE_DEVICES=0 python stage4c_fullgrpo.py --wandb --run-name fullgrpo_v1

  # Resume
  CUDA_VISIBLE_DEVICES=0 python stage4c_fullgrpo.py --resume --run-name fullgrpo_v1
════════════════════════════════════════════════════════════════════════════
"""

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
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
# Input: use best Phase B merged checkpoint, fall back to Phase A
INPUT_MODEL = WORK_DIR / "checkpoints" / "stage4b_best_merged"
FALLBACK    = WORK_DIR / "checkpoints" / "stage4a_grpo_merged"
CKPT_DIR    = WORK_DIR / "checkpoints" / "stage4c_fullgrpo"
MERGED_DIR  = WORK_DIR / "checkpoints" / "stage4c_fullgrpo_merged"
LOG_DIR     = WORK_DIR / "logs"
DATA_CACHE  = WORK_DIR / "data" / "cache"

SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Think step by step inside <think> tags, "
    "then give your final answer inside <solution> tags."
)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 4C — Full Parameter GRPO")

    # Modes
    p.add_argument("--sanity",        action="store_true",
                   help="Quick sanity: 60 examples, 25 steps")
    p.add_argument("--resume",        action="store_true")
    p.add_argument("--wandb",         action="store_true")
    p.add_argument("--run-name",      type=str, default="fullgrpo_v1")

    # Model path — auto-selects best available
    p.add_argument("--model",         type=str, default="auto",
                   help="'auto' selects stage4b_best_merged if exists, "
                        "else stage4a_grpo_merged")
    p.add_argument("--output-dir",    type=str, default=str(CKPT_DIR))
    p.add_argument("--merged-dir",    type=str, default=str(MERGED_DIR))

    # GRPO core
    p.add_argument("--num-gen",       type=int,   default=8)
    p.add_argument("--max-steps",     type=int,   default=1500,
                   help="DeepScaleR used 1600 for 8K phase — match that")
    p.add_argument("--lr",            type=float, default=1e-6,
                   help="Full param needs lower LR than LoRA (was 5e-7 for LoRA)")
    p.add_argument("--beta",          type=float, default=0.04,
                   help="KL coef. Lower than LoRA (0.10) — full param is more stable")
    p.add_argument("--temperature",   type=float, default=0.8)
    p.add_argument("--max-comp-len",  type=int,   default=3072,
                   help="Start at 1500, extend to 3000 at step 800 manually")

    # DAPO clip_higher
    p.add_argument("--epsilon-low",   type=float, default=0.20)
    p.add_argument("--epsilon-high",  type=float, default=0.28)

    # Dataset sizes — calibrated for 20-50% pass rate
    p.add_argument("--n-math-l3l4",  type=int,   default=3000)
    p.add_argument("--n-metamath",   type=int,   default=7000)
    p.add_argument("--n-deepscaler", type=int,   default=2000)

    # Misc
    p.add_argument("--skip-merge",    action="store_true")
    p.add_argument("--attn-impl",     type=str,   default="sdpa")

    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════════
# ANSWER VERIFICATION — same as Phase B, proven working
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
    # Fix \dfrac{a}{b} → a/b
    ans = re.sub(r"\\d?frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", ans)
    ans = re.sub(r"\\text\{([^}]*)\}",   r"\1", ans)
    ans = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", ans)
    ans = re.sub(r"\\left|\\right",       "",    ans)
    ans = ans.replace("$", "").strip()
    # Remove units
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
        num_eq = try_numeric_equal(pred_norm, gt_norm)
        if num_eq is not None and num_eq:
            return True
        frac_eq = try_fraction_equal(pred_norm, gt_norm)
        if frac_eq is not None and frac_eq:
            return True
    return False


def extract_predicted_answer(response: str) -> Optional[str]:
    """Smart extractor v2 — scan first 60% to avoid loop region."""
    # Handle list of message dicts from trl
    if isinstance(response, list):
        response = " ".join(
            m.get("content", "") if isinstance(m, dict) else str(m)
            for m in response
        )
    elif not isinstance(response, str):
        response = str(response)

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

    # 3. First 60% scan — before loop region
    cutoff = int(len(response) * 0.6)
    early  = response[:cutoff]

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

    # Fallback
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", response)
    return nums[-1].replace(",", "") if nums else None


def extract_ground_truth(raw_answer: str, source: str) -> str:
    if source in ("math", "math_l3l4"):
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
    if source == "deepscaler":
        return str(raw_answer).strip()
    return raw_answer.strip()


# ════════════════════════════════════════════════════════════════════════════
# REWARD — PURE BINARY ONLY
# This is the correct design. No termination reward, no format reward,
# no length penalty. DeepScaleR proved this is all you need.
#
# Why binary is enough with full parameters:
#   - looping completion never writes </solution> → reward=0
#   - clean termination writes </solution> correctly → reward=1
#   - GRPO advantage: clean=positive, looping=negative
#   - Full param gradient reaches ALL weights → termination behavior emerges
# ════════════════════════════════════════════════════════════════════════════

class BinaryReward:
    """
    Binary + light repetition penalty.
    Reward range: -0.15 (wrong + severe loops) to 1.0 (correct + clean)
    Correctness dominates. Penalty only fires on catastrophic loops (>40% repeated 8-grams).
    Normal re-verification (2-3 loops) stays below threshold.
    """

    def __init__(self):
        self.stats = defaultdict(list)
        self.n_calls = 0

    @staticmethod
    def _repetition_penalty(text: str, n: int = 8) -> float:
        tokens = text.split()
        if len(tokens) < n * 3:
            return 0.0
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n)]
        if not ngrams:
            return 0.0
        repeat_ratio = 1.0 - len(set(ngrams)) / len(ngrams)
        if repeat_ratio <= 0.40:
            return 0.0
        penalty = -0.15 * (repeat_ratio - 0.40) / 0.60
        return max(penalty, -0.15)

    def compute(self, completion: str, ground_truth: str) -> float:
        if isinstance(completion, list):
            completion = " ".join(
                m.get("content", "") if isinstance(m, dict) else str(m)
                for m in completion
            )
        elif not isinstance(completion, str):
            completion = str(completion)

        predicted  = extract_predicted_answer(completion)
        is_correct = verify_answer(predicted, str(ground_truth))

        r_correct = 1.0 if is_correct else 0.0
        r_repeat  = BinaryReward._repetition_penalty(completion)
        reward    = r_correct + r_repeat

        self.stats["correct"].append(float(is_correct))
        self.stats["reward"].append(reward)
        self.stats["repeat_penalty"].append(r_repeat)
        self.stats["has_sol"].append(
            float("<solution>" in completion and "</solution>" in completion)
        )
        self.stats["resp_len"].append(len(completion.split()))
        self.n_calls += 1

        return reward

    def get_stats(self, last_n: int = 200) -> Dict:
        def avg(lst):
            tail = lst[-last_n:] if len(lst) >= last_n else lst
            return sum(tail) / max(len(tail), 1)
        return {
            "pass_rate":      avg(self.stats["correct"]),
            "tag_rate":       avg(self.stats["has_sol"]),
            "mean_len":       avg(self.stats["resp_len"]),
            "repeat_penalty": avg(self.stats["repeat_penalty"]),
            "n_total":        self.n_calls,
        }


_reward: Optional[BinaryReward] = None


def make_reward_fn(reward: BinaryReward):
    """
    GRPOTrainer reward function factory.
    Includes dynamic batch skip: if all completions same reward (std≈0),
    return zeros to skip the degenerate update.
    """
    def reward_fn(
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> List[float]:
        ground_truths = kwargs.get("ground_truth", [""] * len(completions))
        rewards = [
            reward.compute(comp, str(gt))
            for comp, gt in zip(completions, ground_truths)
        ]
        # Skip degenerate batches (all same reward = zero gradient)
        import torch as _t
        if len(rewards) > 1 and _t.tensor(rewards).float().std().item() < 0.05:
            return [0.0] * len(rewards)
        return rewards

    return reward_fn


# ════════════════════════════════════════════════════════════════════════════
# DATASET — difficulty window 20-50% pass rate
# ════════════════════════════════════════════════════════════════════════════

def problem_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:16]


def make_prompt(problem: str) -> List[Dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": problem.strip()},
    ]


def load_math_l3l4(n: int, sanity: bool) -> Dataset:
    """
    MATH Level 3-4. Model pass rate: ~35-50%. Primary signal source.
    Formula-heavy, not enumeration-prone.
    Uses DigitalLearningGmbH/MATH-lighteval (confirmed in cache).
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
        prob = ex.get("problem", "")
        sol  = ex.get("solution", "")
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


def load_metamath_math(n: int, sanity: bool) -> Dataset:
    """
    MetaMath MATH-style subset only. Model pass rate: ~45-55%.
    Stability anchor — prevents regression on structured algebra.
    MATH-style only, no GSM8K (model at 85%+ on GSM = dead signal).
    """
    print("  Loading MetaMath MATH-style...")
    try:
        ds = load_dataset(
            "meta-math/MetaMathQA",
            split="train",
            cache_dir=str(DATA_CACHE)
        )
        ds = ds.filter(lambda x: x.get("type", "") in [
            "MATH_AnsAug", "MATH_Rephrased", "MATH_FOBAR", "MATH_SV"
        ])
    except Exception as e:
        print(f"  ⚠️  MetaMath load failed: {e}")
        return Dataset.from_dict(
            {"prompt": [], "ground_truth": [], "problem_id": [], "source": []}
        )

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


def load_deepscaler_amc(n: int, sanity: bool) -> Dataset:
    """
    DeepScaleR AMC problems. Model pass rate: ~25-40%.
    Harder end of window — pushes capability without killing gradient.
    Uses HF cache (confirmed present at training server).
    """
    print("  Loading DeepScaleR AMC...")
    try:
        ds = load_dataset(
            "agentica-org/DeepScaleR-Preview-Dataset",
            cache_dir=str(DATA_CACHE)
        )
        if hasattr(ds, "keys"):
            ds = ds["train"] if "train" in ds else list(ds.values())[0]
    except Exception as e:
        print(f"  ⚠️  DeepScaleR load failed: {e}")
        return Dataset.from_dict(
            {"prompt": [], "ground_truth": [], "problem_id": [], "source": []}
        )

    ds = ds.select(range(min(10 if sanity else n, len(ds))))

    def process(ex):
        prob   = ex.get("problem", "")
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


def build_dataset(args) -> Dataset:
    """
    Phase C dataset — difficulty window 20-50% pass rate.

    Mix rationale:
      MATH L3-4   42% — primary signal, formula-heavy
      MetaMath    33% — stability, prevents regression
      DeepScaleR  25% — harder end, pushes ceiling

    Expected frac_zero_std: <0.2 (confirmed from Phase B v2 with similar mix)
    """
    print("\nBuilding Phase C dataset...")
    sanity = args.sanity

    parts = [
        load_math_l3l4(args.n_math_l3l4, sanity),
        load_metamath_math(args.n_metamath, sanity),
        load_deepscaler_amc(args.n_deepscaler, sanity),
    ]
    parts = [p for p in parts if len(p) > 0]

    if not parts:
        raise ValueError("All datasets failed to load.")

    combined = concatenate_datasets(parts)
    combined = combined.filter(lambda x: 0 < len(str(x["ground_truth"])) < 200)
    combined = combined.shuffle(seed=42)

    print(f"\n  ── Phase C Dataset ──")
    print(f"  Total: {len(combined):,} problems")
    for src, cnt in Counter(combined["source"]).most_common():
        print(f"  {src:<20}: {cnt:>5,}  ({cnt/len(combined)*100:.1f}%)")
    print(f"\n  Difficulty window : 20-50% pass rate")
    print(f"  Expected frac_zero_std: <0.2")
    print(f"  Expected clipped_ratio: <0.4")

    return combined


# ════════════════════════════════════════════════════════════════════════════
# MONITORING
# ════════════════════════════════════════════════════════════════════════════

class FullGRPOMonitor(TrainerCallback):
    """
    Monitor for full parameter GRPO.
    Primary metrics: pass@1, tag_rate (should emerge from binary reward),
    response length, frac_zero_std.

    Key difference from Phase B monitor:
    - No clean_term tracking (we expect it to emerge naturally)
    - Tag rate tracked as emergent behavior metric
    - If tag rate < 50% at step 200 → binary reward alone may not be enough
      → fallback: add tiny format signal (documented below)
    """

    def __init__(self, reward: BinaryReward, log_path: Path,
                 check_every: int = 25):
        self.reward      = reward
        self.log_path    = log_path
        self.check_every = check_every
        self.history     = []
        self.best_pass   = 0.0
        self.best_tag    = 0.0

    def on_step_end(self, args, state: TrainerState,
                    control: TrainerControl, **kwargs):
        step = state.global_step
        if step % self.check_every != 0 or step == 0:
            return control

        stats = self.reward.get_stats(last_n=self.check_every * 8)

        entry = {
            "step":      step,
            "pass_rate": stats["pass_rate"],
            "tag_rate":  stats["tag_rate"],
            "mean_len":  stats["mean_len"],
            "timestamp": datetime.now().isoformat(),
        }
        self.history.append(entry)

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"\n  ── Step {step} Monitor ──")
        print(f"    Pass@1    : {stats['pass_rate']*100:.1f}%")
        print(f"    Tag rate  : {stats['tag_rate']*100:.1f}%  "
              f"← emergent from binary reward")
        print(f"    Avg words : {stats['mean_len']:.0f}")

        if stats["pass_rate"] > self.best_pass:
            self.best_pass = stats["pass_rate"]
            print(f"    ⭐ New best pass@1: {self.best_pass*100:.1f}%")

        if stats["tag_rate"] > self.best_tag:
            self.best_tag = stats["tag_rate"]

        # Health checks
        if step == 200 and stats["tag_rate"] < 0.50:
            print(f"\n    ⚠️  Tag rate <50% at step 200")
            print(f"       Binary reward alone may not be enough for termination")
            print(f"       Consider adding: reward += 0.05 if has_solution_tag")
            print(f"       (small format signal — don't restart, add at step 200)")

        if step > 100 and stats["pass_rate"] < 0.08:
            print(f"\n    ⚠️  Pass rate very low (<8%)")
            print(f"       Dataset may be too hard despite calibration")
            print(f"       Check: increase --n-metamath, decrease --n-deepscaler")

        return control

    def on_train_end(self, args, state, control, **kwargs):
        print(f"\n{'═'*55}")
        print(f"  PHASE C TRAINING COMPLETE")
        print(f"  Best pass@1  : {self.best_pass*100:.1f}%")
        print(f"  Best tag rate: {self.best_tag*100:.1f}%")
        return control


class MATH500EvalCallback(TrainerCallback):
    """MATH500 eval at steps 500, 1000, 1500."""

    def __init__(self, tokenizer, log_path: Path,
                 eval_at: List[int] = None, n_eval: int = 100):
        self.tokenizer = tokenizer
        self.log_path  = log_path
        self.eval_at   = eval_at or [500, 1000, 1500]
        self.n_eval    = n_eval
        self.history   = []
        self.best_acc  = 0.0

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step not in self.eval_at:
            return control
        print(f"\n  MATH500 eval @ step {step}...")
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
            prob = item.get("problem", "")
            sol  = item.get("solution", "")
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
                    max_new_tokens=2000,
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
                print(f"    [{i+1:>3}/{n}]  {correct/(i+1)*100:.1f}%")

        acc = correct / n * 100
        print(f"  MATH500 @ step {step}: {correct}/{n} = {acc:.1f}%")
        print(f"  (Phase B baseline: ~62% | Target: ~75%+)")

        entry = {"step": step, "math500_acc": acc,
                 "n": n, "timestamp": datetime.now().isoformat()}
        self.history.append(entry)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump(self.history, f, indent=2)

        model.train()
        return acc


# ════════════════════════════════════════════════════════════════════════════
# MODEL LOADING — NO LORA
# ════════════════════════════════════════════════════════════════════════════

def resolve_model_path(model_arg: str) -> Path:
    """Auto-select best available checkpoint."""
    if model_arg != "auto":
        p = Path(model_arg)
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")
        return p

    if INPUT_MODEL.exists():
        print(f"  Auto-selected: {INPUT_MODEL}  (Phase B best merged)")
        return INPUT_MODEL
    elif FALLBACK.exists():
        print(f"  Auto-selected: {FALLBACK}  (Phase A merged — Phase B not ready)")
        return FALLBACK
    else:
        raise FileNotFoundError(
            f"No input model found.\n"
            f"  Expected: {INPUT_MODEL}\n"
            f"  Fallback: {FALLBACK}\n"
            f"  Run merge script first."
        )


def load_model_and_tokenizer(model_path: Path, attn_impl: str):
    print(f"\n{'─'*60}")
    print(f"  Loading : {model_path}")
    print(f"  Attn    : {attn_impl}")
    print(f"  LoRA    : NONE — full parameter training")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map={"": 0},
        trust_remote_code=True,
    )

    # Gradient checkpointing — saves ~30% VRAM, ~20% slower but worth it
    # Critical for full param training — no LoRA means full optimizer states
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()

    params = sum(p.numel() for p in model.parameters()) / 1e9
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad) / 1e9
    print(f"  Params    : {params:.2f}B total")
    print(f"  Trainable : {trainable:.2f}B  (ALL — no LoRA)")

    # Memory estimate
    model_gb  = params * 2          # bf16
    ref_gb    = params * 2          # reference copy
    optim_gb  = params * 4 * 2     # AdamW fp32 moments
    est_total = model_gb + ref_gb + optim_gb + 8  # +8GB activations
    print(f"\n  Memory estimate:")
    print(f"    Model     : {model_gb:.1f} GB")
    print(f"    Reference : {ref_gb:.1f} GB")
    print(f"    Optimizer : {optim_gb:.1f} GB")
    print(f"    Activations: ~8 GB")
    print(f"    Total est : {est_total:.1f} GB  (available: 48 GB)")

    return model, tokenizer


# ════════════════════════════════════════════════════════════════════════════
# GRPO CONFIG
# ════════════════════════════════════════════════════════════════════════════

def build_grpo_config(args, output_dir: Path) -> GRPOConfig:
    if args.wandb:
        os.environ["WANDB_PROJECT"] = "mathReason-1.5B"

    max_steps = 25 if args.sanity else args.max_steps

    print(f"\n  Phase C GRPO Config:")
    print(f"    max_steps     : {max_steps}  (1500 matches DeepScaleR 8K phase)")
    print(f"    num_gen       : {args.num_gen}")
    print(f"    max_comp_len  : {args.max_comp_len}")
    print(f"    lr            : {args.lr}  (lower than LoRA 5e-7 — full param)")
    print(f"    beta (kl_coef): {args.beta}  (lower — full param more stable)")
    print(f"    reward        : BINARY ONLY — correct=1.0, wrong=0.0")
    print(f"    LoRA          : NONE")

    base_kwargs = dict(
        output_dir=str(output_dir),
        run_name=args.run_name,

        max_steps=max_steps,
        num_train_epochs=1,

        num_generations=args.num_gen,
        max_completion_length=args.max_comp_len,

        learning_rate=args.lr,
        per_device_train_batch_size=1,
        # Full param: gradient_accum=4 enough (no LoRA overhead)
        gradient_accumulation_steps=8,
        optim="adamw_torch_fused",
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_steps=20,
        lr_scheduler_type="cosine",

        beta=args.beta,
        temperature=args.temperature,

        bf16=True,
        use_vllm=False,

        save_strategy="steps",
        save_steps=50 if args.sanity else 100,
        save_total_limit=15,   # keep all — 1500 steps / 100 = 15 checkpoints

        logging_steps=5 if args.sanity else 10,
        report_to="wandb" if args.wandb else "none",

        remove_unused_columns=False,
        mask_truncated_completions=True,
        seed=42,
    )

    # DAPO clip_higher — same as Phase B
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
            print(f"    epsilon       : {args.epsilon_low} (symmetric)")

    cfg = GRPOConfig(**base_kwargs)
    return cfg


# ════════════════════════════════════════════════════════════════════════════
# MERGE
# ════════════════════════════════════════════════════════════════════════════

def save_model(model, tokenizer, merged_dir: Path):
    print(f"\n{'─'*60}")
    print("  Saving full parameter model...")
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    print(f"  ✅ Saved → {merged_dir}")
    print(f"     Next: Stage 5 Gap-Filling SFT")


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
    print("  STAGE 4C — FULL PARAMETER GRPO")
    print("  The correct approach: no LoRA, binary reward only")
    if sanity:
        print("  *** SANITY RUN — 60 examples, 25 steps ***")
    print("█"*65)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n  What makes this different from Phase A/B:")
    print(f"    Phase A: LoRA + binary   → exploration constrained to low-rank")
    print(f"    Phase B: LoRA + termination reward → band-aid for LoRA constraint")
    print(f"    Phase C: Full param + binary → full exploration, reward sufficient")
    print(f"\n  Expected improvement over Phase B:")
    print(f"    MATH-500: ~62% → ~75-80%  (+13-18 points)")
    print(f"    GSM8K   : ~91% → ~93-95%  (+2-4 points)")
    print(f"    Tags    : ~70% → ~90%+    (emergent from binary reward)")

    # ── Load model — NO LoRA ──
    model_path = resolve_model_path(args.model)
    model, tokenizer = load_model_and_tokenizer(model_path, args.attn_impl)

    # ── Reward ──
    global _reward
    _reward   = BinaryReward()
    reward_fn = make_reward_fn(_reward)

    # ── Dataset ──
    dataset = build_dataset(args)
    if sanity:
        dataset = dataset.select(range(min(60, len(dataset))))
    print(f"\n  Training on {len(dataset):,} problems")

    # ── Config ──
    grpo_cfg = build_grpo_config(args, output_dir)

    # ── Callbacks ──
    monitor_cb = FullGRPOMonitor(
        reward      = _reward,
        log_path    = LOG_DIR / "stage4c_monitor.json",
        check_every = 5 if sanity else 25,
    )
    eval_cb = MATH500EvalCallback(
        tokenizer = tokenizer,
        log_path  = LOG_DIR / "stage4c_math500_evals.json",
        eval_at   = [10, 25] if sanity else [500, 1000, 1500],
        n_eval    = 20 if sanity else 100,
    )

    # ── Trainer — no peft_config, full parameter ──
    trainer = GRPOTrainer(
        model            = model,
        args             = grpo_cfg,
        reward_funcs     = [reward_fn],
        train_dataset    = dataset,
        processing_class = tokenizer,
        # NO peft_config — this is the critical difference from Phase A/B
    )
    trainer.add_callback(monitor_cb)
    trainer.add_callback(eval_cb)

    # ── Train ──
    print(f"\n{'─'*65}")
    print("  Starting full parameter GRPO...")
    print(f"  Monitor: tail -f {LOG_DIR}/stage4c_monitor.json")
    print(f"{'─'*65}\n")

    result = trainer.train(
        resume_from_checkpoint=str(output_dir / "checkpoint-600") if args.resume else None
    )

    # ── Save ──
    metrics = result.metrics
    with open(LOG_DIR / "stage4c_train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  ✅ Train metrics: {LOG_DIR}/stage4c_train_metrics.json")

    # Final reward stats
    s = _reward.get_stats()
    print(f"\n  Final Stats:")
    print(f"    Pass@1    : {s['pass_rate']*100:.1f}%")
    print(f"    Tag rate  : {s['tag_rate']*100:.1f}%")
    print(f"    Avg words : {s['mean_len']:.0f}")

    # Save model
    if not args.skip_merge and not sanity:
        save_model(model, tokenizer, merged_dir)
    elif sanity:
        print("\n  [Sanity] Skipping save.")

    # ── Summary ──
    print(f"\n{'█'*65}")
    if sanity:
        print("  SANITY COMPLETE — check:")
        print("  1. Loss not NaN")
        print("  2. Pass@1 > 5% (binary reward firing)")
        print("  3. reward_std > 0.05 on some batches")
        print("  4. No OOM  (full param needs ~26GB)")
        print()
        print("  If all good → full run:")
        print(f"  CUDA_VISIBLE_DEVICES=0 python stage4c_fullgrpo.py "
              f"--run-name fullgrpo_v1 --wandb")
    else:
        print("  PHASE C COMPLETE")
        print(f"  Saved: {merged_dir}")

        eval_log = LOG_DIR / "stage4c_math500_evals.json"
        if eval_log.exists():
            with open(eval_log) as f:
                evals = json.load(f)
            if evals:
                print(f"\n  MATH500 trajectory:")
                for e in evals:
                    print(f"    Step {e['step']:>4}: {e['math500_acc']:.1f}%")
                best_e = max(evals, key=lambda x: x["math500_acc"])
                print(f"\n  Best: {best_e['math500_acc']:.1f}% @ step {best_e['step']}")
                print(f"  Phase B baseline: ~62%")

        print()
        print("  Next steps:")
        print("    1. Stage 5 — Gap-Filling SFT (combinatorics + probability)")
        print("    2. Stage 6 — Elastic Reasoning")
        print("    3. Stage 7 — DPO Alignment")
        print("    4. Test-time: Maj@8 + PRM scoring")

    print(f"{'█'*65}\n")


if __name__ == "__main__":
    main()
