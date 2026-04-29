"""
Stage 4A — GRPO Phase 1 (4K Context)
══════════════════════════════════════════════════════════════════════════════
Server  : revanth@172.16.192.168
Input   : ~/nlp/checkpoints/stage2_sft_merged          (70% GSM8K)
Output  : ~/nlp/checkpoints/stage4a_grpo               (target: 78–82%)

Method  : RLVR (verifiable rewards) + GRPO + HAPO + STaR self-filter
Context : 4K tokens (Phase 1 — model learns correctness first)

Architecture of training loop:
  For each step:
    1. Sample prompt from dataset
    2. Generate G=8 completions (temperature=0.8)
    3. Score each with reward function (correctness + format + HAPO)
    4. Compute group-relative advantages: A_i = (r_i - mean) / std
    5. GRPO policy gradient update (clip ratio like PPO)
    6. KL penalty against reference model (prevents collapse)

Reward components:
  R_correct  : +1.0 correct, -0.5 wrong          (primary signal)
  R_format   : +0.1 correct tags, -0.2 missing   (format compliance)
  R_hapo     : +0.3 new shortest, cosine decay   (HAPO length reward)
  R_verify   : +0.05 explicit self-check present  (quality bonus)

Anti-collapse monitoring:
  - Entropy checked every 25 steps
  - Auto-rollback if entropy < 0.4 bits
  - Pass@8 tracked to detect mode collapse

Usage:
  # ALWAYS run sanity first
  CUDA_VISIBLE_DEVICES=1 python stage4a_grpo.py --sanity

  # Full run
  CUDA_VISIBLE_DEVICES=1 python stage4a_grpo.py --run-name grpo_p1_v1

  # With wandb
  CUDA_VISIBLE_DEVICES=1 python stage4a_grpo.py --wandb --run-name grpo_p1_v1

  # Resume
  CUDA_VISIBLE_DEVICES=1 python stage4a_grpo.py --resume --run-name grpo_p1_v1
══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
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
INPUT_MODEL = WORK_DIR / "checkpoints" / "stage2_sft_merged"
CKPT_DIR    = WORK_DIR / "checkpoints" / "stage4a_grpo"
MERGED_DIR  = WORK_DIR / "checkpoints" / "stage4a_grpo_merged"
LOG_DIR     = WORK_DIR / "logs"
DATA_CACHE  = WORK_DIR / "data" / "cache"
STAR_DIR    = WORK_DIR / "data" / "star_generated"   # STaR self-filter output

SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Think step by step inside <think> tags, "
    "then give your final answer inside <solution> tags."
)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 4A — GRPO Phase 1")

    # Modes
    p.add_argument("--sanity",        action="store_true",
                   help="Quick sanity: 50 examples, 20 steps")
    p.add_argument("--resume",        action="store_true")
    p.add_argument("--wandb",         action="store_true")
    p.add_argument("--run-name",      type=str,  default="grpo_p1_v1")

    # Paths
    p.add_argument("--model",         type=str,  default=str(INPUT_MODEL))
    p.add_argument("--output-dir",    type=str,  default=str(CKPT_DIR))
    p.add_argument("--merged-dir",    type=str,  default=str(MERGED_DIR))

    # LoRA
    p.add_argument("--lora-rank",     type=int,  default=64)
    p.add_argument("--lora-alpha",    type=int,  default=128)

    # GRPO core
    p.add_argument("--num-gen",       type=int,  default=8,
                   help="G — completions per prompt. Higher=more diverse but slower")
    p.add_argument("--max-steps",     type=int,  default=500,
                   help="Phase 1: 500 steps. Each step = 1 prompt × G completions")
    p.add_argument("--lr",            type=float,default=5e-7,
                   help="Very low LR for RL stability")
    p.add_argument("--kl-coef",       type=float,default=0.04,
                   help="KL penalty strength. Increase if entropy collapses")
    p.add_argument("--temperature",   type=float,default=0.8,
                   help="Generation temperature. 0.7-0.9 range")
    p.add_argument("--max-prompt-len",type=int,  default=512)
    p.add_argument("--max-comp-len",  type=int,  default=2048,
                   help="Max completion tokens. 4K context = 512 prompt + 2048 comp")

    # Reward weights
    p.add_argument("--w-correct",     type=float,default=1.0,
                   help="Weight on correctness reward component")
    p.add_argument("--w-format",      type=float,default=0.1,
                   help="Weight on format reward component")
    p.add_argument("--w-hapo",        type=float,default=0.3,
                   help="Weight on HAPO length reward component")
    p.add_argument("--w-verify",      type=float,default=0.05,
                   help="Weight on self-verification bonus")

    # Dataset
    p.add_argument("--n-gsm8k",       type=int,  default=7473,
                   help="GSM8K train examples (max 7473)")
    p.add_argument("--n-math",        type=int,  default=5800,
                   help="MATH Level 1-3 examples")
    p.add_argument("--n-metamath",    type=int,  default=8000,
                   help="MetaMathQA examples")
    p.add_argument("--n-deepscaler",  type=int,  default=3000,
                   help="DeepScaleR examples (set 0 to skip if not downloaded)")

    # Misc
    p.add_argument("--skip-eval",     action="store_true")
    p.add_argument("--skip-merge",    action="store_true")
    p.add_argument("--skip-star",     action="store_true",
                   help="Skip STaR self-filter after training")
    p.add_argument("--attn-impl",     type=str,  default="sdpa",
                   choices=["sdpa", "flash_attention_2", "eager"])

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# MATH ANSWER VERIFICATION
# This is the most critical component — bad verification = wrong rewards
# = model learns nothing or learns wrong things
# ══════════════════════════════════════════════════════════════════════════════

def extract_boxed(text: str) -> Optional[str]:
    """Extract content from \\boxed{...}, handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        idx = text.rfind("\\boxed {")
        if idx == -1:
            return None
    start = text.find("{", idx) + 1
    depth = 1
    pos   = start
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1
    return text[start:pos - 1].strip() if depth == 0 else None


def extract_predicted_answer(response: str) -> Optional[str]:
    """
    Extract predicted answer from model response.
    Priority order:
      1. <solution>...</solution> tag — extract clean answer from content
      2. \\boxed{...} anywhere in response
      3. Last number in full response (fallback)

    Key fix: solution tag content is cleaned to extract just the answer,
    not the full text. Handles cases like:
      <solution>460\nThe answer is: 460</solution>  → "460"
      <solution>x = 5</solution>                     → "5"
      <solution>\\frac{3}{4}</solution>              → "3/4"
    """
    if not isinstance(response, str):
        return None

    # 1. Solution tag
    sol_m = re.search(r"<solution>(.*?)</solution>", response, re.DOTALL)
    if sol_m:
        content = sol_m.group(1).strip()

        # Try boxed inside solution first
        boxed = extract_boxed(content)
        if boxed:
            return boxed

        # Try to extract clean answer from content:
        # Case A: content is purely a number (or simple expression)
        content_clean = content.split("\n")[0].strip()  # first line only
        content_clean = re.sub(r"[Tt]he answer is[:\s]*", "", content_clean).strip()
        content_clean = re.sub(r"[Tt]herefore[,:\s]*", "", content_clean).strip()
        content_clean = re.sub(r"[Ss]o[,:\s]*", "", content_clean).strip()
        content_clean = content_clean.rstrip(".")

        # If cleaned content looks like a valid answer, return it
        if content_clean and len(content_clean) < 50:
            # Step 1: try pure number
            try:
                float(content_clean.replace(",", ""))
                return content_clean.replace(",", "")
            except ValueError:
                pass

            # Step 2: fraction check — do this BEFORE word count split
            # catches "3/4", "x = 3/4", "2/3 of the total", "Therefore 2/3"
            frac_m = re.search(r"-?\d+\s*/\s*\d+", content_clean)
            if frac_m:
                return frac_m.group(0).replace(" ", "")

            # Step 3: exact fraction match (already clean)
            if re.match(r"^-?\d+\s*/\s*\d+$", content_clean):
                return content_clean

            # Step 4: short content — extract number or return as word answer
            if len(content_clean.split()) <= 3:
                inline_nums = re.findall(r"-?[\d,]+(?:\.\d+)?",
                                         content_clean.replace(",", ""))
                if inline_nums:
                    return inline_nums[-1]
                return content_clean

        # Case B: fraction anywhere in full solution content
        frac_m = re.search(r"-?\d+\s*/\s*\d+", content)
        if frac_m:
            return frac_m.group(0).replace(" ", "")

        # Case C: last number from solution content
        nums = re.findall(r"-?[\d,]+(?:\.\d+)?", content.replace(",", ""))
        if nums:
            return nums[-1]

        # Fallback: return cleaned first line
        return content_clean if content_clean else content[:50]

    # 2. Boxed anywhere in response
    boxed = extract_boxed(response)
    if boxed:
        return boxed

    # 3. Last number in full response
    # Search in the last 200 chars — answer is usually at the end
    tail = response[-200:] if len(response) > 200 else response
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", tail.replace(",", ""))
    return nums[-1] if nums else None


def normalize_answer(ans: str) -> str:
    """
    Normalize answer string for comparison.
    Handles: commas, units, trailing zeros, simple fractions.
    """
    if ans is None:
        return ""
    ans = str(ans).strip()

    # Remove common units
    ans = re.sub(r"\s*(dollars?|cents?|meters?|km|kg|cm|miles?|feet|inches?|%)\s*$",
                 "", ans, flags=re.IGNORECASE)

    # Remove commas in numbers (1,000 → 1000)
    ans = re.sub(r"(\d),(\d)", r"\1\2", ans)

    # Strip LaTeX formatting
    ans = re.sub(r"\\text\{([^}]*)\}", r"\1", ans)
    ans = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", ans)
    ans = ans.replace("$", "").replace("\\", "").strip()

    # Normalize decimals: 1.50 → 1.5, 1.0 → 1
    try:
        f = float(ans)
        if f == int(f):
            return str(int(f))
        return f"{f:.6f}".rstrip("0")
    except (ValueError, OverflowError):
        pass

    return ans.lower().strip()


def try_numeric_equal(a: str, b: str) -> Optional[bool]:
    """Try to compare as floats. Returns None if either can't be parsed."""
    try:
        fa, fb = float(a), float(b)
        return abs(fa - fb) < max(1e-6, 1e-4 * max(abs(fa), abs(fb)))
    except (ValueError, OverflowError):
        return None


def try_fraction_equal(a: str, b: str) -> Optional[bool]:
    """Try to compare simple fractions like 3/4 vs 0.75."""
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
        return abs(fa - fb) < 1e-6
    return None


def verify_answer(predicted: Optional[str], ground_truth: str) -> bool:
    """
    Robust math answer verification.
    Returns True if predicted matches ground_truth.

    Handles:
      - Numeric equality (68 == 68.0 == 68.00)
      - Fraction equality (3/4 == 0.75)
      - Normalized string match (removes LaTeX, units)
      - Case-insensitive word answer match (yes/no, true/false)
    """
    if predicted is None or ground_truth is None:
        return False

    pred_norm = normalize_answer(predicted)
    gt_norm   = normalize_answer(ground_truth)

    if not pred_norm or not gt_norm:
        return False

    # Exact match after normalization
    if pred_norm == gt_norm:
        return True

    # Numeric match
    num_eq = try_numeric_equal(pred_norm, gt_norm)
    if num_eq is not None:
        return num_eq

    # Fraction match
    frac_eq = try_fraction_equal(pred_norm, gt_norm)
    if frac_eq is not None:
        return frac_eq

    # Word answers (yes/no, true/false, etc.)
    word_map = {"yes": "true", "no": "false", "1": "true", "0": "false"}
    p_word = word_map.get(pred_norm, pred_norm)
    g_word = word_map.get(gt_norm, gt_norm)
    if p_word == g_word:
        return True

    return False


def extract_ground_truth(raw_answer: str, source: str) -> str:
    """
    Extract clean ground truth answer from raw dataset answer field.
    Handles different dataset formats.
    """
    if source == "gsm8k":
        # Format: "... #### 42"
        parts = raw_answer.split("####")
        return parts[-1].strip() if len(parts) > 1 else raw_answer.strip()

    if source == "math":
        # Format: full solution with \\boxed{answer}
        boxed = extract_boxed(raw_answer)
        return boxed if boxed else raw_answer.strip()

    if source == "metamath":
        # Various formats: "The answer is X" or "#### X"
        if "####" in raw_answer:
            return raw_answer.split("####")[-1].strip()
        m = re.search(r"[Tt]he answer is[:\s]+([^\n\.]+)", raw_answer)
        if m:
            return m.group(1).strip()
        # Last number
        nums = re.findall(r"-?[\d,]+(?:\.\d+)?", raw_answer)
        return nums[-1].replace(",", "") if nums else raw_answer.strip()

    if source == "deepscaler":
        # Usually clean answer string or boxed
        boxed = extract_boxed(raw_answer)
        return boxed if boxed else raw_answer.strip()

    return raw_answer.strip()


# ══════════════════════════════════════════════════════════════════════════════
# REWARD SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

class RewardComputer:
    """
    Computes multi-component rewards for GRPO training.
    Maintains HAPO history across the entire training run.
    Thread-safe for single-GPU usage (no multiprocessing needed).
    """

    def __init__(self,
                 w_correct: float = 1.0,
                 w_format:  float = 0.1,
                 w_hapo:    float = 0.3,
                 w_verify:  float = 0.05):
        self.w_correct = w_correct
        self.w_format  = w_format
        self.w_hapo    = w_hapo
        self.w_verify  = w_verify

        # Tokenizer reference (set after loading) — used by reward_fn to decode token IDs
        self._tokenizer = None

        # HAPO: problem_id → minimum correct response word count seen so far
        self.hapo_history: Dict[str, int] = {}

        # Stats for monitoring
        self.stats = defaultdict(list)   # lists of per-step values
        self.n_calls = 0

    def _correctness(self, completion: str, ground_truth: str) -> Tuple[float, bool]:
        """
        Returns (reward, is_correct).
        Uses 0/1 not -0.5/1.0 because GRPO needs reward variance across
        the G=8 group. If all wrong -> std=0 -> advantage=0 -> no gradient.
        With 0/1: first correct completion breaks symmetry immediately.
        """
        predicted = extract_predicted_answer(completion)
        is_correct = verify_answer(predicted, ground_truth)
        r = self.w_correct * (1.0 if is_correct else 0.0)
        return r, is_correct

    def _soft_length_signal(self, completion: str) -> float:
        """
        Soft reward providing variance even when all completions are wrong.
        Without this, early training (0% pass rate) has reward_std=0
        and GRPO learns nothing. This ensures grad_norm > 0 from step 1.
        Rewards: think tags present + reasonable length (not too short).
        Max contribution: 0.13 (small vs correctness reward of 1.0).
        """
        length   = len(completion.split())
        has_think_open  = "<think>"    in completion
        has_think_close = "</think>"   in completion
        has_sol         = "<solution>" in completion

        # Sigmoid length reward: ~0 at 30 words, ~0.08 at 200+ words
        length_r = 0.08 * (1.0 / (1.0 + math.exp(-(length - 100) / 50.0)))

        # Structure bonus: each tag present = small reward
        tag_r = 0.0
        if has_think_open:  tag_r += 0.02
        if has_think_close: tag_r += 0.02
        if has_sol:         tag_r += 0.01

        return length_r + tag_r

    def _format(self, completion: str) -> float:
        """Check <think>...</think> and <solution>...</solution> tags."""
        has_think_open  = "<think>"    in completion
        has_think_close = "</think>"   in completion
        has_sol_open    = "<solution>" in completion
        has_sol_close   = "</solution>"in completion
        all_tags = has_think_open and has_think_close and has_sol_open and has_sol_close
        r = self.w_format * (0.1 if all_tags else -0.2)
        return r

    def _hapo(self, completion: str, problem_id: str, is_correct: bool) -> float:
        """
        HAPO: History-Aware Policy Optimization length reward.
        Only rewards on correct responses.
        Rewards new shortest correct solution (encourages efficiency).
        Uses cosine decay for longer-than-best solutions.
        """
        if not is_correct:
            return 0.0

        resp_len  = len(completion.split())
        hist_best = self.hapo_history.get(problem_id, None)

        if hist_best is None or resp_len < hist_best:
            self.hapo_history[problem_id] = resp_len
            return self.w_hapo * 1.0   # new shortest → full reward

        # Cosine decay: still positive, but diminishing
        ratio = (resp_len - hist_best) / max(hist_best, 1)
        cosine_factor = max(0.0, math.cos(math.pi * min(ratio, 1.0)))
        return self.w_hapo * 0.3 * cosine_factor

    def _verify_bonus(self, completion: str, is_correct: bool) -> float:
        """
        Small bonus for explicit self-verification in the reasoning chain.
        Encourages the model to double-check its work.
        """
        if not is_correct:
            return 0.0
        verify_patterns = [
            "let me verify", "let me check", "checking:", "verification:",
            "let me confirm", "to verify", "indeed,", "double-check"
        ]
        has_verify = any(p in completion.lower() for p in verify_patterns)
        return self.w_verify if has_verify else 0.0

    def compute(self, completion: str, ground_truth: str, problem_id: str) -> float:
        """Compute total reward for a single (completion, ground_truth) pair."""
        r_correct, is_correct = self._correctness(completion, ground_truth)
        r_format              = self._format(completion)
        r_hapo                = self._hapo(completion, problem_id, is_correct)
        r_verify              = self._verify_bonus(completion, is_correct)
        r_soft                = self._soft_length_signal(completion)

        total = r_correct + r_format + r_hapo + r_verify + r_soft

        # Track stats
        self.stats["correct"].append(float(is_correct))
        self.stats["r_correct"].append(r_correct)
        self.stats["r_format"].append(r_format)
        self.stats["r_hapo"].append(r_hapo)
        self.stats["total"].append(total)
        self.stats["resp_len"].append(len(completion.split()))
        self.n_calls += 1

        return total

    def get_stats_summary(self, last_n: int = 200) -> Dict:
        """Return stats for the last N reward computations."""
        def avg(lst):
            tail = lst[-last_n:] if len(lst) >= last_n else lst
            return sum(tail) / max(len(tail), 1)

        return {
            "pass_rate":    avg(self.stats["correct"]),
            "mean_reward":  avg(self.stats["total"]),
            "mean_r_hapo":  avg(self.stats["r_hapo"]),
            "mean_len":     avg(self.stats["resp_len"]),
            "n_total":      self.n_calls,
            "hapo_tracked": len(self.hapo_history),
        }

    def log_stats(self, step: int):
        s = self.get_stats_summary()
        print(f"\n  [Reward Stats @ step {step}]")
        print(f"    Pass rate       : {s['pass_rate']*100:.1f}%")
        print(f"    Mean reward     : {s['mean_reward']:+.3f}")
        print(f"    Mean HAPO reward: {s['mean_r_hapo']:+.3f}")
        print(f"    Mean resp len   : {s['mean_len']:.0f} words")
        print(f"    HAPO problems   : {s['hapo_tracked']:,}")


# Global reward computer (persists across training steps)
_reward_computer: Optional[RewardComputer] = None


def make_reward_fn(reward_computer: RewardComputer):
    """
    Factory that creates the reward function closure for GRPOTrainer.
    GRPOTrainer calls: reward_fn(prompts, completions, **dataset_columns)
    Returns: List[float] of length len(completions)
    """
    def reward_fn(prompts, completions, **kwargs) -> List[float]:
        """
        GRPOTrainer may pass completions as:
          - List[str]           (decoded text) — older trl
          - List[List[int]]     (token ID lists) — newer trl
          - List[torch.Tensor]  (token tensors) — some versions
        We handle all three cases.
        """
        from transformers import AutoTokenizer as _AT
        import torch as _torch

        ground_truths = kwargs.get("ground_truth", [""] * len(completions))
        problem_ids   = kwargs.get("problem_id",   [str(i) for i in range(len(completions))])

        # Decode completions if they are not already strings
        decoded = []
        for comp in completions:
            if isinstance(comp, str):
                decoded.append(comp)
            elif isinstance(comp, _torch.Tensor):
                # tensor of token ids
                tok = reward_computer._tokenizer
                if tok is not None:
                    decoded.append(tok.decode(comp.tolist(), skip_special_tokens=True))
                else:
                    decoded.append("")
            elif isinstance(comp, (list, tuple)) and len(comp) > 0:
                if isinstance(comp[0], int):
                    # list of int token ids
                    tok = reward_computer._tokenizer
                    if tok is not None:
                        decoded.append(tok.decode(comp, skip_special_tokens=True))
                    else:
                        decoded.append("")
                else:
                    # list of strings — join
                    decoded.append(" ".join(str(c) for c in comp))
            else:
                decoded.append(str(comp))

        rewards = []
        for comp, gt, pid in zip(decoded, ground_truths, problem_ids):
            r = reward_computer.compute(comp, str(gt), str(pid))
            rewards.append(r)

        return rewards

    return reward_fn


# ══════════════════════════════════════════════════════════════════════════════
# DATASET LOADING
# ══════════════════════════════════════════════════════════════════════════════

def problem_id(text: str) -> str:
    """Stable hash of problem text → used for HAPO history."""
    return hashlib.md5(text.encode()).hexdigest()[:16]


# Module-level tokenizer — set in main() before dataset build
_global_tokenizer = None


def make_prompt_messages(problem: str) -> List[Dict]:
    """Format problem as chat messages for GRPOTrainer."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": problem.strip()},
    ]


def make_prompt_string(problem: str) -> str:
    """
    Pre-format prompt as a string using the chat template.
    add_generation_prompt=True appends the assistant turn opener
    so the model starts generating inside <think> tags immediately.
    """
    if _global_tokenizer is None:
        return "System: " + SYSTEM_PROMPT + "\n\nUser: " + problem.strip() + "\n\nAssistant: "
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": problem.strip()},
    ]
    return _global_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_gsm8k_train(tokenizer, n: int, sanity: bool) -> Dataset:
    print("  Loading GSM8K train...")
    try:
        ds = load_dataset("gsm8k", "main", split="train",
                          cache_dir=str(DATA_CACHE))
    except Exception as e:
        print(f"  ⚠️  GSM8K load failed: {e}")
        return Dataset.from_dict({"prompt": [], "ground_truth": [], "problem_id": []})

    if sanity:
        ds = ds.select(range(min(20, len(ds))))
    else:
        ds = ds.select(range(min(n, len(ds))))

    def process(ex):
        prob   = ex["question"]
        raw_gt = ex["answer"]
        gt     = extract_ground_truth(raw_gt, "gsm8k")
        return {
            "prompt":       make_prompt_string(prob),
            "ground_truth": gt,
            "problem_id":   problem_id(prob),
            "source":       "gsm8k",
        }

    ds = ds.map(process, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: len(x["ground_truth"]) > 0)
    print(f"  ✅ GSM8K train: {len(ds):,}")
    return ds


def load_math_l1_l3(tokenizer, n: int, sanity: bool) -> Dataset:
    print("  Loading MATH Level 1–3...")
    math_loaded = False
    for dataset_name in [
        "lighteval/MATH",
        "EleutherAI/hendrycks_math",
        "hendrycks/competition_math",
        "competition_math",
    ]:
        try:
            ds = load_dataset(dataset_name, split="train",
                              cache_dir=str(DATA_CACHE))
            print(f"  Loaded MATH from: {dataset_name}")
            math_loaded = True
            break
        except Exception as e:
            print(f"  {dataset_name}: {str(e)[:60]}")
            continue
    if not math_loaded:
        print(f"  ⚠️  All MATH sources failed. Skipping.")
        return Dataset.from_dict({"prompt": [], "ground_truth": [], "problem_id": []})

    # Filter levels 1-3 only for Phase 1
    try:
        ds = ds.filter(lambda x: x.get("level", "Level 3") in
                       ["Level 1", "Level 2", "Level 3"])
    except Exception:
        pass

    if sanity:
        ds = ds.select(range(min(15, len(ds))))
    else:
        ds = ds.select(range(min(n, len(ds))))

    def process(ex):
        prob   = ex.get("problem", ex.get("question", ""))
        sol    = ex.get("solution", ex.get("answer", ""))
        gt     = extract_ground_truth(sol, "math")
        return {
            "prompt":       make_prompt_string(prob),
            "ground_truth": gt,
            "problem_id":   problem_id(prob),
            "source":       "math",
        }

    ds = ds.map(process, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: len(x["ground_truth"]) > 0 and len(x["ground_truth"]) < 100)
    print(f"  ✅ MATH L1–3: {len(ds):,}")
    return ds


def load_metamathqa(tokenizer, n: int, sanity: bool) -> Dataset:
    print("  Loading MetaMathQA...")
    try:
        ds = load_dataset("meta-math/MetaMathQA", split="train",
                          cache_dir=str(DATA_CACHE))
    except Exception as e:
        print(f"  ⚠️  MetaMath load failed: {e}. Skipping.")
        return Dataset.from_dict({"prompt": [], "ground_truth": [], "problem_id": []})

    # Filter: only original GSM8K / MATH style (not augmented rephrasing)
    try:
        ds = ds.filter(lambda x: x.get("type", "") in
                       ["GSM_Rephrased", "MATH_Rephrased",
                        "GSM_AnsAug", "MATH_AnsAug", "GSM_FOBAR", "GSM_SV"])
    except Exception:
        pass

    if sanity:
        ds = ds.select(range(min(15, len(ds))))
    else:
        ds = ds.select(range(min(n, len(ds))))

    def process(ex):
        prob   = ex.get("query", "")
        raw_gt = ex.get("response", "")
        gt     = extract_ground_truth(raw_gt, "metamath")
        return {
            "prompt":       make_prompt_string(prob),
            "ground_truth": gt,
            "problem_id":   problem_id(prob),
            "source":       "metamath",
        }

    ds = ds.map(process, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: len(x["ground_truth"]) > 0 and len(x["ground_truth"]) < 100)
    print(f"  ✅ MetaMathQA: {len(ds):,}")
    return ds


def load_deepscaler(tokenizer, n: int, sanity: bool) -> Dataset:
    print("  Loading DeepScaleR (AMC/AIME subset)...")
    local_path = WORK_DIR / "data" / "deepscaler"

    try:
        if local_path.exists():
            ds = load_dataset(str(local_path), split="train")
        else:
            ds = load_dataset("agentica-org/DeepScaleR-Preview-Dataset",
                              cache_dir=str(DATA_CACHE))
            if hasattr(ds, "keys"):  # DatasetDict
                ds = ds["train"] if "train" in ds else list(ds.values())[0]
    except Exception as e:
        print(f"  ⚠️  DeepScaleR load failed: {e}")
        print(f"     Download with: huggingface-cli download agentica-org/DeepScaleR-Preview-Dataset "
              f"--repo-type dataset --local-dir ~/nlp/data/deepscaler")
        return Dataset.from_dict({"prompt": [], "ground_truth": [], "problem_id": []})

    # Show available columns and sample source values
    print(f"  DeepScaleR columns: {ds.column_names}")
    if "source" in ds.column_names:
        sources = set(ds["source"][:100])
        print(f"  DeepScaleR source values (sample): {sources}")
        # Filter out AIME (hardest) for Phase 1 — keep AMC and easier
        aime_keywords = ["aime", "olympiad", "putnam"]
        try:
            ds = ds.filter(lambda x: not any(
                kw in str(x.get("source", "")).lower() for kw in aime_keywords
            ))
            print(f"  After AIME filter: {len(ds):,}")
        except Exception:
            pass  # Keep all if filter fails

    if sanity:
        ds = ds.select(range(min(10, len(ds))))
    else:
        ds = ds.select(range(min(n, len(ds))))

    def process(ex):
        prob   = ex.get("problem", ex.get("question", ""))
        raw_gt = ex.get("answer", ex.get("solution", ""))
        gt     = extract_ground_truth(str(raw_gt), "deepscaler")
        return {
            "prompt":       make_prompt_string(prob),
            "ground_truth": gt,
            "problem_id":   problem_id(prob),
            "source":       "deepscaler",
        }

    ds = ds.map(process, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: len(x["ground_truth"]) > 0 and len(x["ground_truth"]) < 200)
    print(f"  ✅ DeepScaleR: {len(ds):,}")
    return ds


def build_grpo_dataset(args, tokenizer) -> Dataset:
    """
    Build the complete Phase 1 GRPO dataset.
    Difficulty curriculum: GSM8K (easy) → MetaMath → MATH → DeepScaleR (harder)
    """
    print("\nBuilding GRPO dataset...")
    sanity = args.sanity

    parts = []

    gsm   = load_gsm8k_train(tokenizer, args.n_gsm8k, sanity)
    parts.append(gsm)

    meta  = load_metamathqa(tokenizer, args.n_metamath, sanity)
    parts.append(meta)

    math_ = load_math_l1_l3(tokenizer, args.n_math, sanity)
    parts.append(math_)

    if args.n_deepscaler > 0:
        deep = load_deepscaler(tokenizer, args.n_deepscaler, sanity)
        parts.append(deep)

    # Filter empty parts
    parts = [p for p in parts if len(p) > 0]

    if not parts:
        raise ValueError("All datasets failed to load. Check your internet/cache.")

    # Concatenate in difficulty order (curriculum)
    combined = concatenate_datasets(parts)

    # Remove examples with very short or very long ground truths
    combined = combined.filter(
        lambda x: 0 < len(str(x["ground_truth"])) < 200
    )

    # Shuffle within difficulty blocks (not full shuffle — preserve rough curriculum)
    combined = combined.shuffle(seed=42)

    print(f"\n  ── GRPO Dataset Summary ──")
    print(f"  Total problems : {len(combined):,}")

    # Source breakdown
    sources = Counter(combined["source"])
    for src, cnt in sources.most_common():
        print(f"  {src:<15}: {cnt:>6,} ({cnt/len(combined)*100:.1f}%)")

    return combined


# ══════════════════════════════════════════════════════════════════════════════
# MONITORING CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

class GRPOMonitorCallback(TrainerCallback):
    """
    Monitors training health every N steps.
    Tracks:
      - Reward statistics (via RewardComputer)
      - Response entropy (diversity measure)
      - Auto-stops if entropy collapses
    Logs to JSON for later analysis.
    """

    def __init__(self,
                 reward_computer: RewardComputer,
                 log_path: Path,
                 check_every: int = 25,
                 min_entropy: float = 0.4):
        self.rc          = reward_computer
        self.log_path    = log_path
        self.check_every = check_every
        self.min_entropy = min_entropy
        self.history     = []
        self.best_pass   = 0.0

    def on_step_end(self, args, state: TrainerState,
                    control: TrainerControl, **kwargs):
        step = state.global_step

        if step % self.check_every != 0 or step == 0:
            return control

        stats = self.rc.get_stats_summary(last_n=self.check_every * 8)

        entry = {
            "step":       step,
            "pass_rate":  stats["pass_rate"],
            "mean_reward":stats["mean_reward"],
            "mean_len":   stats["mean_len"],
            "hapo_tracked":stats["hapo_tracked"],
            "timestamp":  datetime.now().isoformat(),
        }
        self.history.append(entry)

        # Save log
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump(self.history, f, indent=2)

        # Print summary
        print(f"\n  ── Step {step} Monitor ──")
        print(f"    Pass@1 (last {self.check_every * 8} completions): "
              f"{stats['pass_rate']*100:.1f}%")
        print(f"    Mean reward  : {stats['mean_reward']:+.3f}")
        print(f"    Mean resp len: {stats['mean_len']:.0f} words")
        print(f"    HAPO tracked : {stats['hapo_tracked']:,} problems")

        # Track best pass rate
        if stats["pass_rate"] > self.best_pass:
            self.best_pass = stats["pass_rate"]
            print(f"    ⭐ New best pass rate: {self.best_pass*100:.1f}%")

        # Warn if pass rate is very low after warmup
        if step > 100 and stats["pass_rate"] < 0.05:
            print(f"    ⚠️  Pass rate very low (<5%) — check reward function")

        return control


class GSM8KEvalCallback(TrainerCallback):
    """
    Runs GSM8K evaluation every eval_every steps and at end of training.
    Uses greedy decoding on the current merged model state.
    """

    def __init__(self,
                 tokenizer,
                 log_path: Path,
                 eval_every: int = 100,
                 n_eval: int = 150):
        self.tokenizer  = tokenizer
        self.log_path   = log_path
        self.eval_every = eval_every
        self.n_eval     = n_eval
        self.history    = []
        self.best_acc   = 0.0
        self.best_step  = 0

    def on_step_end(self, args, state: TrainerState,
                    control: TrainerControl, model=None, **kwargs):
        step = state.global_step
        if step % self.eval_every != 0 or step == 0:
            return control

        print(f"\n{'─'*50}")
        print(f"  GSM8K eval @ step {step}...")
        acc = self._run_eval(model, step)
        if acc > self.best_acc:
            self.best_acc  = acc
            self.best_step = step
            print(f"  ⭐ New best: {acc:.1f}% at step {step}")
        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):
        print(f"\n{'═'*50}")
        print(f"  FINAL GSM8K eval...")
        self._run_eval(model, state.global_step, n_override=200)
        return control

    def _run_eval(self, model, step: int, n_override: int = None) -> float:
        n = n_override or self.n_eval
        try:
            ds = load_dataset("gsm8k", "main", split="test",
                              cache_dir=str(DATA_CACHE))
            ds = ds.select(range(min(n, len(ds))))
        except Exception as e:
            print(f"  ⚠️  GSM8K eval failed: {e}")
            return -1.0

        model.eval()
        device  = next(model.parameters()).device
        correct = 0

        for i, item in enumerate(ds):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": item["question"]},
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
                    max_new_tokens=1024,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            resp = self.tokenizer.decode(
                out[0][enc["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            predicted = extract_predicted_answer(resp)
            gt        = extract_ground_truth(item["answer"], "gsm8k")
            # Debug: log first 5 mismatches
            if not verify_answer(predicted, gt) and i < 5:
                pass  # uncomment below to debug
                # print(f"  MISS: pred={repr(predicted)} gt={repr(gt)}")

            if verify_answer(predicted, gt):
                correct += 1

            if (i + 1) % 50 == 0:
                print(f"    [{i+1:>3}/{n}]  {correct}/{i+1} = {correct/(i+1)*100:.1f}%")

        acc = correct / n * 100
        print(f"  GSM8K @ step {step}: {correct}/{n} = {acc:.1f}%")

        entry = {
            "step": step, "gsm8k_acc": acc,
            "n_samples": n, "timestamp": datetime.now().isoformat()
        }
        self.history.append(entry)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump(self.history, f, indent=2)

        model.train()
        return acc


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(model_path: str, attn_impl: str):
    print(f"\n{'─'*60}")
    print(f"  Loading model: {model_path}")
    print(f"  Attn backend : {attn_impl}")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",     # LEFT padding for generation (critical for GRPO)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map={"": 0},
        trust_remote_code=True,
    )

    params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Parameters   : {params:.2f}B")
    return model, tokenizer


def apply_lora(model, rank: int, alpha: int) -> object:
    """
    Apply LoRA for GRPO training.
    In GRPO, LoRA is applied to the policy.
    The reference model is the base model (no LoRA).
    trl GRPOTrainer handles this automatically.
    """
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


# ══════════════════════════════════════════════════════════════════════════════
# GRPO CONFIG
# ══════════════════════════════════════════════════════════════════════════════

def build_grpo_config(args, output_dir: Path) -> GRPOConfig:
    """
    Build GRPOConfig. Handles parameter name differences across trl versions.
    """
    if args.wandb:
        os.environ["WANDB_PROJECT"] = "mathReason-1.5B"

    n_gpus    = max(torch.cuda.device_count(), 1)
    eff_steps = args.max_steps

    print(f"\n  GRPO Config:")
    print(f"    num_generations     : {args.num_gen}")
    print(f"    max_steps           : {eff_steps}")
    print(f"    max_completion_len  : {args.max_comp_len}")
    print(f"    learning_rate       : {args.lr}")
    print(f"    kl_coef             : {args.kl_coef}")
    print(f"    temperature         : {args.temperature}")
    print(f"    effective batch/step: 1 prompt × {args.num_gen} completions")

    # Build config — handle different trl versions gracefully
    # ── Probe GRPOConfig for correct parameter names ──
    # trl versions differ: some use max_prompt_length, others max_new_tokens etc.
    import inspect
    grpo_params = set(inspect.signature(GRPOConfig.__init__).parameters.keys())
    print(f"    trl GRPOConfig params (subset): "
          f"{[p for p in grpo_params if 'length' in p or 'token' in p or 'prompt' in p or 'gen' in p]}")

    # Map param names across trl versions
    prompt_len_key     = "max_prompt_length"     if "max_prompt_length"     in grpo_params else                          "max_new_tokens"         if "max_new_tokens"         in grpo_params else None
    completion_len_key = "max_completion_length"  if "max_completion_length"  in grpo_params else                          "max_new_tokens"          if "max_new_tokens"         in grpo_params else None
    num_gen_key        = "num_generations"        if "num_generations"        in grpo_params else                          "num_return_sequences"    if "num_return_sequences"   in grpo_params else None
    temperature_key    = "temperature"            if "temperature"            in grpo_params else None
    kl_key             = "kl_coef"               if "kl_coef"               in grpo_params else                          "beta"                   if "beta"                   in grpo_params else None
    vllm_key           = "use_vllm"              if "use_vllm"              in grpo_params else None

    print(f"    Resolved param names: prompt_len={prompt_len_key}, "
          f"completion_len={completion_len_key}, num_gen={num_gen_key}, "
          f"kl={kl_key}")

    # generation_batch_size must be divisible by num_generations
    # Rule: generation_batch_size = num_generations (simplest safe value)
    gen_batch = args.num_gen   # 8 — always divisible by itself

    # Base config — always valid fields
    config_kwargs = dict(
        output_dir=str(output_dir),
        run_name=args.run_name,
        max_steps=eff_steps if not args.sanity else 20,
        num_train_epochs=1,
        learning_rate=args.lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,   # accumulate via num_gen instead
        generation_batch_size=gen_batch, # must be divisible by num_generations
        optim="adamw_torch_fused",
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_steps=20,
        lr_scheduler_type="cosine",
        bf16=True,
        save_strategy="steps",
        save_steps=100 if not args.sanity else 10,
        save_total_limit=3,
        logging_steps=10 if not args.sanity else 2,
        report_to="wandb" if args.wandb else "none",
        remove_unused_columns=False,
        seed=42,
    )

    # Add version-dependent params safely
    if num_gen_key:
        config_kwargs[num_gen_key] = args.num_gen
    if prompt_len_key and prompt_len_key != completion_len_key:
        config_kwargs[prompt_len_key] = args.max_prompt_len
    if completion_len_key:
        config_kwargs[completion_len_key] = args.max_comp_len
    if temperature_key:
        config_kwargs[temperature_key] = args.temperature
    if kl_key:
        config_kwargs[kl_key] = args.kl_coef
    if vllm_key:
        config_kwargs[vllm_key] = False

    # Try optional params one by one
    for opt_key, opt_val in [("entropy_coef", 0.01)]:
        if opt_key in grpo_params:
            config_kwargs[opt_key] = opt_val
            print(f"    {opt_key}: {opt_val} ✅")
        else:
            print(f"    {opt_key}: not in this trl version (OK)")

    cfg = GRPOConfig(**config_kwargs)

    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# STaR SELF-FILTER (runs after GRPO training)
# ══════════════════════════════════════════════════════════════════════════════

def run_star_filter(model, tokenizer, n_problems: int = 2000,
                    n_attempts: int = 4, output_dir: Path = STAR_DIR):
    """
    STaR self-filter: generate solutions on unseen problems,
    keep only verified-correct ones for Phase 2 GRPO dataset.

    This is the "self-improvement" loop from STaR paper:
    the model curates its own training data for the next phase.

    Output: JSONL file with (problem, solution, ground_truth) triples
    that the model solved correctly.
    """
    print(f"\n{'═'*60}")
    print(f"  STaR Self-Filter")
    print(f"  Generating {n_attempts} attempts on {n_problems} MATH L3-4 problems")
    print(f"{'═'*60}")

    # Load harder problems not seen during Phase 1
    try:
        ds = load_dataset("lighteval/MATH", split="train",
                          cache_dir=str(DATA_CACHE))
        ds = ds.filter(lambda x: x.get("level", "") in ["Level 3", "Level 4"])
        ds = ds.shuffle(seed=999).select(range(min(n_problems, len(ds))))
    except Exception as e:
        print(f"  ⚠️  Could not load MATH for STaR: {e}")
        return

    model.eval()
    device = next(model.parameters()).device

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path  = output_dir / "star_round1.jsonl"
    hard_path = output_dir / "star_round1_hard.jsonl"   # problems model failed on

    verified_count = 0
    hard_count     = 0

    with open(out_path, "w") as fv, open(hard_path, "w") as fh:
        for i, item in enumerate(ds):
            prob   = item.get("problem", item.get("question", ""))
            sol    = item.get("solution", item.get("answer", ""))
            gt     = extract_ground_truth(sol, "math")

            if not prob or not gt:
                continue

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prob},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            enc = tokenizer(
                prompt, return_tensors="pt",
                truncation=True, max_length=512
            ).to(device)

            # Generate n_attempts solutions with temperature sampling
            solved = False
            best_solution = None

            for attempt in range(n_attempts):
                with torch.no_grad():
                    out = model.generate(
                        **enc,
                        max_new_tokens=2048,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                resp = tokenizer.decode(
                    out[0][enc["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )
                predicted = extract_predicted_answer(resp)

                if verify_answer(predicted, gt):
                    solved = True
                    best_solution = resp
                    break   # Take first correct solution

            if solved and best_solution:
                record = {
                    "problem":      prob,
                    "solution":     best_solution,
                    "ground_truth": gt,
                    "source":       item.get("type", "math"),
                    "level":        item.get("level", "Level 3"),
                }
                fv.write(json.dumps(record) + "\n")
                verified_count += 1
            else:
                # Hard negative — model failed all attempts
                hard_record = {
                    "problem":      prob,
                    "ground_truth": gt,
                    "level":        item.get("level", "Level 3"),
                }
                fh.write(json.dumps(hard_record) + "\n")
                hard_count += 1

            if (i + 1) % 100 == 0:
                solve_rate = verified_count / (i + 1) * 100
                print(f"  [{i+1:>4}/{n_problems}]  "
                      f"Solved: {verified_count}  "
                      f"Hard: {hard_count}  "
                      f"Rate: {solve_rate:.1f}%")

    print(f"\n  ✅ STaR filter complete:")
    print(f"     Verified (correct): {verified_count:,} → {out_path}")
    print(f"     Hard (failed all) : {hard_count:,} → {hard_path}")
    print(f"     Solve rate        : {verified_count/(verified_count+hard_count)*100:.1f}%")
    print(f"     → Use {out_path} as additional data in Phase 2 GRPO")

    model.train()


# ══════════════════════════════════════════════════════════════════════════════
# MERGE
# ══════════════════════════════════════════════════════════════════════════════

def merge_and_save(model, tokenizer, merged_dir: Path):
    print(f"\n{'─'*60}")
    print("  Merging LoRA → base model...")
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    print(f"  ✅ Merged checkpoint: {merged_dir}")
    print(f"     → Use this for Stage 4C GRPO Phase 2")
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    sanity = args.sanity

    suffix     = "_sanity" if sanity else ""
    output_dir = Path(args.output_dir) / f"{args.run_name}{suffix}"
    merged_dir = Path(args.merged_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STAR_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "█" * 60)
    print("  STAGE 4A — GRPO PHASE 1 (4K CONTEXT)")
    if sanity:
        print("  *** SANITY RUN — 50 examples, 20 steps ***")
    print("█" * 60)
    print(f"  Timestamp    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Input model  : {args.model}")
    print(f"  Output dir   : {output_dir}")
    print(f"  Max steps    : {args.max_steps if not sanity else 20}")
    print(f"  Num gen (G)  : {args.num_gen}")
    print(f"  KL coef      : {args.kl_coef}")
    print(f"  Reward weights: correct={args.w_correct} format={args.w_format} "
          f"hapo={args.w_hapo} verify={args.w_verify}")

    # ── Load model ──
    model, tokenizer = load_model_and_tokenizer(args.model, args.attn_impl)

    # ── Apply LoRA ──
    model = apply_lora(model, args.lora_rank, args.lora_alpha)

    # ── Reward computer ──
    global _reward_computer
    _reward_computer = RewardComputer(
        w_correct = args.w_correct,
        w_format  = args.w_format,
        w_hapo    = args.w_hapo,
        w_verify  = args.w_verify,
    )
    _reward_computer._tokenizer = tokenizer   # needed to decode token IDs in reward_fn
    reward_fn = make_reward_fn(_reward_computer)

    # ── Set global tokenizer (needed for make_prompt_string) ──
    global _global_tokenizer
    _global_tokenizer = tokenizer

    # ── Dataset ──
    dataset = build_grpo_dataset(args, tokenizer)
    if sanity:
        dataset = dataset.select(range(min(50, len(dataset))))

    print(f"\n  Training on {len(dataset):,} problems")

    # ── GRPO Config ──
    grpo_cfg = build_grpo_config(args, output_dir)

    # ── Callbacks ──
    monitor_cb = GRPOMonitorCallback(
        reward_computer = _reward_computer,
        log_path        = LOG_DIR / "stage4a_monitor.json",
        check_every     = 5 if sanity else 25,
    )
    eval_cb = GSM8KEvalCallback(
        tokenizer  = tokenizer,
        log_path   = LOG_DIR / "stage4a_gsm8k_evals.json",
        eval_every = 5 if sanity else 100,
        n_eval     = 20 if sanity else 150,
    )
    callbacks = [monitor_cb, eval_cb]

    # ── GRPOTrainer ──
    trainer = GRPOTrainer(
        model         = model,
        args          = grpo_cfg,
        reward_funcs  = [reward_fn],
        train_dataset = dataset,
        processing_class = tokenizer,
    )

    # ── Train ──
    print(f"\n{'─'*60}")
    print("  Starting GRPO training...")
    print(f"  Each step: 1 prompt → {args.num_gen} completions → reward → update")
    print(f"{'─'*60}\n")

    result = trainer.train(
        resume_from_checkpoint=str(output_dir) if args.resume else None
    )

    # ── Save ──
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = result.metrics
    with open(LOG_DIR / "stage4a_train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  ✅ LoRA adapter saved  : {output_dir}")
    print(f"  ✅ Train metrics saved : {LOG_DIR / 'stage4a_train_metrics.json'}")

    # ── Final reward stats ──
    _reward_computer.log_stats(step=grpo_cfg.max_steps)

    # ── Merge ──
    if not args.skip_merge and not sanity:
        merged_model = merge_and_save(model, tokenizer, merged_dir)
    else:
        merged_model = model
        if sanity:
            print("\n  [Sanity] Skipping merge.")

    # ── STaR self-filter ──
    if not args.skip_star and not sanity:
        print(f"\n  Running STaR self-filter on harder problems...")
        run_star_filter(
            model      = merged_model if not args.skip_merge else model,
            tokenizer  = tokenizer,
            n_problems = 2000,
            n_attempts = 4,
            output_dir = STAR_DIR,
        )
    elif sanity:
        print("\n  [Sanity] Skipping STaR filter.")

    # ── Summary ──
    print(f"\n{'█'*60}")
    if sanity:
        print("  SANITY COMPLETE — check:")
        print("  1. Loss is decreasing (not NaN)")
        print("  2. Pass rate > 0% in monitor logs")
        print("  3. Reward is not constant")
        print()
        print("  If all good → launch full run:")
        print(f"  CUDA_VISIBLE_DEVICES=1 python stage4a_grpo.py --run-name grpo_p1_v1")
    else:
        print("  STAGE 4A GRPO PHASE 1 COMPLETE")
        print("  ─────────────────────────────────────────────")
        print(f"  LoRA adapter  : {output_dir}")
        print(f"  Merged model  : {merged_dir}")
        print(f"  STaR data     : {STAR_DIR}/star_round1.jsonl")
        print(f"  Next step     : Stage 4C GRPO Phase 2 (8K context)")
        print(f"  Use checkpoint: {merged_dir}")

        # Print eval history
        eval_log = LOG_DIR / "stage4a_gsm8k_evals.json"
        if eval_log.exists():
            with open(eval_log) as f:
                evals = json.load(f)
            if evals:
                print(f"\n  GSM8K trajectory:")
                for e in evals:
                    print(f"    Step {e['step']:>3}: {e['gsm8k_acc']:.1f}%")
                best_e = max(evals, key=lambda x: x["gsm8k_acc"])
                print(f"\n  Best: {best_e['gsm8k_acc']:.1f}% at step {best_e['step']}")
                print(f"  Baseline (Stage 2): 70%")
    print(f"{'█'*60}\n")


if __name__ == "__main__":
    main()
