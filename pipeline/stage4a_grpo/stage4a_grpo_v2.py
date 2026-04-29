"""
Stage 4A — GRPO Phase A v2  (4K Context, Termination + Anti-Loop Training)
══════════════════════════════════════════════════════════════════════════════
Server  : revanth@172.16.192.168
Input   : ~/nlp/checkpoints/stage3_distilled_merged     (real GSM8K ~85%)
Output  : ~/nlp/checkpoints/stage4a_grpo_v2             (target: ~88% GSM8K)

Why v2 differs from v1
──────────────────────
v1 failures (now fixed):
  1. HAPO weight 0.3 was too large → model learned to give short wrong answers
     Fix: Length signal weight dropped to 0.1, only on CORRECT answers
  2. Answer extractor had a bug → "460\nThe answer is: 460" → False negative
     Fix: Smart extractor with 7-priority fallback chain
  3. GSM8K in dataset at 85% pass rate → dead gradient (7/8 completions correct)
     Fix: GSM8K completely removed. Competition problems only.
  4. Wrong penalty was 0.0 → model couldn't distinguish wrong from skipped
     Fix: Wrong penalty -0.3 (distinguishes wrong from unattempted)

v2 additions (new):
  5. Termination reward: rewards clean </solution>→STOP structure (+0.20/-0.20)
  6. Anti-loop penalty: penalises all 3 loop types found in Stage 3 diagnostic
     - Re-verification spiral ("let me check again" ×3+) → -0.20
     - "Wait, no" catastrophic loop → -0.25
     - Infinite repetition trap (same 50-word chunk repeated) → -0.15
  7. Dynamic batch skipping: if reward_std < 0.05, skip step (zero gradient batch)
  8. max_comp_len = 1500 (hard cap prevents 2000-token loop generation)

Reward architecture
───────────────────
  R_correct    : +1.0 correct | -0.3 wrong            (primary signal, largest)
  R_terminate  : +0.35 clean end | +0.09 tag+continues | -0.35 no tag
  R_anti_loop  : +0.05 clean | 0.0 one-check | -0.20 spiral | -0.25 wait-loop
  R_length     : +0.10 <300w | +0.05 <500w | -0.10 >900w  (correct only)
  R_format     : +0.05 both tags | -0.05 partial | -0.15 no tags

Dataset (NO GSM8K)
──────────────────
  DeepScaleR AMC subset       : 15,000   (model solves ~40-60%)
  MATH L3-4 (hendrycks)       :  8,000   (model solves ~35-55%)
  NuminaMath competition       :  3,000   (model solves ~20-40%)
  ─────────────────────────────────────
  Total                        : ~26,000

Usage
─────
  # ALWAYS sanity first (20 steps, 60 problems, ~15 min)
  CUDA_VISIBLE_DEVICES=1 python stage4a_grpo_v2.py --sanity

  # Full run (~18-22 hours)
  CUDA_VISIBLE_DEVICES=1 python stage4a_grpo_v2.py --run-name grpo_v2_r1

  # With wandb
  CUDA_VISIBLE_DEVICES=1 python stage4a_grpo_v2.py --wandb --run-name grpo_v2_r1

  # Resume
  CUDA_VISIBLE_DEVICES=1 python stage4a_grpo_v2.py --resume --run-name grpo_v2_r1

Sanity checks (what to look for)
─────────────────────────────────
  ✅ reward_std         > 0.05    (non-degenerate batches)
  ✅ grad_norm          > 0.01    (model is actually learning)
  ✅ pass_rate          > 5%      (competition problems, expect 20-50%)
  ✅ anti_loop_fired    > 0%      (reward catching loops)
  ✅ termination_rate   > 0%      (model sometimes terminates cleanly)
  ✅ skipped_batches    < 30%     (dynamic skip not firing too often)
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
INPUT_MODEL = WORK_DIR / "checkpoints" / "stage3_distilled_merged"   # v2: start from stage3
CKPT_DIR    = WORK_DIR / "checkpoints" / "stage4a_grpo_v2"
MERGED_DIR  = WORK_DIR / "checkpoints" / "stage4a_grpo_v2_merged"
LOG_DIR     = WORK_DIR / "logs"
DATA_CACHE  = WORK_DIR / "data" / "cache"
STAR_DIR    = WORK_DIR / "data" / "star_generated"

SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Think step by step inside <think> tags, "
    "then give your final answer inside <solution> tags."
)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 4A GRPO v2 — Termination + Anti-Loop")

    # Modes
    p.add_argument("--sanity",        action="store_true",
                   help="Quick sanity: 60 problems, 20 steps (~15 min)")
    p.add_argument("--resume",        action="store_true")
    p.add_argument("--wandb",         action="store_true")
    p.add_argument("--run-name",      type=str, default="grpo_v2_r1")

    # Paths
    p.add_argument("--model",         type=str, default=str(INPUT_MODEL),
                   help="Input checkpoint (default: stage3_distilled_merged)")
    p.add_argument("--output-dir",    type=str, default=str(CKPT_DIR))
    p.add_argument("--merged-dir",    type=str, default=str(MERGED_DIR))

    # LoRA
    p.add_argument("--lora-rank",     type=int, default=64)
    p.add_argument("--lora-alpha",    type=int, default=128)

    # GRPO core
    p.add_argument("--num-gen",       type=int, default=8,
                   help="Completions per prompt.")
    p.add_argument("--max-steps",     type=int, default=500)
    p.add_argument("--lr",            type=float, default=5e-7)
    p.add_argument("--kl-coef",       type=float, default=0.15,
                   help="KL penalty. 0.15 keeps model close to distilled checkpoint. "
                        "Lower to 0.04 only if pass rate plateaus above 60%%.")
    p.add_argument("--epsilon-low",   type=float, default=0.20,
                   help="Lower clip bound for policy ratio (standard PPO/GRPO)")
    p.add_argument("--epsilon-high",  type=float, default=0.28,
                   help="Upper clip bound — DAPO clip_higher. Asymmetric clipping "
                        "preserves high-entropy reasoning tokens (wait/however/reconsider) "
                        "preventing entropy collapse. Research-backed value: 0.28")
    p.add_argument("--temperature",   type=float, default=0.8)
    p.add_argument("--max-prompt-len",type=int, default=512)

    # Phase controls iterative context lengthening (DeepScaleR strategy)
    p.add_argument("--phase",         type=str, default="A",
                   choices=["A", "B", "C"],
                   help="Training phase: A=2K ctx, B=4K ctx, C=8K ctx. "
                        "Run A first, then B from A checkpoint, then C from B.")
    p.add_argument("--max-comp-len",  type=int, default=0,
                   help="Override completion length. 0=auto from --phase "
                        "(A=2048, B=4096, C=8192)")

    # Dataset sizes — calibrated for 30-70% pass rate window
    # Research (GHPO, GRPO-LEAD) confirms: problems must be in model's
    # "learnable zone" — not too easy (dead signal) not too hard (reward sparsity)
    # Our model after distillation: GSM8K ~85%, MATH L1-2 ~65%, MetaMath ~55%,
    #                               MATH L3 ~40%, MATH L4 ~20%, DeepScaleR ~10%
    p.add_argument("--n-metamath",    type=int, default=8000,
                   help="MetaMathQA — model solves ~55%%. Primary training signal.")
    p.add_argument("--n-gsm8k-hard", type=int, default=4000,
                   help="GSM8K hardest 20%% — model solves ~45%%. Good mixed batches.")
    p.add_argument("--n-math-l2l3",  type=int, default=6000,
                   help="MATH Level 2-3 — model solves ~40-65%%. Core difficulty target.")
    p.add_argument("--n-math-easy",  type=int, default=0,
                   help="MATH Level 1-2 — now replaced by MetaMath as easy anchor. "
                        "Set >0 only if MetaMath fails to load.")
    p.add_argument("--n-math",       type=int, default=2000,
                   help="MATH Level 4 — model solves ~20%%. Ceiling pusher, keep small.")
    p.add_argument("--n-deepscaler", type=int, default=0,
                   help="DeepScaleR — model solves ~8%% = reward sparsity. "
                        "Disabled by default. Enable in Phase B after termination learned.")
    p.add_argument("--n-numina",     type=int, default=2000,
                   help="NuminaMath competition — model solves ~20%%. Keep small.")

    # Eval
    p.add_argument("--eval-every",    type=int, default=250,
                   help="Run MATH500 eval every N steps")
    p.add_argument("--n-eval",        type=int, default=150,
                   help="Problems per eval run")

    # Misc
    p.add_argument("--skip-eval",     action="store_true")
    p.add_argument("--skip-merge",    action="store_true")
    p.add_argument("--skip-star",     action="store_true")
    p.add_argument("--attn-impl",     type=str, default="sdpa",
                   choices=["sdpa", "flash_attention_2", "eager"])

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# ANSWER VERIFICATION
# The most critical component — a bad verifier produces wrong rewards,
# which trains the model on wrong signal, causing silent degradation.
# ══════════════════════════════════════════════════════════════════════════════

def extract_boxed(text: str) -> Optional[str]:
    """Extract content from \\boxed{...}, handling nested braces correctly."""
    # Try rightmost \boxed{} first (last computed answer is most likely final)
    idx = text.rfind("\\boxed{")
    if idx == -1:
        idx = text.rfind("\\boxed {")
    if idx == -1:
        return None

    start = text.find("{", idx) + 1
    depth = 1
    pos   = start
    while pos < len(text) and depth > 0:
        if   text[pos] == "{": depth += 1
        elif text[pos] == "}": depth -= 1
        pos += 1

    return text[start:pos - 1].strip() if depth == 0 else None


def extract_predicted_answer(response: str) -> Optional[str]:
    """
    Smart answer extractor — 7 priority levels.

    WHY THIS ORDER MATTERS:
      After distillation the model reasons like R1: correct answer appears at
      token ~200-300, then re-verification loops may overwrite it.
      We want the FIRST confident conclusion, not the last number.
      The first 60% rule (levels 4-5) exploits this: loops start late.

    Priority:
      1. <solution>...</solution> tag  — our canonical format
      2. \\boxed{...} last occurrence  — LaTeX math format
      3. **Final Answer** block        — markdown bold header
      4. First "So/Therefore/Thus = X" in first 60% of response
      5. Last "= X" pattern in first 60% of response
      6. "answer/result/total is X" phrase, anywhere
      7. Last number in full response  (true fallback — least reliable)
    """
    if not response:
        return None
    if not isinstance(response, str):
        response = str(response)

    # 1. <solution> tag — highest priority, this is what we train for
    sol_m = re.search(r"<solution>(.*?)</solution>", response, re.DOTALL)
    if sol_m:
        content = sol_m.group(1).strip()
        # Try boxed inside solution first
        boxed = extract_boxed(content)
        if boxed:
            return boxed
        # Otherwise extract last number from solution content
        nums = re.findall(r"-?[\d,]+(?:\.\d+)?", content.replace(",", ""))
        if nums:
            return nums[-1]
        return content if content else None

    # 2. \boxed{...} — LaTeX format
    boxed = extract_boxed(response)
    if boxed:
        return boxed

    # 3. **Final Answer** block (markdown format)
    fa_m = re.search(r"\*\*[Ff]inal [Aa]nswer[:\s*]*\*\*(.*?)(?:\n|$)", response)
    if fa_m:
        content = fa_m.group(1).strip().rstrip("*").strip()
        nums = re.findall(r"-?[\d,]+(?:\.\d+)?", content.replace(",", ""))
        if nums:
            return nums[-1]

    # For levels 4-7: scan first 60% of the response only.
    # Loops typically start after 60% of tokens — the correct answer is usually
    # stated in the first clean reasoning pass, before any re-verification.
    words     = response.split()
    cutoff    = max(50, int(len(words) * 0.60))
    early     = " ".join(words[:cutoff])

    # 4. First confident conclusion: "So/Therefore/Thus, X = Y" or "= Y"
    concl_m = re.search(
        r"(?:so|therefore|thus|hence|answer is)[,\s]+(?:.*?=\s*)?(-?[\d,]+(?:\.\d+)?)",
        early, re.IGNORECASE
    )
    if concl_m:
        return concl_m.group(1).replace(",", "")

    # 5. Last "= NUMBER" pattern in early region
    eq_matches = re.findall(r"=\s*(-?[\d,]+(?:\.\d+)?)", early)
    if eq_matches:
        return eq_matches[-1].replace(",", "")

    # 6. "The answer is X" / "result is X" / "total is X" — anywhere in response
    ans_m = re.search(
        r"(?:the\s+)?(?:answer|result|total|value|solution)\s+is\s+(-?[\d,]+(?:\.\d+)?)",
        response, re.IGNORECASE
    )
    if ans_m:
        return ans_m.group(1).replace(",", "")

    # 7. Last number in full response — true last-resort fallback
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", response.replace(",", ""))
    return nums[-1] if nums else None


def normalize_answer(ans: str) -> str:
    """Normalize answer string for comparison. Handles units, commas, LaTeX."""
    if ans is None:
        return ""
    ans = str(ans).strip()

    # Strip common units
    ans = re.sub(
        r"\s*(dollars?|cents?|meters?|km|kg|cm|miles?|feet|inches?|%|degrees?)\s*$",
        "", ans, flags=re.IGNORECASE
    )
    # Remove commas in numbers
    ans = re.sub(r"(\d),(\d)", r"\1\2", ans)
    # Strip LaTeX formatting
    ans = re.sub(r"\\text\{([^}]*)\}", r"\1", ans)
    ans = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", ans)
    ans = re.sub(r"\\dfrac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", ans)
    ans = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", ans)
    ans = ans.replace("$", "").replace("\\", "").strip()

    # Normalize decimals: 1.50 → 1.5, 2.0 → 2
    try:
        f = float(ans)
        if f == int(f) and abs(f) < 1e12:
            return str(int(f))
        return f"{f:.8f}".rstrip("0").rstrip(".")
    except (ValueError, OverflowError):
        pass

    return ans.lower().strip()


def _try_numeric(a: str, b: str) -> Optional[bool]:
    """Compare as floats with relative tolerance."""
    try:
        fa, fb = float(a), float(b)
        tol = max(1e-6, 1e-4 * max(abs(fa), abs(fb)))
        return abs(fa - fb) < tol
    except (ValueError, OverflowError):
        return None


def _try_fraction(a: str, b: str) -> Optional[bool]:
    """Compare simple fractions: 3/4 == 0.75."""
    def to_float(s: str) -> Optional[float]:
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
    Handles: numeric equality, fractions, LaTeX, units, word answers.
    """
    if predicted is None or ground_truth is None:
        return False

    pred_n = normalize_answer(predicted)
    gt_n   = normalize_answer(ground_truth)

    if not pred_n or not gt_n:
        return False

    # Exact string match after normalization
    if pred_n == gt_n:
        return True

    # Numeric equality
    num_eq = _try_numeric(pred_n, gt_n)
    if num_eq is not None:
        return num_eq

    # Fraction equality
    frac_eq = _try_fraction(pred_n, gt_n)
    if frac_eq is not None:
        return frac_eq

    # Word answer normalization (yes/no, true/false)
    word_map = {"yes": "true", "no": "false", "1": "true", "0": "false"}
    p_w = word_map.get(pred_n, pred_n)
    g_w = word_map.get(gt_n, gt_n)
    if p_w == g_w:
        return True

    return False


def extract_ground_truth(raw: str, source: str) -> str:
    """Extract clean ground truth from raw dataset answer field."""
    if source == "gsm8k":
        parts = raw.split("####")
        return parts[-1].strip() if len(parts) > 1 else raw.strip()

    if source in ("math", "math_l3l4"):
        boxed = extract_boxed(raw)
        return boxed if boxed else raw.strip()

    if source == "numina":
        # NuminaMath: answer field is usually clean, or boxed
        boxed = extract_boxed(raw)
        if boxed:
            return boxed
        # "The answer is X"
        m = re.search(r"[Tt]he answer is[:\s]+(-?[\d\/\.\,\s]+)", raw)
        if m:
            return m.group(1).strip().rstrip(".")
        return raw.strip()

    if source == "deepscaler":
        # DeepScaleR: clean answer string or boxed
        boxed = extract_boxed(raw)
        return boxed if boxed else raw.strip()

    return raw.strip()


# ══════════════════════════════════════════════════════════════════════════════
# REWARD SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

# Phrases that trigger re-verification spiral (Type 1 loop)
# One occurrence is acceptable epistemic practice; 2+ is a loop
_REVERIF_PHRASES = [
    "let me check again",
    "let me verify again",
    "let me think again",
    "let me double-check",
    "let me re-check",
    "let me recalculate",
    "let me re-verify",
    "let me try again",
    "let me reconsider",
    "i need to recalculate",
    "i made an error, let me",
    "wait, let me redo",
]

# Phrases that signal the "Wait, no" catastrophic contradiction loop (Type 2)
_WAIT_PHRASES = [
    "wait, no, that's not possible",
    "wait, but that can't be right",
    "wait, that contradicts",
    "wait, that's wrong",
    "wait, that doesn't make sense",
    "wait, that's not correct",
    "no wait, that can't be",
    "hmm, that doesn't seem right, let me",
]


def _detect_repetition_trap(text: str, chunk_words: int = 50) -> int:
    """
    Detect infinite repetition trap (Type 3 loop).
    Splits response into non-overlapping 50-word chunks and counts
    near-duplicate pairs. Returns number of duplicate pair found.
    A count >= 2 indicates a repetition trap.
    """
    words  = text.split()
    chunks = [
        " ".join(words[i: i + chunk_words])
        for i in range(0, len(words) - chunk_words + 1, chunk_words)
    ]
    if len(chunks) < 2:
        return 0

    dup_count = 0
    seen      = {}
    for chunk in chunks:
        # Simple fingerprint: first 30 chars (fast, tolerates minor variation)
        fp = chunk[:30].lower().strip()
        if fp in seen:
            dup_count += 1
        seen[fp] = True

    return dup_count


class BinaryReward:
    """
    Binary reward: correct=1.0, wrong=0.0.

    Why binary only — backed by research:
      - DeepScaleR (43% AIME24):   binary reward only
      - JustRL    (58% avg):       binary reward only
      - Original DeepSeekMath:     binary reward only
      - Every working 1.5B recipe: binary reward only

    The loops, format issues, and verbosity we were trying to fix with
    5 reward components all resolve naturally as correctness improves.
    Adding non-correctness rewards creates conflicting gradients that
    push the model away from the distilled checkpoint without giving it
    a clear signal about what "better" looks like.

    Stats tracking kept for monitoring — we still want to see pass rate,
    response length trends, and batch variance during training.
    """

    def __init__(self):
        self._stats: Dict[str, List[float]] = defaultdict(list)
        self._max_stat_len = 500
        self.n_calls            = 0
        self.n_skipped_batches  = 0
        self.n_total_batches    = 0

    def compute(self, completion: str, ground_truth: str) -> float:
        """Binary: 1.0 if correct, 0.0 if wrong."""
        predicted  = extract_predicted_answer(completion)
        is_correct = verify_answer(predicted, ground_truth)
        reward     = 1.0 if is_correct else 0.0

        # Track stats
        self._log("correct",    float(is_correct))
        self._log("reward",     reward)
        self._log("resp_words", len(completion.split()))

        # Track format compliance — informational only, not used in reward
        has_think = "<think>"    in completion and "</think>"   in completion
        has_sol   = "<solution>" in completion and "</solution>"in completion
        clean_end = has_sol and len(completion.split("</solution>")[-1].strip().split()) <= 1
        self._log("has_tags",   float(has_think and has_sol))
        self._log("clean_term", float(clean_end))

        # Loop detection — informational only
        reverif = sum(completion.lower().count(p) for p in _REVERIF_PHRASES)
        self._log("reverif",    float(reverif))

        self.n_calls += 1
        return reward

    def compute_batch(self, completions: List[str],
                      ground_truths: List[str]) -> Tuple[List[float], bool]:
        """
        Compute rewards for a full group of G completions.
        Hard filter: completions with no <solution> tag get reward=0.0
        and are excluded from std calculation (treated as non-attempt,
        not as wrong answer). This keeps the gradient signal clean —
        the model learns from correct vs wrong, not correct vs unformatted.
        """
        rewards = []
        valid_rewards = []  # only completions that attempted an answer

        for c, gt in zip(completions, ground_truths):
            has_solution = "<solution>" in c and "</solution>" in c
            if not has_solution:
                # Hard filter: no tag = no gradient signal, just 0
                rewards.append(0.0)
                self.compute(c, gt)  # still log stats
            else:
                r = self.compute(c, gt)
                rewards.append(r)
                valid_rewards.append(r)

        self.n_total_batches += 1

        # Compute std only over valid (tagged) completions
        # If all completions lack tags, skip batch
        if len(valid_rewards) < 2:
            self.n_skipped_batches += 1
            return [0.0] * len(rewards), True

        mean_r = sum(valid_rewards) / len(valid_rewards)
        var_r  = sum((r - mean_r) ** 2 for r in valid_rewards) / len(valid_rewards)
        std_r  = math.sqrt(var_r)
        self._log("batch_std", std_r)

        if std_r < 0.001:
            self.n_skipped_batches += 1
            return [0.0] * len(rewards), True

        return rewards, False

    def _log(self, key: str, value: float):
        lst = self._stats[key]
        lst.append(value)
        if len(lst) > self._max_stat_len:
            lst.pop(0)

    def get_summary(self, last_n: int = 200) -> Dict:
        def avg(key):
            lst = self._stats.get(key, [])
            tail = lst[-last_n:] if len(lst) >= last_n else lst
            return sum(tail) / max(len(tail), 1)

        skip_rate = self.n_skipped_batches / max(self.n_total_batches, 1) * 100
        return {
            "pass_rate":       avg("correct"),
            "mean_reward":     avg("correct"),  # binary: mean(correct) = pass rate
            "mean_words":      avg("resp_words"),
            "has_tags_rate":   avg("has_tags"),
            "clean_term_rate": avg("clean_term"),
            "mean_reverif":    avg("reverif"),
            "batch_std_mean":  avg("batch_std"),
            "skip_rate_pct":   skip_rate,
            "n_skipped":       self.n_skipped_batches,
            "n_batches":       self.n_total_batches,
            "n_calls":         self.n_calls,
        }

    def print_summary(self, step: int, last_n: int = 200):
        s = self.get_summary(last_n)
        print(f"\n  ── Reward Summary @ step {step} (last {last_n} completions) ──")
        print(f"    Pass@1 (correct)    : {s['pass_rate']*100:5.1f}%")
        print(f"    Mean reward (pass%)  : {s['mean_reward']*100:5.1f}%  (binary reward = pass rate)")
        print(f"    Mean response words : {s['mean_words']:.0f}")
        print(f"    Has tags rate       : {s['has_tags_rate']*100:5.1f}%  (informational)")
        print(f"    Clean term rate     : {s['clean_term_rate']*100:5.1f}%  (informational)")
        print(f"    Mean re-verif count : {s['mean_reverif']:.2f}  (informational)")
        print(f"    Batch skip rate     : {s['skip_rate_pct']:.1f}%  "
              f"({s['n_skipped']}/{s['n_batches']})")
        print(f"    Batch reward std    : {s['batch_std_mean']:.3f}")

        if s["pass_rate"] < 0.05 and self.n_calls > 400:
            print("    ⚠️  WARN: Pass rate < 5% — problems may be too hard for this model")
        if s["skip_rate_pct"] > 40:
            print("    ⚠️  WARN: >40% batches skipped — shift to easier problems")
        if s["pass_rate"] > 0.80:
            print("    ⚠️  WARN: Pass rate > 80% — problems too easy, shift harder")


    """
    Multi-component reward for GRPO Phase A v2.

    Key design principles vs v1:
      - Correctness dominates everything: +1.0 / -0.3
      - Termination is second-priority: model must learn to STOP cleanly
      - Anti-loop is equally important: loops corrupt otherwise-correct answers
      - Length is a gentle nudge, NOT a driver (w=0.1, correct answers only)
      - Format is the weakest signal (w=0.05) — model already knows the format

    Stats tracking:
      - All per-component rewards logged separately for debugging
      - Degenerate batch detection (std < threshold → skip)
      - Termination rate tracked to verify reward is working
    """

    def __init__(self,
                 w_correct:   float = 1.0,
                 w_terminate: float = 0.20,
                 w_antiloop:  float = 0.25,
                 w_length:    float = 0.10,
                 w_format:    float = 0.05,
                 skip_std_threshold: float = 0.01):
        self.w_correct            = w_correct
        self.w_terminate          = w_terminate
        self.w_antiloop           = w_antiloop
        self.w_length             = w_length
        self.w_format             = w_format
        self.skip_std_threshold   = skip_std_threshold

        # Running stats for monitoring (ring buffer logic: keep last 500)
        self._stats: Dict[str, List[float]] = defaultdict(list)
        self._max_stat_len = 500
        self.n_calls       = 0
        self.n_skipped_batches = 0
        self.n_total_batches   = 0

    # ── Component: Correctness ───────────────────────────────────────────────

    def _r_correct(self, completion: str, gt: str) -> Tuple[float, bool]:
        """Returns (reward, is_correct). Core signal.
        Wrong penalty is -0.10 (not -0.30) — we're training termination,
        not correctness. Harsh wrong penalties cause the model to abandon
        reasoning structure entirely to avoid the signal.
        """
        predicted  = extract_predicted_answer(completion)
        is_correct = verify_answer(predicted, gt)
        r = self.w_correct * (1.0 if is_correct else -0.10)
        return r, is_correct

    # ── Component: Termination ───────────────────────────────────────────────

    def _r_terminate(self, completion: str) -> float:
        """
        Reward clean termination structure:
          BEST:    <think>...</think><solution>X</solution>  [STOP]
          OK:      <solution> tag present but model keeps generating after
          BAD:     No <solution> tag at all

        The key invariant we want: nothing meaningful after </solution>.
        """
        has_sol_open  = "<solution>"  in completion
        has_sol_close = "</solution>" in completion

        if not (has_sol_open and has_sol_close):
            # No solution tag at all — worst case
            return -self.w_terminate

        # Check how much content follows </solution>
        # Use WORD COUNT not char count — "Actually, maybe 6?" is 18 chars but 3 words
        # 0 words = only whitespace/EOS after tag (ideal)
        # 1 word  = maybe a stray punctuation token (tolerate)
        # 2+ words = model is genuinely continuing (penalise)
        after_sol  = completion.split("</solution>")[-1].strip()
        clean_stop = len(after_sol.split()) <= 1

        if clean_stop:
            return +self.w_terminate      # perfect clean termination
        else:
            # Tag present but model keeps generating after it
            # Partial credit — better than no tag, worse than clean stop
            return +self.w_terminate * 0.25

    # ── Component: Anti-Loop ─────────────────────────────────────────────────

    def _r_anti_loop(self, completion: str) -> Tuple[float, Dict]:
        """
        Penalise all three loop types identified in Stage 3 diagnostic.

        Graduated penalties:
          Type 1 (re-verification spiral):
            0 occurrences  → +0.05 bonus (clean reasoning)
            1 occurrence   → 0.00  (one check is fine)
            2 occurrences  → -0.40 × w_antiloop
            3+ occurrences → -0.80 × w_antiloop

          Type 2 ("Wait, no" catastrophic contradiction):
            0 occurrences  → no effect (handled by Type 1 bonus)
            1 occurrence   → -0.60 × w_antiloop
            2+ occurrences → -1.00 × w_antiloop  ← worst case, full penalty

          Type 3 (infinite repetition trap):
            0 dup pairs    → no effect
            1 dup pair     → -0.40 × w_antiloop
            2+ dup pairs   → -0.60 × w_antiloop

        The final r_loop is the minimum of all applicable penalties
        (i.e. multiple loop types stack to the worst applicable penalty,
        but do not compound beyond -w_antiloop total).
        """
        comp_lower = completion.lower()

        # Count Type 1: re-verification spiral
        reverif_count = sum(comp_lower.count(p) for p in _REVERIF_PHRASES)

        # Count Type 2: "Wait, no" catastrophic loop
        wait_count = sum(comp_lower.count(p) for p in _WAIT_PHRASES)

        # Detect Type 3: repetition trap
        rep_count = _detect_repetition_trap(completion)

        # ── Compute penalty ──
        # Start with the clean-reasoning bonus
        r = +0.05 if (reverif_count == 0 and wait_count == 0 and rep_count == 0) else 0.0

        # Type 1: re-verification spiral
        if reverif_count == 1:
            r = min(r, 0.00)                                   # one check OK, no bonus
        elif reverif_count == 2:
            r = min(r, -self.w_antiloop * 0.40)
        elif reverif_count >= 3:
            r = min(r, -self.w_antiloop * 0.80)

        # Type 2: catastrophic contradiction loop (overrides Type 1 if worse)
        if wait_count == 1:
            r = min(r, -self.w_antiloop * 0.60)
        elif wait_count >= 2:
            r = min(r, -self.w_antiloop * 1.00)                # full penalty

        # Type 3: repetition trap
        if rep_count == 1:
            r = min(r, -self.w_antiloop * 0.40)
        elif rep_count >= 2:
            r = min(r, -self.w_antiloop * 0.60)

        # Diagnostic dict for monitoring
        diag = {
            "reverif_count": reverif_count,
            "wait_count":    wait_count,
            "rep_count":     rep_count,
        }
        return r, diag

    # ── Component: Length Efficiency ─────────────────────────────────────────

    def _r_length(self, completion: str, is_correct: bool) -> float:
        """
        Gentle nudge toward concise responses.

        CRITICAL CONSTRAINT: zero weight on incorrect responses.
        If we penalise length on wrong answers, the model learns:
        "give a short wrong answer → avoid length penalty"
        → actively harmful. Length signal must be correctness-gated.

        We do NOT use HAPO (history-aware minimum tracking) here.
        HAPO with w=0.3 caused regression in v1 by overriding correctness.
        """
        if not is_correct:
            return 0.0

        words = len(completion.split())
        if   words < 300: return +self.w_length * 1.0    # concise and correct
        elif words < 500: return +self.w_length * 0.5    # moderate length
        elif words < 700: return  0.0                     # neutral zone
        elif words < 900: return -self.w_length * 0.5    # getting long
        else:             return -self.w_length * 1.0    # overthinking

    # ── Component: Format ────────────────────────────────────────────────────

    def _r_format(self, completion: str) -> float:
        """
        Check for <think>...</think><solution>...</solution> structure.
        Format compliance is already high (~99%) from SFT/distillation,
        so this is a weak maintenance signal, not a driver.
        """
        has_think_o = "<think>"    in completion
        has_think_c = "</think>"   in completion
        has_sol_o   = "<solution>" in completion
        has_sol_c   = "</solution>"in completion

        all_four  = has_think_o and has_think_c and has_sol_o and has_sol_c
        some_tags = has_think_o or has_sol_o

        if all_four:   return +self.w_format           # all tags present
        elif some_tags: return -self.w_format * 1.0    # partial structure
        else:           return -self.w_format * 3.0    # completely unformatted

    # ── Total Reward ─────────────────────────────────────────────────────────

    def compute(self, completion: str, ground_truth: str) -> float:
        """Compute total reward for a single (completion, ground_truth) pair."""
        r_correct, is_correct      = self._r_correct(completion, ground_truth)
        r_terminate                = self._r_terminate(completion)
        r_loop, loop_diag          = self._r_anti_loop(completion)
        r_length                   = self._r_length(completion, is_correct)
        r_format                   = self._r_format(completion)

        total = r_correct + r_terminate + r_loop + r_length + r_format

        # Track all components for monitoring
        self._log("correct",       float(is_correct))
        self._log("r_correct",     r_correct)
        self._log("r_terminate",   r_terminate)
        self._log("r_loop",        r_loop)
        self._log("r_length",      r_length)
        self._log("r_format",      r_format)
        self._log("r_total",       total)
        self._log("resp_words",    len(completion.split()))
        self._log("reverif_count", loop_diag["reverif_count"])
        self._log("wait_count",    loop_diag["wait_count"])
        self._log("rep_count",     loop_diag["rep_count"])

        # Track termination rate
        has_clean_term = ("<solution>" in completion and "</solution>" in completion
                          and len(completion.split("</solution>")[-1].strip()) < 20)
        self._log("clean_term",    float(has_clean_term))

        self.n_calls += 1
        return total

    def _log(self, key: str, value: float):
        """Append to rolling stats buffer."""
        lst = self._stats[key]
        lst.append(value)
        if len(lst) > self._max_stat_len:
            lst.pop(0)


# Global reward object (must persist across training steps for stats)
_reward: Optional[BinaryReward] = None


def make_reward_fn(reward_obj: BinaryReward):
    """
    Factory for GRPOTrainer-compatible reward function.
    Binary reward: 1.0 if correct, 0.0 if wrong. Nothing else.
    """
    def reward_fn(prompts: List[str],
                  completions: List[str],
                  **kwargs) -> List[float]:

        ground_truths = kwargs.get("ground_truth", [""] * len(completions))

        def to_str(x) -> str:
            if isinstance(x, str): return x
            if isinstance(x, (list, tuple)):
                try:    return reward_obj._tokenizer.decode(x, skip_special_tokens=True)
                except: return " ".join(str(t) for t in x)
            if hasattr(x, "item"): return str(x.item())
            return str(x)

        completions   = [to_str(c) for c in completions]
        ground_truths = [str(g) for g in ground_truths]

        rewards, skip = reward_obj.compute_batch(completions, list(ground_truths))
        return rewards

    return reward_fn




# ══════════════════════════════════════════════════════════════════════════════
# DATASET LOADING
# No GSM8K. Competition problems only. Difficulty window: 20–60% pass rate.
# ══════════════════════════════════════════════════════════════════════════════

def problem_id(text: str) -> str:
    """Stable 16-char hash of problem text for deduplication."""
    return hashlib.md5(text.encode()).hexdigest()[:16]


def make_chat_prompt(problem: str) -> List[Dict]:
    """Format problem as chat messages list for GRPOTrainer."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": problem.strip()},
    ]


def _load_deepscaler(n: int, sanity: bool, cache: str) -> Dataset:
    """
    DeepScaleR AMC subset.
    Expected pass rate on stage3_distilled_merged: ~40–60%.
    This is the sweet spot — model gets roughly half right.

    We use the AMC subset (amc10, amc_2022, amc_2023, amc10_2023)
    rather than AIME subset because AIME is too hard for Phase A
    (model solves ~15–20%, too many all-wrong batches → too many skips).
    """
    print("  Loading DeepScaleR (AMC subset)...")

    local_path = WORK_DIR / "data" / "deepscaler"
    try:
        if local_path.exists():
            ds = load_dataset(str(local_path), split="train")
        else:
            ds = load_dataset(
                "agentica-org/DeepScaleR-Preview-Dataset",
                cache_dir=cache,
            )
            if hasattr(ds, "keys"):
                ds = ds["train"] if "train" in ds else list(ds.values())[0]
    except Exception as e:
        print(f"  ⚠️  DeepScaleR load failed: {e}")
        print(f"     Download: huggingface-cli download agentica-org/DeepScaleR-Preview-Dataset "
              f"--repo-type dataset --local-dir ~/nlp/data/deepscaler")
        return Dataset.from_dict({"prompt": [], "ground_truth": [], "problem_id": [], "source": []})

    # Use AMC10-style problems for Phase A (harder than GSM8K, not as hard as AIME)
    try:
        amc_sources = {"amc10", "amc_2022", "amc_2023", "amc10_2023",
                       "amc12_2022", "amc12_2023", "amc_2021", "amc10_2021"}
        ds_filtered = ds.filter(lambda x: x.get("source", "") in amc_sources)
        if len(ds_filtered) < 500:
            # Fallback: if source field differs, take first n directly
            ds_filtered = ds
        ds = ds_filtered
    except Exception:
        pass  # source field missing or different format — use all

    n_actual = min(10 if sanity else n, len(ds))
    ds = ds.shuffle(seed=42).select(range(n_actual))

    def process(ex):
        prob   = ex.get("problem", ex.get("question", ""))
        raw_gt = str(ex.get("answer", ex.get("solution", "")))
        gt     = extract_ground_truth(raw_gt, "deepscaler")
        return {
            "prompt":       make_chat_prompt(prob),
            "ground_truth": gt,
            "problem_id":   problem_id(prob),
            "source":       "deepscaler",
        }

    ds = ds.map(process, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: 0 < len(x["ground_truth"]) < 200)
    print(f"  ✅ DeepScaleR AMC: {len(ds):,}")
    return ds


def _load_math_l1l2(n: int, sanity: bool, cache: str) -> Dataset:
    """
    MATH Levels 1–2. Model solves ~65-75% of these after distillation.
    Purpose: ensure enough mixed batches (some correct, some wrong per group)
    so reward_std stays above skip threshold.
    These are NOT the primary training signal — L3-4 and DeepScaleR are.
    Think of L1-2 as the "training wheels" that keep gradient flowing.
    """
    print("  Loading MATH Level 1–2 (mixed-batch anchor)...")

    ds = _load_math_dataset(split="train", cache=cache, silent=True)
    if ds is None:
        print("  ⚠️  MATH L1-2 skipped — dataset not available")
        return Dataset.from_dict({"prompt": [], "ground_truth": [], "problem_id": [], "source": []})

    try:
        ds = ds.filter(lambda x: x.get("level", "") in ["Level 1", "Level 2"])
    except Exception:
        pass

    n_actual = min(5 if sanity else n, len(ds))
    if n_actual == 0:
        return Dataset.from_dict({"prompt": [], "ground_truth": [], "problem_id": [], "source": []})
    ds = ds.shuffle(seed=42).select(range(n_actual))

    def process(ex):
        prob = ex.get("problem", ex.get("question", ""))
        sol  = ex.get("solution", ex.get("answer", ""))
        gt   = extract_ground_truth(sol, "math_l3l4")
        return {
            "prompt":       make_chat_prompt(prob),
            "ground_truth": gt,
            "problem_id":   problem_id(prob),
            "source":       "math_l1l2",
        }

    ds = ds.map(process, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: 0 < len(x["ground_truth"]) < 150)
    print(f"  ✅ MATH L1–2: {len(ds):,}")
    return ds



def _load_metamathqa(n: int, sanity: bool, cache: str) -> Dataset:
    """
    MetaMathQA — augmented GSM8K/MATH with rephrased + augmented problems.
    Model solves ~55% of these after distillation → ideal GRPO signal zone.
    These are the PRIMARY training signal for Phase A v3.

    Why MetaMath works better than DeepScaleR for a 1.5B model:
      - Pass rate ~55% = near-perfect for GRPO (mixed correct/wrong per batch)
      - Problems are GSM8K-style but rephrased → model can't just memorize
      - Reward std consistently > 0.3 on these batches
    """
    print("  Loading MetaMathQA (primary signal ~55% pass rate)...")
    try:
        ds = load_dataset("meta-math/MetaMathQA", split="train", cache_dir=cache)
    except Exception as e:
        print(f"  ⚠️  MetaMathQA failed: {e}")
        return Dataset.from_dict({"prompt": [], "ground_truth": [], "problem_id": [], "source": []})

    # Filter to rephrased/augmented variants only — avoid exact GSM8K duplicates
    # These types are harder than vanilla GSM8K but model still solves ~50-60%
    try:
        keep_types = {
            "GSM_Rephrased", "MATH_Rephrased",
            "GSM_AnsAug",    "MATH_AnsAug",
            "GSM_FOBAR",     "GSM_SV",
        }
        ds = ds.filter(lambda x: x.get("type", "") in keep_types)
    except Exception:
        pass

    n_actual = min(10 if sanity else n, len(ds))
    ds = ds.shuffle(seed=42).select(range(n_actual))

    def process(ex):
        prob   = ex.get("query", "")
        raw_gt = ex.get("response", "")
        # MetaMath answers: "The answer is: X" or "#### X"
        gt = ""
        if "####" in raw_gt:
            gt = raw_gt.split("####")[-1].strip()
        else:
            m = re.search(r"[Tt]he answer is[:\s]+(-?[\d,\.]+)", raw_gt)
            gt = m.group(1).replace(",","").strip() if m else ""
        if not gt:
            nums = re.findall(r"-?[\d,]+(?:\.\d+)?", raw_gt.replace(",",""))
            gt = nums[-1] if nums else ""
        return {
            "prompt":       make_chat_prompt(prob),
            "ground_truth": gt,
            "problem_id":   problem_id(prob),
            "source":       "metamath",
        }

    ds = ds.map(process, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: 0 < len(x["ground_truth"]) < 100)
    print(f"  ✅ MetaMathQA: {len(ds):,}")
    return ds


def _load_gsm8k_hard(n: int, sanity: bool, cache: str) -> Dataset:
    """
    GSM8K hardest ~20% of problems by solution length.
    Proxy for difficulty: longer solutions = more steps = harder.
    Model solves ~45% of these → good mixed batches.

    We use solution LENGTH as a difficulty proxy because GSM8K has no
    difficulty labels. Problems requiring 6+ steps are harder than 2-step ones.
    """
    print("  Loading GSM8K hard subset (~45% pass rate)...")
    try:
        ds = load_dataset("gsm8k", "main", split="train", cache_dir=cache)
    except Exception as e:
        print(f"  ⚠️  GSM8K failed: {e}")
        return Dataset.from_dict({"prompt": [], "ground_truth": [], "problem_id": [], "source": []})

    # Sort by solution length descending, take top 20% (the harder ones)
    # Solution length correlates with number of steps required
    try:
        ds = ds.map(lambda x: {"sol_len": len(x["answer"].split())})
        ds = ds.sort("sol_len", reverse=True)
        hard_n = max(100, int(len(ds) * 0.20))   # top 20% = ~1495 problems
        ds = ds.select(range(hard_n))
    except Exception:
        pass  # if sort fails, use full dataset

    n_actual = min(10 if sanity else n, len(ds))
    ds = ds.shuffle(seed=42).select(range(n_actual))

    def process(ex):
        prob   = ex["question"]
        raw_gt = ex["answer"]
        parts  = raw_gt.split("####")
        gt     = parts[-1].strip() if len(parts) > 1 else raw_gt.strip()
        return {
            "prompt":       make_chat_prompt(prob),
            "ground_truth": gt,
            "problem_id":   problem_id(prob),
            "source":       "gsm8k_hard",
        }

    ds = ds.map(process, remove_columns=[c for c in ds.column_names if c != "problem_id"])
    ds = ds.filter(lambda x: 0 < len(x["ground_truth"]) < 50)
    print(f"  ✅ GSM8K hard: {len(ds):,}")
    return ds


def _load_math_l2l3(n: int, sanity: bool, cache: str) -> Dataset:
    """
    MATH Levels 2-3.
    Model solves ~40-65% of these → core difficulty target for Phase A.
    L2 gives the easy end (65%), L3 gives the harder end (40%).
    Together they span the ideal 40-65% pass rate window.
    """
    print("  Loading MATH Level 2–3 (40-65% pass rate)...")

    ds = _load_math_dataset(split="train", cache=cache, silent=True)
    if ds is None:
        print("  ⚠️  MATH L2-3 skipped — dataset not available")
        return Dataset.from_dict({"prompt": [], "ground_truth": [], "problem_id": [], "source": []})

    try:
        ds = ds.filter(lambda x: x.get("level", "") in ["Level 2", "Level 3"])
    except Exception:
        pass

    n_actual = min(10 if sanity else n, len(ds))
    if n_actual == 0:
        return Dataset.from_dict({"prompt": [], "ground_truth": [], "problem_id": [], "source": []})
    ds = ds.shuffle(seed=42).select(range(n_actual))

    def process(ex):
        prob = ex.get("problem", ex.get("question", ""))
        sol  = ex.get("solution", ex.get("answer", ""))
        gt   = extract_ground_truth(sol, "math_l3l4")
        return {
            "prompt":       make_chat_prompt(prob),
            "ground_truth": gt,
            "problem_id":   problem_id(prob),
            "source":       "math_l2l3",
        }

    ds = ds.map(process, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: 0 < len(x["ground_truth"]) < 150)
    print(f"  ✅ MATH L2–3: {len(ds):,}")
    return ds


def _load_math_l3l4(n: int, sanity: bool, cache: str) -> Dataset:
    """
    MATH Levels 3–4.
    Expected pass rate on stage3_distilled_merged: ~35–55%.
    These are the core difficulty problems: harder than GSM8K,
    easier than AIME, exactly where the model needs improvement.
    """
    print("  Loading MATH Level 3–4...")

    ds = _load_math_dataset(split="train", cache=cache, silent=False)

    if ds is None:
        print("  ⚠️  Could not load MATH dataset from any source.")
        print("  ── Download it manually:")
        print("     huggingface-cli download EleutherAI/hendrycks_math \\")
        print("       --repo-type dataset --local-dir ~/nlp/data/math")
        print("  ── Continuing without MATH — DeepScaleR + NuminaMath only.")
        return Dataset.from_dict({"prompt": [], "ground_truth": [], "problem_id": [], "source": []})

    # Filter to levels 3 and 4 only
    # Level 1-2: too easy after distillation (>70% pass rate → dead signal)
    # Level 5: too hard for Phase A (model solves <20%)
    try:
        ds = ds.filter(lambda x: x.get("level", "Level 3") in ["Level 3", "Level 4"])
    except Exception:
        pass

    n_actual = min(15 if sanity else n, len(ds))
    ds = ds.shuffle(seed=42).select(range(n_actual))

    def process(ex):
        prob = ex.get("problem", ex.get("question", ""))
        sol  = ex.get("solution", ex.get("answer", ""))
        gt   = extract_ground_truth(sol, "math_l3l4")
        return {
            "prompt":       make_chat_prompt(prob),
            "ground_truth": gt,
            "problem_id":   problem_id(prob),
            "source":       "math_l3l4",
        }

    ds = ds.map(process, remove_columns=ds.column_names)
    # Filter: ground truth must be non-empty and reasonably short
    ds = ds.filter(lambda x: 0 < len(x["ground_truth"]) < 150)
    print(f"  ✅ MATH L3–4: {len(ds):,}")
    return ds


def _load_numina_competition(n: int, sanity: bool, cache: str) -> Dataset:
    """
    NuminaMath competition problems (amc_aime + olympiads subset).
    Expected pass rate on stage3_distilled_merged: ~20–40%.
    These are the hardest problems in Phase A — they push the ceiling.

    We only take amc_aime and olympiads sources (not cn_contest) because:
    - cn_contest problems sometimes have ambiguous answer formats
    - amc_aime and olympiads have cleaner, more verifiable answers
    """
    print("  Loading NuminaMath competition subset...")

    try:
        ds = load_dataset("AI-MO/NuminaMath-CoT", split="train", cache_dir=cache)
    except Exception as e:
        print(f"  ⚠️  NuminaMath load failed: {e}")
        return Dataset.from_dict({"prompt": [], "ground_truth": [], "problem_id": [], "source": []})

    # Competition sources only — skip cn_contest for cleaner answers
    try:
        comp_sources = {"amc_aime", "olympiads", "aops_forum"}
        ds = ds.filter(lambda x: x.get("source", "") in comp_sources)
    except Exception:
        pass

    n_actual = min(10 if sanity else n, len(ds))
    ds = ds.shuffle(seed=42).select(range(n_actual))

    def process(ex):
        prob   = ex.get("problem", ex.get("question", ""))
        raw_gt = ex.get("solution", ex.get("answer", ""))
        gt     = extract_ground_truth(raw_gt, "numina")
        return {
            "prompt":       make_chat_prompt(prob),
            "ground_truth": gt,
            "problem_id":   problem_id(prob),
            "source":       "numina",
        }

    ds = ds.map(process, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: 0 < len(x["ground_truth"]) < 200)
    print(f"  ✅ NuminaMath competition: {len(ds):,}")
    return ds


def build_dataset(args) -> Dataset:
    """
    Build the Phase A GRPO dataset — v3 difficulty-calibrated mix.

    Problem: original mix (DeepScaleR + MATH L3-4) had ~7.5% pass rate.
    Research (GHPO, GRPO-LEAD) confirms this causes reward sparsity:
    all-8-wrong batches → std=0 → zero gradient → model drifts without learning.

    Fix: use problems where model currently solves 30-70%.
    This ensures mixed batches (some correct, some wrong) = high reward_std
    = actual gradient signal every step.

    New mix (target pass rates based on stage3_distilled_merged):
      MetaMathQA       8K  ~55% pass  ← PRIMARY signal, ideal GRPO zone
      GSM8K hard       4K  ~45% pass  ← good mixed batches
      MATH L2-3        6K  ~50% pass  ← core difficulty
      MATH L4          2K  ~20% pass  ← ceiling pusher (keep small)
      NuminaMath       2K  ~20% pass  ← ceiling pusher (keep small)
      DeepScaleR        0  ~8%  pass  ← DISABLED (reward sparsity)

    Total: ~22K problems, expected overall pass rate ~45%
    Expected batch_skip: <10% (was 51% in v2 with old mix)
    """
    print(f"\n{'─'*60}")
    print("  Building Phase A dataset — v3 difficulty-calibrated mix")
    print("  Target: 30-70% pass rate per problem type")
    print(f"{'─'*60}")

    cache  = str(DATA_CACHE)
    sanity = args.sanity
    parts  = []

    # PRIMARY: MetaMathQA ~55% pass rate
    meta = _load_metamathqa(args.n_metamath, sanity, cache)
    if len(meta) > 0:
        parts.append(meta)

    # SECONDARY: GSM8K hard ~45% pass rate
    gsm_hard = _load_gsm8k_hard(args.n_gsm8k_hard, sanity, cache)
    if len(gsm_hard) > 0:
        parts.append(gsm_hard)

    # CORE: MATH L2-3 ~50% pass rate
    math_l2l3 = _load_math_l2l3(args.n_math_l2l3, sanity, cache)
    if len(math_l2l3) > 0:
        parts.append(math_l2l3)

    # CEILING: MATH L4 ~20% pass rate (keep small — just enough to push ceiling)
    math_l4 = _load_math_l3l4(args.n_math, sanity, cache)
    if len(math_l4) > 0:
        parts.append(math_l4)

    # CEILING: NuminaMath competition ~20% pass rate (keep small)
    if args.n_numina > 0:
        numina = _load_numina_competition(args.n_numina, sanity, cache)
        if len(numina) > 0:
            parts.append(numina)

    # DISABLED: DeepScaleR — ~8% pass rate = reward sparsity
    # Re-enable in Phase B after termination is learned
    if args.n_deepscaler > 0:
        deep = _load_deepscaler(args.n_deepscaler, sanity, cache)
        if len(deep) > 0:
            parts.append(deep)

    # FALLBACK: MATH L1-2 if MetaMath failed to load
    if args.n_math_easy > 0:
        math_easy = _load_math_l1l2(args.n_math_easy, sanity, cache)
        if len(math_easy) > 0:
            parts.append(math_easy)

    if not parts:
        raise RuntimeError(
            "All datasets failed to load.\n"
            "At minimum gsm8k or meta-math/MetaMathQA must be available."
        )

    combined = concatenate_datasets(parts)

    # Deduplication
    seen_ids = set()
    def is_unique(ex):
        pid = ex["problem_id"]
        if pid in seen_ids: return False
        seen_ids.add(pid); return True
    combined = combined.filter(is_unique)

    # Shuffle — mixed difficulty throughout, not curriculum sorted
    combined = combined.shuffle(seed=42)

    print(f"\n  ── Dataset Summary ──")
    print(f"  Total problems (deduplicated): {len(combined):,}")
    src_counts = Counter(combined["source"])
    for src, cnt in src_counts.most_common():
        pct = cnt / len(combined) * 100
        expected = {
            "metamath":    "~55%", "gsm8k_hard": "~45%",
            "math_l2l3":   "~50%", "math_l3l4":  "~20%",
            "math_l1l2":   "~65%", "numina":      "~20%",
            "deepscaler":  "~8%",
        }.get(src, "?")
        print(f"  {src:<20}: {cnt:>6,}  ({pct:.1f}%)  expected pass rate: {expected}")
    print(f"{'─'*60}")

    return combined



# ── Shared MATH dataset loader (used by training loader + eval callbacks) ──────

# All known HF names for the MATH benchmark, tried in order
_MATH_HF_NAMES = [
    "lighteval/MATH",
    "hendrycks/competition_math",
    "EleutherAI/hendrycks_math",
    "HuggingFaceH4/MATH-500",
    "DigitalLearningGmbH/MATH-lighteval",
    "competition_math",
]

# Local paths where it might be cached on the server
_MATH_LOCAL_PATHS = [
    WORK_DIR / "data" / "math",
    WORK_DIR / "data" / "MATH",
    WORK_DIR / "data" / "competition_math",
    WORK_DIR / "data" / "cache" / "math",
]


def _load_math_dataset(split: str = "train", cache: str = str(DATA_CACHE),
                        silent: bool = False) -> Optional[Dataset]:
    """
    Robust MATH dataset loader with full fallback chain.
    Tries local paths first (no network), then HF Hub names.
    Returns None if all sources fail.
    """
    # Local paths first
    for local_path in _MATH_LOCAL_PATHS:
        if local_path.exists():
            try:
                ds = load_dataset(str(local_path), split=split)
                if not silent:
                    print(f"  ✅ MATH loaded from local path: {local_path}")
                return ds
            except Exception:
                continue

    # HF Hub
    for name in _MATH_HF_NAMES:
        try:
            ds = load_dataset(name, split=split, cache_dir=cache)
            if not silent:
                print(f"  ✅ MATH loaded from HF: {name}")
            return ds
        except Exception as e:
            short = str(e)[:70].replace("\n", " ")
            if not silent:
                print(f"  ⚠️  {name}: {short}")
            continue

    return None



def _forced_think_generate(model, tokenizer, enc: dict, max_new_tokens: int, device) -> str:
    """
    Force generation to start with <think> prefix.

    Why needed:
      GRPO trains at temperature=0.8 (sampling). Eval uses greedy (argmax).
      After partial GRPO training with negative rewards, the argmax head may
      land on a direct-response mode (no tags) while the sampling distribution
      still produces think-tagged responses.
      Forcing <think> guarantees the correct format for fair eval numbers.
    """
    prefix_ids = tokenizer(
        "<think>\n", return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)

    full_ids  = torch.cat([enc["input_ids"],      prefix_ids],              dim=1)
    full_mask = torch.cat([enc["attention_mask"],  torch.ones_like(prefix_ids)], dim=1)

    with torch.no_grad():
        out = model.generate(
            input_ids=full_ids,
            attention_mask=full_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = out[0][full_ids.shape[1]:]
    return "<think>\n" + tokenizer.decode(new_tokens, skip_special_tokens=True)


class RewardMonitorCallback(TrainerCallback):
    """
    Prints reward component breakdown and health checks every N steps.
    Saves running stats to JSON for post-training analysis.
    """

    def __init__(self, reward_obj: BinaryReward, log_path: Path,
                 check_every: int = 25):
        self.rc          = reward_obj
        self.log_path    = log_path
        self.check_every = check_every
        self.history     = []
        self.best_pass   = 0.0
        self.best_term   = 0.0

    def on_step_end(self, args, state: TrainerState,
                    control: TrainerControl, **kwargs):
        step = state.global_step
        if step == 0 or step % self.check_every != 0:
            return control

        self.rc.print_summary(step, last_n=self.check_every * 8)
        summary = self.rc.get_summary(last_n=self.check_every * 8)
        entry   = {"step": step, **summary, "timestamp": datetime.now().isoformat()}
        self.history.append(entry)

        # Save to disk
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump(self.history, f, indent=2)

        # Track improvements
        if summary["pass_rate"] > self.best_pass:
            self.best_pass = summary["pass_rate"]
            print(f"    ⭐ New best pass rate: {self.best_pass*100:.1f}%")
        if summary["clean_term_rate"] > self.best_term:
            self.best_term = summary["clean_term_rate"]
            print(f"    ⭐ New best termination rate: {self.best_term*100:.1f}%")

        return control


class Math500EvalCallback(TrainerCallback):
    """
    Runs MATH500 evaluation every eval_every steps.
    MATH500 is the primary eval for Phase A — GSM8K is saturated at 85%.

    Uses greedy decoding (temperature=0) for deterministic eval.
    Uses smart extractor (same as reward function) for consistent measurement.
    """

    def __init__(self, tokenizer, log_path: Path,
                 eval_every: int = 100, n_eval: int = 50,
                 max_new_tokens: int = 1500, sanity: bool = False):
        self.tokenizer      = tokenizer
        self.log_path       = log_path
        self.eval_every     = eval_every
        self.n_eval         = n_eval
        self.max_new_tokens = max_new_tokens
        self.sanity         = sanity
        self.history        = []
        self.best_acc       = 0.0
        self.best_step      = 0

    def on_step_end(self, args, state: TrainerState,
                    control: TrainerControl, model=None, **kwargs):
        step = state.global_step
        if step == 0 or step % self.eval_every != 0:
            return control
        print(f"\n{'═'*60}")
        print(f"  MATH500 eval @ step {step} ({self.n_eval} problems)...")
        acc = self._run_eval(model, step, verbose=False)
        if acc > self.best_acc:
            self.best_acc  = acc
            self.best_step = step
            print(f"  ⭐ New best MATH500: {acc:.1f}% at step {step}")
        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):
        print(f"\n{'═'*60}")
        if self.sanity:
            # Sanity: 3 problems, full solution printed for each
            print("  FINAL MATH500 eval — SANITY (3 problems, solutions printed)")
            self._run_eval(model, state.global_step, n_override=3, verbose=True)
        else:
            print("  FINAL MATH500 eval (200 problems)...")
            self._run_eval(model, state.global_step, n_override=200, verbose=False)
        return control

    def _run_eval(self, model, step: int, n_override: int = None,
                  verbose: bool = False) -> float:
        if model is None:
            return -1.0
        n = n_override or self.n_eval

        ds = _load_math_dataset(split="test", cache=str(DATA_CACHE), silent=True)
        if ds is None:
            print("  ⚠️  Could not load MATH test split — skipping eval")
            return -1.0

        n_test = min(500, len(ds))
        ds     = ds.shuffle(seed=0).select(range(n_test))
        ds     = ds.select(range(min(n, n_test)))

        model.eval()
        device  = next(model.parameters()).device
        correct = 0

        for i, item in enumerate(ds):
            prob = item.get("problem", item.get("question", ""))
            sol  = item.get("solution", item.get("answer",  ""))
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

            resp = _forced_think_generate(
                model, self.tokenizer, enc, self.max_new_tokens, device
            )
            predicted  = extract_predicted_answer(resp)
            is_correct = verify_answer(predicted, gt)
            if is_correct:
                correct += 1

            if verbose:
                status  = "✅" if is_correct else "❌"
                print(f"\n  {'─'*56}")
                print(f"  Q{i+1} {status}")
                print(f"  Problem  : {prob[:120].strip()}{'...' if len(prob)>120 else ''}")
                print(f"  GT answer: {gt}")
                print(f"  Predicted: {predicted}")
                think_m = re.search(r"<think>(.*?)</think>", resp, re.DOTALL)
                sol_m   = re.search(r"<solution>(.*?)</solution>", resp, re.DOTALL)
                after   = resp.split("</solution>")[-1].strip() if "</solution>" in resp else ""
                if think_m:
                    think_text = think_m.group(1).strip()
                    if len(think_text) > 600:
                        think_text = think_text[:300] + "\n  ... [truncated] ...\n" + think_text[-200:]
                    print(f"  <think>  :\n    {think_text.replace(chr(10), chr(10)+'    ')}")
                else:
                    print(f"  <think>  : (no think tag)")
                if sol_m:
                    print(f"  <solution>: {sol_m.group(1).strip()}")
                else:
                    print(f"  <solution>: (no solution tag — fallback extraction used)")
                if after:
                    print(f"  After </solution>: {after[:100]}{'...' if len(after)>100 else ''}")
                reverif = sum(resp.lower().count(p) for p in _REVERIF_PHRASES)
                waits   = sum(resp.lower().count(p) for p in _WAIT_PHRASES)
                words   = len(resp.split())
                clipped = words >= (self.max_new_tokens * 0.95)
                print(f"  Words: {words}  |  re-verif: {reverif}  |  wait-loops: {waits}"
                      f"  |  clipped: {'YES ⚠️' if clipped else 'no'}")

            elif (i + 1) % 50 == 0:
                print(f"    [{i+1:>3}/{n}]  {correct}/{i+1} = {correct/(i+1)*100:.1f}%")

        acc = correct / n * 100
        if verbose:
            print(f"\n  {'─'*56}")
        print(f"  MATH500 @ step {step}: {correct}/{n} = {acc:.1f}%  "
              f"(target: 65%+, baseline: ~55%)")

        entry = {
            "step": step, "math500": acc,
            "n_samples": n, "timestamp": datetime.now().isoformat(),
        }
        self.history.append(entry)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump(self.history, f, indent=2)

        model.train()
        return acc


class GSM8KEvalCallback(TrainerCallback):
    """
    GSM8K eval every eval_every steps.
    Secondary metric in Phase A (baseline ~85%, target ~88%).
    Runs at final step only by default to avoid compute overhead.
    """

    def __init__(self, tokenizer, log_path: Path,
                 eval_every: int = 250, n_eval: int = 50,
                 max_new_tokens: int = 1500, sanity: bool = False):
        self.tokenizer      = tokenizer
        self.log_path       = log_path
        self.eval_every     = eval_every
        self.n_eval         = n_eval
        self.max_new_tokens = max_new_tokens
        self.sanity         = sanity
        self.history        = []

    def on_step_end(self, args, state: TrainerState,
                    control: TrainerControl, model=None, **kwargs):
        step = state.global_step
        if step == 0 or step % self.eval_every != 0:
            return control
        print(f"\n{'─'*50}")
        print(f"  GSM8K eval @ step {step}...")
        self._run_eval(model, step, verbose=False)
        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):
        print(f"\n{'═'*50}")
        if self.sanity:
            print("  FINAL GSM8K eval — SANITY (3 problems, solutions printed)")
            self._run_eval(model, state.global_step, n_override=3, verbose=True)
        else:
            print("  FINAL GSM8K eval (200 problems)...")
            self._run_eval(model, state.global_step, n_override=200, verbose=False)
        return control

    def _run_eval(self, model, step: int, n_override: int = None,
                  verbose: bool = False) -> float:
        if model is None:
            return -1.0
        n = n_override or self.n_eval

        try:
            ds = load_dataset("gsm8k", "main", split="test",
                              cache_dir=str(DATA_CACHE))
            ds = ds.select(range(min(n, len(ds))))
        except Exception as e:
            print(f"  ⚠️  GSM8K load failed: {e}")
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

            resp       = _forced_think_generate(
                model, self.tokenizer, enc, self.max_new_tokens, device
            )
            predicted  = extract_predicted_answer(resp)
            gt         = extract_ground_truth(item["answer"], "gsm8k")
            is_correct = verify_answer(predicted, gt)
            if is_correct:
                correct += 1

            if verbose:
                status = "✅" if is_correct else "❌"
                print(f"\n  {'─'*56}")
                print(f"  Q{i+1} {status}")
                print(f"  Problem  : {item['question'][:120].strip()}")
                print(f"  GT answer: {gt}")
                print(f"  Predicted: {predicted}")
                think_m = re.search(r"<think>(.*?)</think>", resp, re.DOTALL)
                sol_m   = re.search(r"<solution>(.*?)</solution>", resp, re.DOTALL)
                after   = resp.split("</solution>")[-1].strip() if "</solution>" in resp else ""
                if think_m:
                    think_text = think_m.group(1).strip()
                    if len(think_text) > 600:
                        think_text = think_text[:300] + "\n  ... [truncated] ...\n" + think_text[-200:]
                    print(f"  <think>  :\n    {think_text.replace(chr(10), chr(10)+'    ')}")
                else:
                    print(f"  <think>  : (no think tag)")
                if sol_m:
                    print(f"  <solution>: {sol_m.group(1).strip()}")
                else:
                    print(f"  <solution>: (no solution tag)")
                if after:
                    print(f"  After </solution>: {after[:100]}{'...' if len(after)>100 else ''}")
                reverif = sum(resp.lower().count(p) for p in _REVERIF_PHRASES)
                waits   = sum(resp.lower().count(p) for p in _WAIT_PHRASES)
                words   = len(resp.split())
                clipped = words >= (self.max_new_tokens * 0.95)
                print(f"  Words: {words}  |  re-verif: {reverif}  |  wait-loops: {waits}"
                      f"  |  clipped: {'YES ⚠️' if clipped else 'no'}")

            elif (i + 1) % 50 == 0:
                print(f"    [{i+1:>3}/{n}]  {correct}/{i+1} = {correct/(i+1)*100:.1f}%")

        acc = correct / n * 100
        if verbose:
            print(f"\n  {'─'*56}")
        print(f"  GSM8K @ step {step}: {correct}/{n} = {acc:.1f}%  "
              f"(target: 88%+, baseline: ~85%)")

        entry = {
            "step": step, "gsm8k": acc,
            "n_samples": n, "timestamp": datetime.now().isoformat(),
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
    print(f"  Loading : {model_path}")
    print(f"  Attn    : {attn_impl}")
    print(f"{'─'*60}")

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Expected: stage3_distilled_merged checkpoint"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",   # LEFT padding is critical for generation in GRPO
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map={"": 0},    # Single GPU (GPU 1 via CUDA_VISIBLE_DEVICES=1)
        trust_remote_code=True,
    )

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Loaded  : {n_params:.2f}B parameters")
    return model, tokenizer


def apply_lora(model, rank: int, alpha: int):
    """
    Apply LoRA adapters to all attention + MLP projections.
    GRPOTrainer uses LoRA model as policy, base model as reference.
    trl handles the reference model split automatically.
    """
    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",   # attention
            "gate_proj", "up_proj", "down_proj",        # MLP (SwiGLU)
        ],
        inference_mode=False,
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  LoRA trainable: {trainable/1e6:.1f}M / {total/1e9:.2f}B "
          f"({trainable/total*100:.1f}%)")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# GRPO CONFIG
# ══════════════════════════════════════════════════════════════════════════════

def build_grpo_config(args, output_dir: Path) -> GRPOConfig:
    """
    Build GRPOConfig with trl version compatibility guard.

    Key differences vs v1:
      - max_comp_len: 4096 → 1500  (hard cap prevents runaway loops)
      - kl_coef:      0.04 → 0.05  (stronger protection of distillation gains)
      - warmup_steps: 20   → 50    (more stable ramp-up on competition problems)
      - entropy_coef: 0.01 if available (prevents mode collapse)
    """
    if args.wandb:
        os.environ["WANDB_PROJECT"] = "mathReason-1.5B"

    max_steps  = 20 if args.sanity else args.max_steps
    save_steps = 5  if args.sanity else 100
    log_steps  = 2  if args.sanity else 10
    eval_every = 5  if args.sanity else args.eval_every
    n_eval     = 10 if args.sanity else args.n_eval

    print(f"\n  GRPO Config (v2):")
    print(f"    Input model       : {args.model}")
    print(f"    Max steps         : {max_steps}")
    print(f"    num_generations   : {args.num_gen}")
    print(f"    max_comp_len      : {args.max_comp_len} tokens (hard loop cap)")
    print(f"    kl_coef           : {args.kl_coef}")
    print(f"    temperature       : {args.temperature}")
    print(f"    lr                : {args.lr}")
    print(f"    Reward            : BINARY (correct=1.0, wrong=0.0)")
    print(f"    clip_higher       : ε_low={args.epsilon_low}  ε_high={args.epsilon_high} (DAPO)")

    # ── Probe GRPOConfig signature at runtime ───────────────────────────────
    # Different trl versions rename or drop params without warning.
    # We inspect the actual __init__ signature and silently drop anything
    # that isn't there, then report what was accepted vs skipped.
    import inspect
    _grpo_sig_params = set(inspect.signature(GRPOConfig.__init__).parameters.keys())

    def _supported(key: str) -> bool:
        return key in _grpo_sig_params

    # ── Build kwargs, skipping unsupported params ────────────────────────────
    # IMPORTANT: every param that has been renamed across trl versions is listed
    # here with its alternative name as a comment so it's easy to debug.

    base_kwargs: Dict = {}

    # Always supported
    base_kwargs["output_dir"]  = str(output_dir)
    base_kwargs["run_name"]    = args.run_name
    base_kwargs["max_steps"]   = max_steps
    base_kwargs["num_train_epochs"] = 1
    base_kwargs["learning_rate"]    = args.lr
    # generation_batch_size = per_device_train_batch_size × gradient_accumulation_steps
    # This MUST be divisible by num_generations (8).
    # With batch_size=1, grad_accum must equal num_generations exactly.
    base_kwargs["per_device_train_batch_size"] = 1
    base_kwargs["gradient_accumulation_steps"] = args.num_gen   # 1 × 8 = 8 ✅
    base_kwargs["weight_decay"]     = 0.01
    base_kwargs["max_grad_norm"]    = 1.0
    # Sanity uses constant LR — cosine over 20 steps decays to ~5e-9 by step 20
    # (100× below peak), causing the model to effectively stop learning in the
    # last 5 steps. Full run uses cosine over 500 steps (still ~4.9e-7 at step 20).
    base_kwargs["lr_scheduler_type"] = "cosine" if not args.sanity else "constant"
    base_kwargs["warmup_steps"]      = 50 if not args.sanity else 2
    base_kwargs["bf16"]             = True
    base_kwargs["save_strategy"]    = "steps"
    base_kwargs["save_steps"]       = save_steps
    base_kwargs["save_total_limit"] = 3
    base_kwargs["logging_steps"]    = log_steps
    base_kwargs["report_to"]        = "wandb" if args.wandb else "none"
    base_kwargs["remove_unused_columns"] = False
    base_kwargs["seed"]             = 42

    # num_generations — renamed to n_generations in some versions
    for key in ["num_generations", "n_generations"]:
        if _supported(key):
            base_kwargs[key] = args.num_gen
            break

    # max_completion_length — renamed to max_new_tokens or max_length in some
    for key in ["max_completion_length", "max_new_tokens", "max_length"]:
        if _supported(key):
            base_kwargs[key] = args.max_comp_len
            break

    # max_prompt_length — optional in many versions, skip if missing
    for key in ["max_prompt_length", "max_prompt_len"]:
        if _supported(key):
            base_kwargs[key] = args.max_prompt_len
            break
    # If neither variant exists, we skip it entirely — the model truncates via
    # the tokenizer max_length in the dataset collator

    # kl_coef — renamed to beta or kl_penalty_coef in some forks
    for key in ["kl_coef", "beta", "kl_penalty_coef"]:
        if _supported(key):
            base_kwargs[key] = args.kl_coef
            break

    # temperature — may live in generation_config in newer trl
    if _supported("temperature"):
        base_kwargs["temperature"] = args.temperature

    # use_vllm — not present in older trl
    if _supported("use_vllm"):
        base_kwargs["use_vllm"] = False

    # optim — may not be present in all trl versions (falls back to TrainingArguments)
    if _supported("optim"):
        base_kwargs["optim"] = "adamw_torch_fused"

    # Report what will be passed
    skipped = [
        "max_prompt_length", "entropy_coef", "use_vllm", "temperature"
    ]
    accepted_special = {k: base_kwargs[k] for k in base_kwargs
                        if k not in ["output_dir", "run_name", "seed"]}
    print(f"    GRPOConfig params   : {len(base_kwargs)} accepted")
    for k in ["num_generations", "n_generations", "max_completion_length",
              "max_new_tokens", "kl_coef", "beta", "temperature",
              "max_prompt_length", "use_vllm"]:
        if k in base_kwargs:
            print(f"      {k:<28}: {base_kwargs[k]}")
        elif k in ["max_prompt_length"]:
            print(f"      {k:<28}: (not supported — OK, tokenizer handles truncation)")

    # Try adding entropy_coef (trl >= 0.12 only)
    if _supported("entropy_coef"):
        base_kwargs["entropy_coef"] = 0.01
        print(f"      entropy_coef              : 0.01 ✅")
    else:
        print(f"      entropy_coef              : (not supported in this trl — OK)")

    # ── clip_higher: DAPO asymmetric clipping ───────────────────────────────
    # Standard GRPO uses symmetric clip: ratio ∈ [1-ε, 1+ε]  (ε=0.2)
    # This symmetrically suppresses both low AND high probability tokens.
    # Problem: high-entropy tokens like "wait", "however", "let me reconsider"
    # are legitimate reasoning sparks. Suppressing them causes entropy collapse
    # where the model converges to a narrow reasoning style and stops exploring.
    #
    # DAPO fix (clip_higher): ratio ∈ [1-ε_low, 1+ε_high]
    #   ε_low  = 0.20  (same as before — still prevent large drops)
    #   ε_high = 0.28  (slightly higher — allow reasoning tokens to grow)
    #
    # Effect: model can increase probability of high-entropy reasoning tokens
    # more freely, maintaining exploration throughout training.
    # Research: clip_higher is a core DAPO component that prevents the entropy
    # collapse observed in standard GRPO during long training runs.
    #
    # In trl this may be called: clip_higher, epsilon_high, or epsilon_low+high
    clip_set = False
    for high_key, low_key in [
        ("clip_higher", None),
        ("epsilon_high", "epsilon_low"),
        ("ratio_clip_max", "ratio_clip_min"),
    ]:
        if _supported(high_key):
            base_kwargs[high_key] = args.epsilon_high
            if low_key and _supported(low_key):
                base_kwargs[low_key] = args.epsilon_low
            print(f"      clip_higher ({high_key})     : {args.epsilon_high} ✅  (DAPO asymmetric)")
            clip_set = True
            break

    if not clip_set:
        if _supported("epsilon"):
            base_kwargs["epsilon"] = args.epsilon_low
            print(f"      epsilon                   : {args.epsilon_low} (symmetric, clip_higher N/A)")
        else:
            print(f"      clip_higher               : (not in this trl — standard clipping)")
        print(f"      NOTE: upgrade trl for full DAPO clip_higher support")

    cfg = GRPOConfig(**base_kwargs)
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# STaR SELF-FILTER (after training — generates Phase B dataset)
# ══════════════════════════════════════════════════════════════════════════════

def run_star_filter(model, tokenizer, n_problems: int = 2000,
                    n_attempts: int = 4, output_dir: Path = STAR_DIR):
    """
    STaR self-filter: run the Phase A checkpoint on harder problems
    (MATH L4-5), keep verified-correct solutions.

    Purpose: generate curriculum data for Phase B GRPO.
    The model curates its own training data — problems it can sometimes
    solve but not always are exactly in the 20–60% pass-rate window for Phase B.

    Output files:
      star_round1.jsonl        — verified correct (problem, solution, GT)
      star_round1_hard.jsonl   — failed all attempts (high-priority for Phase B)
    """
    print(f"\n{'═'*60}")
    print(f"  STaR Self-Filter  ({n_problems} problems, {n_attempts} attempts each)")
    print(f"  Using MATH Level 4–5 (harder than Phase A training data)")
    print(f"{'═'*60}")

    ds = _load_math_dataset(split="train", cache=str(DATA_CACHE), silent=True)
    if ds is None:
        print("  ⚠️  Could not load MATH for STaR. Skipping.")
        return
    try:
        # Level 4-5 only — harder than what we trained on (L3-4)
        ds = ds.filter(lambda x: x.get("level", "") in ["Level 4", "Level 5"])
        ds = ds.shuffle(seed=999).select(range(min(n_problems, len(ds))))
    except Exception as e:
        print(f"  ⚠️  MATH filter failed: {e} — using full dataset")
        ds = ds.shuffle(seed=999).select(range(min(n_problems, len(ds))))

    model.eval()
    device = next(model.parameters()).device
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path  = output_dir / "star_round1.jsonl"
    hard_path = output_dir / "star_round1_hard.jsonl"

    verified = 0
    hard     = 0
    total    = 0

    with open(out_path, "w") as fv, open(hard_path, "w") as fh:
        for i, item in enumerate(ds):
            prob = item.get("problem", item.get("question", ""))
            sol  = item.get("solution", item.get("answer", ""))
            gt   = extract_ground_truth(sol, "math_l3l4")

            if not prob or not gt:
                continue
            total += 1

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prob},
            ]
            prompt_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            enc = tokenizer(
                prompt_str, return_tensors="pt",
                truncation=True, max_length=512
            ).to(device)

            solved        = False
            best_solution = None

            for attempt in range(n_attempts):
                with torch.no_grad():
                    out = model.generate(
                        **enc,
                        max_new_tokens=1500,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                resp      = tokenizer.decode(
                    out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
                )
                predicted = extract_predicted_answer(resp)

                if verify_answer(predicted, gt):
                    solved        = True
                    best_solution = resp
                    break

            if solved and best_solution:
                verified += 1
                fv.write(json.dumps({
                    "problem":      prob,
                    "solution":     best_solution,
                    "ground_truth": gt,
                    "level":        item.get("level", "Level 4"),
                    "source":       "star_round1",
                }) + "\n")
            else:
                hard += 1
                fh.write(json.dumps({
                    "problem":      prob,
                    "ground_truth": gt,
                    "level":        item.get("level", "Level 4"),
                }) + "\n")

            if (i + 1) % 100 == 0:
                solve_rate = verified / max(total, 1) * 100
                print(f"  [{i+1:>4}/{n_problems}]  "
                      f"Verified: {verified}  Hard: {hard}  "
                      f"Solve rate: {solve_rate:.1f}%")

    solve_rate = verified / max(total, 1) * 100
    print(f"\n  ✅ STaR complete:")
    print(f"     Verified (use as Phase B data) : {verified:,}  → {out_path}")
    print(f"     Hard (failed all {n_attempts} attempts): {hard:,}  → {hard_path}")
    print(f"     Solve rate                      : {solve_rate:.1f}%")
    print(f"     → Add {out_path} to Phase B dataset")

    model.train()


# ══════════════════════════════════════════════════════════════════════════════
# MERGE
# ══════════════════════════════════════════════════════════════════════════════

def merge_lora(model, tokenizer, merged_dir: Path):
    """Merge LoRA adapters into base weights. Output is Phase B input."""
    print(f"\n{'─'*60}")
    print("  Merging LoRA → base weights...")
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    size_gb = sum(
        p.nbytes for p in merged.parameters()
    ) / 1e9
    print(f"  ✅ Merged checkpoint: {merged_dir}")
    print(f"     Model size        : {size_gb:.1f} GB")
    print(f"     → Use as input for GRPO Phase B (8K context)")
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    sanity = args.sanity

    # ── Resolve completion length from phase ──────────────────────────────────
    PHASE_COMP_LEN = {"A": 2048, "B": 4096, "C": 8192}
    comp_len = args.max_comp_len if args.max_comp_len > 0 else PHASE_COMP_LEN[args.phase]
    # Patch back so everything downstream sees the resolved value
    args.max_comp_len = comp_len

    suffix     = f"_phase{args.phase}" + ("_sanity" if sanity else "")
    output_dir = Path(args.output_dir) / f"{args.run_name}{suffix}"
    merged_dir = Path(args.merged_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STAR_DIR.mkdir(parents=True, exist_ok=True)
    DATA_CACHE.mkdir(parents=True, exist_ok=True)

    # ── Banner ──
    print("\n" + "█" * 60)
    print(f"  STAGE 4A — GRPO  (Phase {args.phase}  |  ctx={comp_len} tokens)")
    print(f"  Binary reward: correct=1.0, wrong=0.0")
    if sanity:
        print("  *** SANITY RUN — 80 problems, 20 steps ***")
    print("█" * 60)
    print(f"  Timestamp    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Input        : {args.model}")
    print(f"  Output       : {output_dir}")
    print(f"  Phase        : {args.phase}  →  max_comp_len={comp_len}")
    print(f"  kl_coef      : {args.kl_coef}")
    print(f"  lr           : {args.lr}")

    # ── Load model ──
    model, tokenizer = load_model_and_tokenizer(args.model, args.attn_impl)

    # ── Apply LoRA ──
    model = apply_lora(model, args.lora_rank, args.lora_alpha)

    # ── Build reward object — BINARY ONLY ────────────────────────────────────
    global _reward
    _reward = BinaryReward()
    _reward._tokenizer = tokenizer
    reward_fn = make_reward_fn(_reward)

    print(f"\n  Reward: BINARY (correct=1.0  wrong=0.0)")
    print(f"  No termination/loop/length/format components.")
    print(f"  Loops resolve naturally as correctness improves.")

    # ── Build dataset ──
    dataset = build_dataset(args)

    if sanity:
        # 80 problems for sanity — enough to see real pass rate distribution
        dataset = dataset.select(range(min(80, len(dataset))))
        print(f"\n  [Sanity] {len(dataset)} problems × {args.num_gen} completions/step = 20 steps")

    print(f"\n  Training on {len(dataset):,} problems")
    print(f"  Each step: 1 problem × {args.num_gen} completions → binary reward → GRPO update")

    # ── GRPO Config ──
    grpo_cfg = build_grpo_config(args, output_dir)

    # ── Callbacks ──
    monitor_cb = RewardMonitorCallback(
        reward_obj  = _reward,
        log_path    = LOG_DIR / "stage4a_v2_monitor.json",
        check_every = 10 if sanity else 25,
    )
    math500_cb = Math500EvalCallback(
        tokenizer      = tokenizer,
        log_path       = LOG_DIR / "stage4a_v2_math500_evals.json",
        eval_every     = 999999 if sanity else 550,               # sanity: end-only
        n_eval         = 10 if sanity else args.n_eval,
        max_new_tokens = args.max_comp_len,
        sanity         = sanity,
    )
    gsm8k_cb = GSM8KEvalCallback(
        tokenizer      = tokenizer,
        log_path       = LOG_DIR / "stage4a_v2_gsm8k_evals.json",
        eval_every     = 999999,                                  # disabled entirely
        n_eval         = 10 if sanity else 150,
        max_new_tokens = args.max_comp_len,
        sanity         = sanity,
    )
    callbacks = [monitor_cb, math500_cb, gsm8k_cb]

    if args.skip_eval:
        callbacks = [monitor_cb]

    # ── GRPOTrainer ──
    trainer = GRPOTrainer(
        model            = model,
        args             = grpo_cfg,
        reward_funcs     = [reward_fn],
        train_dataset    = dataset,
        processing_class = tokenizer,
        callbacks        = callbacks,
    )

    # ── Pre-training reward sanity check ──────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  Binary reward verification (correct=1.0, wrong=0.0):")
    checks = [
        ("<think>\n2+2=4\n</think>\n<solution>\n4\n</solution>", "4",  "correct + tags"),
        ("<think>\n2+2=5\n</think>\n<solution>\n5\n</solution>", "4",  "wrong + tags"),
        ("The answer is 4.",                                      "4",  "correct no tags"),
        ("<think>\n2+2=4. Let me check again. Yes 4. Let me check again. 4.</think>\n<solution>4</solution>", "4", "correct + loops"),
    ]
    for comp, gt, label in checks:
        r = _reward.compute(comp, gt)
        print(f"  {label:<35}: {r:.1f}  {'✅' if (r==1.0)==('correct' in label) else '❌'}")
    _reward._stats.clear(); _reward.n_calls = 0
    print()

    # ── Train ──
    print(f"{'─'*60}")
    print("  Starting GRPO training...")
    print(f"{'─'*60}\n")

    result = trainer.train(
        resume_from_checkpoint=str(output_dir / "checkpoint-200") if args.resume else None
    )

    # ── Save ──
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = result.metrics
    with open(LOG_DIR / "stage4a_v2_train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  ✅ LoRA adapter saved  : {output_dir}")
    print(f"  ✅ Train metrics saved : {LOG_DIR / 'stage4a_v2_train_metrics.json'}")

    # ── Final reward summary ──
    _reward.print_summary(
        step   = grpo_cfg.max_steps,
        last_n = 500,
    )

    # ── Merge ──
    merged_model = None
    if not args.skip_merge and not sanity:
        merged_model = merge_lora(model, tokenizer, merged_dir)
    elif sanity:
        print("\n  [Sanity] Skipping merge — run full training first")

    # ── STaR self-filter ──
    if not args.skip_star and not sanity:
        eval_model = merged_model if merged_model is not None else model
        print(f"\n  Running STaR self-filter on MATH L4-5...")
        run_star_filter(
            model      = eval_model,
            tokenizer  = tokenizer,
            n_problems = 2000,
            n_attempts = 4,
            output_dir = STAR_DIR,
        )
    elif sanity:
        print("\n  [Sanity] Skipping STaR filter")

    # ── Final summary ──
    print(f"\n{'█'*60}")
    if sanity:
        print("  SANITY COMPLETE")
        print("  ─────────────────────────────────────────────────────")
        print("  What to check:")
        print("    ✅ pass_rate     > 20%   (MetaMath/MATH L2-3 at 40-55%)")
        print("    ✅ batch_skip    < 20%   (was 51% with old dataset)")
        print("    ✅ reward_std    > 0.1   (binary gives 0 or 1, std ≈ 0.5 for 50% pass)")
        print("    ✅ grad_norm     > 0.01  (model is updating)")
        print("    ✅ MATH500 eval  ≥ 55%   (no regression from binary reward)")
        print()
        print(f"  If all green → launch Phase A full run:")
        print(f"  CUDA_VISIBLE_DEVICES=1 python stage4a_grpo_v2.py --phase A --run-name grpo_binary_r1 --max-steps 500")
        print()
        print("  Monitor during full run:")
        print("    MATH500 trajectory (every 250 steps) — primary metric")
        print("    clean_term_rate should increase from ~10% to ~60%+")
        print("    mean_reverif should decrease from ~2.0 to ~0.5")
        print("    mean resp words should decrease from ~900 to ~500")
    else:
        print("  STAGE 4A GRPO PHASE A v2 COMPLETE")
        print("  ─────────────────────────────────────────────────────")
        print(f"  LoRA adapter  : {output_dir}")
        if not args.skip_merge:
            print(f"  Merged model  : {merged_dir}")
            print(f"  → Use merged as input for Gap-filling SFT (Stage 5)")
        if not args.skip_star:
            print(f"  STaR data     : {STAR_DIR}/star_round1.jsonl")
            print(f"  → Add to Phase B GRPO dataset")

        # Print eval trajectories
        for eval_log_path, bench_key, bench_name in [
            (LOG_DIR / "stage4a_v2_math500_evals.json", "math500", "MATH500"),
            (LOG_DIR / "stage4a_v2_gsm8k_evals.json",  "gsm8k",   "GSM8K"),
        ]:
            if eval_log_path.exists():
                with open(eval_log_path) as f:
                    evals = json.load(f)
                if evals:
                    print(f"\n  {bench_name} trajectory:")
                    for e in evals:
                        print(f"    Step {e['step']:>3}: {e[bench_key]:.1f}%")
                    best_e = max(evals, key=lambda x: x[bench_key])
                    print(f"  Best {bench_name}: {best_e[bench_key]:.1f}% @ step {best_e['step']}")

        print()
        print("  Targets:")
        print("    MATH500: ~65%  (baseline ~55%)")
        print("    GSM8K:   ~88%  (baseline ~85%)")
        print("    clean_term_rate: ~60%+ (was ~10%)")

    print(f"{'█'*60}\n")


if __name__ == "__main__":
    main()
