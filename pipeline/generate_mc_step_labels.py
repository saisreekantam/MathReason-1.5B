"""
generate_mc_step_labels.py  —  Stage 8A: MC Step Label Generation
══════════════════════════════════════════════════════════════════════════════
Server  : revanth@172.16.192.168
GPU     : 0  (leaves GPU 1 free for anything else)
Input   : ~/nlp/checkpoints/stage7_dpo_merged   (generator)
          fallback → stage4d_gdpo_merged
Dataset : MATH train + GSM8K hard  (~3,000 problems)
Output  : ~/nlp/data/prm/mc_step_labels.jsonl

Algorithm (Math-Shepherd style):
  For each problem:
    1. Generate one full reference solution from stage7 model.
    2. Extract the <think> block and split into steps at \n\n.
    3. For each step prefix [s0..st]:
         - Build prompt = original question + solution prefix up to step t
         - Sample K=8 rollout completions
         - MC score(t) = fraction that reach the correct final answer
    4. Label:
         - Steps where MC score >= threshold → label 1 (good)
         - First step where score drops below threshold → label 0 (first error)
         - Stop labeling after first bad step (no point labeling later steps)
    5. Emit one record per step with: problem, prefix, mc_score, label

Usage:
  tmux new -s gen_prm
  conda activate nlp
  CUDA_VISIBLE_DEVICES=0 python generate_mc_step_labels.py

  # Resume (safe to re-run):
  CUDA_VISIBLE_DEVICES=0 python generate_mc_step_labels.py --resume

  # Quick sanity (10 problems, 2 rollouts):
  CUDA_VISIBLE_DEVICES=0 python generate_mc_step_labels.py --sanity
══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from datasets import concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── Paths ────────────────────────────────────────────────────────────────────

WORK_DIR      = Path("~/nlp").expanduser()
GEN_MODEL     = WORK_DIR / "checkpoints" / "stage7_dpo_merged"
GEN_FALLBACK  = WORK_DIR / "checkpoints" / "stage4d_gdpo_merged"
OUT_DIR       = WORK_DIR / "data" / "prm"
OUT_FILE      = OUT_DIR / "mc_step_labels.jsonl"
DONE_FILE     = OUT_DIR / "mc_done_keys.txt"

SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Think step by step inside <think> tags, "
    "then give your final answer inside <solution> tags."
)

# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MC Step Label Generator for PRM")
    p.add_argument("--sanity",     action="store_true",
                   help="10 problems, 2 rollouts/step — quick test")
    p.add_argument("--resume",     action="store_true",
                   help="Skip problems already in DONE_FILE")
    p.add_argument("--n-probs",    type=int,   default=3000,
                   help="Total problems to process")
    p.add_argument("--k-rollouts", type=int,   default=8,
                   help="Rollout completions per step prefix")
    p.add_argument("--threshold",  type=float, default=0.3,
                   help="MC score below which a step is considered bad")
    p.add_argument("--max-new",    type=int,   default=1024,
                   help="Max new tokens per rollout completion")
    p.add_argument("--max-ref",    type=int,   default=2048,
                   help="Max new tokens for reference solution generation")
    p.add_argument("--model",      type=str,   default="auto")
    return p.parse_args()

# ─── Answer extraction ─────────────────────────────────────────────────────────

_SOLUTION_RE = re.compile(r"<solution>(.*?)</solution>", re.DOTALL | re.IGNORECASE)
_NUMBER_RE   = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?")

def extract_answer(text: str) -> Optional[str]:
    """Smart extractor: check first 60% of text for <solution> tag."""
    cutoff = max(len(text) * 60 // 100, 200)
    search_zone = text[:cutoff]
    m = _SOLUTION_RE.search(search_zone)
    if m:
        return m.group(1).strip()
    # fallback: last number in full text
    nums = _NUMBER_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None

def normalize(s: str) -> str:
    s = s.strip().lower().replace(",", "").replace(" ", "")
    # strip trailing .0
    if s.endswith(".0"):
        s = s[:-2]
    return s

def verify(predicted: Optional[str], ground_truth: str) -> bool:
    if predicted is None:
        return False
    try:
        return abs(float(normalize(predicted)) - float(normalize(ground_truth))) < 1e-4
    except ValueError:
        return normalize(predicted) == normalize(ground_truth)

# ─── Step splitting ────────────────────────────────────────────────────────────

def extract_think_block(solution: str) -> str:
    """Pull the text inside <think>...</think>."""
    m = re.search(r"<think>(.*?)(?:</think>|$)", solution, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else solution.strip()

def split_into_steps(think_text: str) -> List[str]:
    """
    Split think block into logical steps.
    Primary split: double-newline (\n\n).
    Each step must be non-trivial (>= 20 chars).
    Returns list of step strings (not cumulative — individual chunks).
    """
    raw = [s.strip() for s in re.split(r"\n\n+", think_text)]
    steps = [s for s in raw if len(s) >= 20]
    return steps if steps else [think_text.strip()]

def build_prefixes(steps: List[str]) -> List[str]:
    """Convert step list into cumulative prefixes for rollout prompting."""
    prefixes = []
    accumulated = ""
    for step in steps:
        accumulated = (accumulated + "\n\n" + step).strip()
        prefixes.append(accumulated)
    return prefixes

# ─── Model loading ─────────────────────────────────────────────────────────────

def resolve_model(model_arg: str) -> Path:
    if model_arg != "auto":
        p = Path(model_arg)
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")
        return p
    if GEN_MODEL.exists():
        print(f"  Generator: {GEN_MODEL}  (stage7_dpo_merged ✅)")
        return GEN_MODEL
    if GEN_FALLBACK.exists():
        print(f"  Generator: {GEN_FALLBACK}  (fallback — stage7 not found)")
        return GEN_FALLBACK
    raise FileNotFoundError(
        f"No generator model found.\n"
        f"  Tried: {GEN_MODEL}\n"
        f"         {GEN_FALLBACK}"
    )

def load_model_and_tokenizer(model_path: Path):
    print(f"  Loading model from {model_path} ...")
    tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",   # flash_attn unavailable on this server
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Model loaded — {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")
    return model, tok

# ─── Generation helpers ────────────────────────────────────────────────────────

def build_prompt(tokenizer, problem: str, partial_think: Optional[str] = None) -> str:
    """
    Build chat-formatted prompt.
    If partial_think is given, the assistant turn is pre-filled with
    <think>\n{partial_think}  so the model continues from that prefix.
    """
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": problem},
    ]
    if partial_think is not None:
        messages.append({
            "role":    "assistant",
            "content": f"<think>\n{partial_think}",
        })
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=(partial_think is None),
            continue_final_message=(partial_think is not None),
        )
    except Exception:
        # Fallback for tokenizers without chat template support
        if partial_think is None:
            prompt = (
                f"<|system|>\n{SYSTEM_PROMPT}\n"
                f"<|user|>\n{problem}\n"
                f"<|assistant|>\n"
            )
        else:
            prompt = (
                f"<|system|>\n{SYSTEM_PROMPT}\n"
                f"<|user|>\n{problem}\n"
                f"<|assistant|>\n<think>\n{partial_think}"
            )
    return prompt

@torch.inference_mode()
def generate_completions(
    model, tokenizer, prompt: str,
    n: int, max_new_tokens: int,
    temperature: float = 0.8,
) -> List[str]:
    """
    Generate n completions for a given prompt.
    If temperature <= 0 or n == 1 with temp=0, uses greedy (do_sample=False).
    """
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = enc["input_ids"].shape[1]

    greedy = (temperature <= 0.0)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=not greedy,
        num_return_sequences=n,
        pad_token_id=tokenizer.eos_token_id,
    )
    if not greedy:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"]       = 0.95

    outputs = model.generate(**enc, **gen_kwargs)
    completions = []
    for out in outputs:
        text = tokenizer.decode(out[input_len:], skip_special_tokens=True)
        completions.append(text)
    return completions

# ─── Dataset loading ──────────────────────────────────────────────────────────

def load_problems(n_total: int, cache_dir: Path) -> List[dict]:
    """Load MATH train + GSM8K hard, return list of {problem, answer} dicts."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    problems = []

    # MATH train
    try:
        ds_math = load_dataset(
            "lighteval/MATH", split="train",
            cache_dir=str(cache_dir), trust_remote_code=True
        )
        for item in ds_math:
            problems.append({
                "key":     f"math_{item.get('unique_id', len(problems))}",
                "problem": item["problem"],
                "answer":  item["solution"],   # raw solution (extract from it)
                "source":  "math",
            })
    except Exception as e:
        print(f"  [warn] Could not load MATH: {e}")

    # GSM8K train (hard problems only — filter for multi-step)
    try:
        ds_gsm = load_dataset(
            "openai/gsm8k", "main", split="train",
            cache_dir=str(cache_dir)
        )
        for item in ds_gsm:
            # extract final answer from "#### <number>" format
            ans_match = re.search(r"####\s*([\d,]+)", item["answer"])
            if ans_match:
                problems.append({
                    "key":     f"gsm_{len(problems)}",
                    "problem": item["question"],
                    "answer":  ans_match.group(1).replace(",", ""),
                    "source":  "gsm8k",
                })
    except Exception as e:
        print(f"  [warn] Could not load GSM8K: {e}")

    if not problems:
        raise RuntimeError("No problems loaded — check dataset availability.")

    # shuffle and cap
    import random
    random.seed(42)
    random.shuffle(problems)
    return problems[:n_total]

# ─── Main logic ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.sanity:
        args.n_probs    = 10
        args.k_rollouts = 2
        print("*** SANITY MODE: 10 problems, 2 rollouts/step ***\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load already-done keys for resume
    done_keys = set()
    if args.resume and DONE_FILE.exists():
        done_keys = set(DONE_FILE.read_text().splitlines())
        print(f"  Resume: {len(done_keys)} problems already done, skipping.")

    # Load model
    model_path = resolve_model(args.model)
    model, tokenizer = load_model_and_tokenizer(model_path)

    # Load problems
    print(f"\n  Loading problems (target: {args.n_probs}) ...")
    cache_dir = WORK_DIR / "data" / "cache"
    problems  = load_problems(args.n_probs, cache_dir)
    problems  = [p for p in problems if p["key"] not in done_keys]
    print(f"  Problems to process: {len(problems)}")

    # Open output files
    mode = "a" if args.resume else "w"
    fout  = open(OUT_FILE,  mode, buffering=1)
    fdone = open(DONE_FILE, mode, buffering=1)

    total_step_records = 0
    total_good = 0
    total_bad  = 0

    for prob_idx, item in enumerate(problems):
        problem   = item["problem"]
        gt_answer = item["answer"]
        key       = item["key"]

        print(f"\n[{prob_idx+1:>4}/{len(problems)}]  key={key}")

        # ── Step 1: Generate reference solution ──────────────────────────────
        ref_prompt = build_prompt(tokenizer, problem)
        ref_completions = generate_completions(
            model, tokenizer, ref_prompt,
            n=1, max_new_tokens=args.max_ref, temperature=0.0,
        )
        ref_solution = ref_completions[0]

        # Verify reference is correct — only label problems where model finds
        # the right answer (otherwise no positive steps to learn from)
        ref_pred = extract_answer(ref_solution)
        if not verify(ref_pred, gt_answer):
            print(f"  ✗ Reference solution wrong (pred={ref_pred}, gt={gt_answer}) — skipping")
            fdone.write(key + "\n")
            continue

        # ── Step 2: Split into steps ─────────────────────────────────────────
        think_text = extract_think_block(ref_solution)
        steps      = split_into_steps(think_text)
        prefixes   = build_prefixes(steps)

        if len(steps) < 2:
            print(f"  ✗ Too few steps ({len(steps)}) — skipping")
            fdone.write(key + "\n")
            continue

        print(f"  ✓ Reference correct  |  {len(steps)} steps extracted")

        # ── Step 3: MC rollouts per step ─────────────────────────────────────
        prev_score = 1.0
        any_label_written = False

        for t, prefix in enumerate(prefixes):
            rollout_prompt = build_prompt(tokenizer, problem, partial_think=prefix)
            rollouts = generate_completions(
                model, tokenizer, rollout_prompt,
                n=args.k_rollouts, max_new_tokens=args.max_new,
                temperature=0.8,
            )

            correct_count = sum(
                1 for r in rollouts
                if verify(extract_answer(r), gt_answer)
            )
            mc_score = correct_count / args.k_rollouts
            is_good  = int(mc_score >= args.threshold)

            record = {
                "key":       key,
                "problem":   problem,
                "answer":    gt_answer,
                "source":    item["source"],
                "step_idx":  t,
                "n_steps":   len(steps),
                "prefix":    prefix,
                "mc_score":  round(mc_score, 4),
                "label":     is_good,
            }
            fout.write(json.dumps(record) + "\n")
            total_step_records += 1
            any_label_written   = True

            if is_good:
                total_good += 1
            else:
                total_bad += 1

            print(
                f"    step {t+1}/{len(steps)}:  "
                f"MC={mc_score:.2f}  ({correct_count}/{args.k_rollouts})  "
                f"label={'✓ good' if is_good else '✗ bad'}"
            )

            # Stop after first bad step (first error identified)
            if not is_good and prev_score >= args.threshold:
                print(f"    → First error at step {t+1}. Stopping labeling.")
                break

            prev_score = mc_score

        if any_label_written:
            fdone.write(key + "\n")

        # Progress summary every 50 problems
        if (prob_idx + 1) % 50 == 0:
            print(
                f"\n  ── Progress: {prob_idx+1}/{len(problems)} problems  |  "
                f"{total_step_records} step records  |  "
                f"good={total_good}  bad={total_bad} ──\n"
            )

    fout.close()
    fdone.close()

    print("\n" + "═" * 60)
    print(f"  ✅ MC label generation complete!")
    print(f"     Output     : {OUT_FILE}")
    print(f"     Step records: {total_step_records}")
    print(f"     Good steps : {total_good}")
    print(f"     Bad steps  : {total_bad}")
    print(f"\n  Next: run  train_prm.py  to train the PRM on these labels.")
    print("═" * 60)

if __name__ == "__main__":
    main()
