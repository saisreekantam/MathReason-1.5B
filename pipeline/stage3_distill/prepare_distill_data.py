"""
Stage 3 — Filter & Convert OpenR1-Math-220k
─────────────────────────────────────────────
Path: ~/nlp/scripts/stage3_distill/prepare_distill_data.py

Filters for:
  - Correct traces (correctness_math_verify = True)
  - Complete reasoning (is_reasoning_complete = True)
  - Hard sources only (olympiads, amc_aime, aops_forum, cn_contest)
  - Think tokens: 200–4096 (learnability range for 1.5B)
  - No language mixing
  - No multiple choice (answer is not a single letter A/B/C/D/E)

Usage:
  python3 prepare_distill_data.py --sanity   # 20 samples
  python3 prepare_distill_data.py            # full run
"""

import json
import re
import argparse
from pathlib import Path
from collections import Counter

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer


# ─── Paths ───────────────────────────────────────────────────────────────────

WORK_DIR     = Path("/home/revanth/nlp")
CACHE_DIR    = WORK_DIR / "data" / "cache"
FINAL_DIR    = WORK_DIR / "data" / "stage3_final"
STUDENT_CKPT = WORK_DIR / "checkpoints" / "stage2_sft_merged"

# ─── Config ──────────────────────────────────────────────────────────────────

STUDENT_SYSTEM = (
    "You are a mathematical reasoning assistant. "
    "Think step by step inside <think> tags, "
    "then give your final answer inside <solution> tags."
)

HARD_SOURCES     = {"olympiads", "amc_aime", "aops_forum", "cn_contest"}
MIN_THINK_TOKENS = 200
MAX_THINK_TOKENS = 4096


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sanity", action="store_true",
                   help="Inspect only first 20 accepted samples")
    return p.parse_args()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def extract_boxed_answer(text: str):
    """Extract content from \\boxed{...} handling nested braces."""
    match = re.search(r'\\boxed\{', text)
    if not match:
        return None
    start = match.end()
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if   text[i] == '{': depth += 1
        elif text[i] == '}': depth -= 1
        i += 1
    return text[start:i - 1].strip() if depth == 0 else None


def extract_think_and_answer(generation: str):
    """Parse <think>...</think>\\boxed{answer} from one R1 generation."""
    think_match = re.search(r'<think>(.*?)</think>', generation, re.DOTALL)
    if not think_match:
        return None, None
    think_content = think_match.group(1).strip()
    after_think   = generation[think_match.end():]
    answer = extract_boxed_answer(after_think) or extract_boxed_answer(generation)
    return think_content, answer


def is_multiple_choice_answer(answer: str) -> bool:
    """
    Return True if the answer is just a letter — A/B/C/D/E.
    These are MCQ problems — not useful for reasoning training.
    """
    if answer is None:
        return False
    return bool(re.fullmatch(r'[A-Ea-e]', answer.strip()))


def is_good_trace(think_content, answer, tokenizer) -> tuple[bool, str]:
    """Returns (keep, reject_reason)"""
    if think_content is None:              return False, "no_think"
    if answer is None:                     return False, "no_boxed"
    if len(think_content.strip()) < 100:   return False, "think_too_short_chars"
    if is_multiple_choice_answer(answer):  return False, "multiple_choice"

    n_tokens = len(tokenizer.encode(think_content))
    if n_tokens < MIN_THINK_TOKENS: return False, f"tok_short_{n_tokens}"
    if n_tokens > MAX_THINK_TOKENS: return False, f"tok_long_{n_tokens}"

    non_ascii = sum(1 for c in think_content if ord(c) > 127)
    if non_ascii / max(len(think_content), 1) > 0.05:
        return False, "lang_mix"

    return True, "ok"


def to_student_format(problem: str, think_content: str,
                      answer: str, tokenizer) -> str:
    """Convert to Qwen2.5 chat template with student <solution> format."""
    assistant_turn = (
        f"<think>\n{think_content}\n</think>\n"
        f"<solution>\n{answer}\n</solution>"
    )
    messages = [
        {"role": "system",    "content": STUDENT_SYSTEM},
        {"role": "user",      "content": problem.strip()},
        {"role": "assistant", "content": assistant_turn},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading student tokenizer from {STUDENT_CKPT}...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(STUDENT_CKPT), trust_remote_code=True
    )

    print("Loading open-r1/OpenR1-Math-220k...")
    ds = load_dataset(
        "open-r1/OpenR1-Math-220k",
        split="train",
        cache_dir=str(CACHE_DIR),
    )
    print(f"Raw examples: {len(ds):,}\n")

    stats = {
        "total_examples":  0,
        "total_traces":    0,
        "accepted":        0,
        "easy_source":     0,
        "wrong_answer":    0,
        "incomplete":      0,
        "multiple_choice": 0,
        "no_think":        0,
        "no_boxed":        0,
        "tok_short":       0,
        "tok_long":        0,
        "lang_mix":        0,
        "other_reject":    0,
    }

    accepted_records = []

    print("Filtering and converting...")
    for item in ds:
        stats["total_examples"] += 1

        source      = item.get("source", "")
        problem     = item.get("problem", "").strip()
        generations = item.get("generations") or []
        correct     = item.get("correctness_math_verify") or []
        complete    = item.get("is_reasoning_complete") or []

        if not problem or not generations:
            continue

        # Hard source filter
        if source not in HARD_SOURCES:
            stats["easy_source"] += 1
            continue

        # Try each generation — take first that passes all filters
        accepted_this_problem = False
        for i, gen in enumerate(generations):
            stats["total_traces"] += 1

            if i < len(correct) and correct[i] is False:
                stats["wrong_answer"] += 1
                continue

            if i < len(complete) and complete[i] is False:
                stats["incomplete"] += 1
                continue

            think_content, answer = extract_think_and_answer(gen)
            keep, reason = is_good_trace(think_content, answer, tokenizer)

            if not keep:
                if "no_think"         in reason: stats["no_think"]        += 1
                elif "no_boxed"       in reason: stats["no_boxed"]        += 1
                elif "tok_short"      in reason: stats["tok_short"]       += 1
                elif "tok_long"       in reason: stats["tok_long"]        += 1
                elif "lang_mix"       in reason: stats["lang_mix"]        += 1
                elif "multiple_choice" in reason: stats["multiple_choice"] += 1
                else:                             stats["other_reject"]   += 1
                continue

            # ── Accepted ──
            think_tokens = len(tokenizer.encode(think_content))
            student_text = to_student_format(
                problem, think_content, answer, tokenizer
            )
            accepted_records.append({
                "problem":       problem,
                "source":        source,
                "think_content": think_content,
                "answer":        answer,
                "think_tokens":  think_tokens,
                "gen_index":     i,
                "text":          student_text,
            })
            stats["accepted"] += 1
            accepted_this_problem = True
            break  # one trace per problem

        if args.sanity and stats["accepted"] >= 20:
            break

    # ── Stats ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FILTER RESULTS")
    print(f"{'='*60}")
    print(f"  Total examples    : {stats['total_examples']:,}")
    print(f"  Total traces      : {stats['total_traces']:,}")
    print(f"  ✅ Accepted        : {stats['accepted']:,}")
    print(f"  ── Rejections ──")
    print(f"  Easy source       : {stats['easy_source']:,}")
    print(f"  Wrong answer      : {stats['wrong_answer']:,}")
    print(f"  Incomplete        : {stats['incomplete']:,}")
    print(f"  Multiple choice   : {stats['multiple_choice']:,}")
    print(f"  No <think>        : {stats['no_think']:,}")
    print(f"  No \\boxed         : {stats['no_boxed']:,}")
    print(f"  Too short (<200t) : {stats['tok_short']:,}")
    print(f"  Too long (>4096t) : {stats['tok_long']:,}")
    print(f"  Language mixing   : {stats['lang_mix']:,}")

    if accepted_records:
        toks = [r["think_tokens"] for r in accepted_records]
        buckets = {"<500": 0, "500-1K": 0, "1K-2K": 0, "2K-4K": 0}
        for t in toks:
            if   t < 500:  buckets["<500"]   += 1
            elif t < 1000: buckets["500-1K"] += 1
            elif t < 2000: buckets["1K-2K"]  += 1
            else:          buckets["2K-4K"]  += 1

        print(f"\n  Think token distribution:")
        for b, c in buckets.items():
            bar = "█" * (c * 30 // max(buckets.values(), default=1))
            print(f"    {b:<10} {c:>6,}  {bar}")
        print(f"\n  Avg think tokens  : {sum(toks)/len(toks):.0f}")

        src_counts = Counter(r["source"] for r in accepted_records)
        print(f"\n  Source distribution:")
        for src, cnt in src_counts.most_common():
            print(f"    {src:<30} {cnt:>6,}")

    # ── Sanity print ─────────────────────────────────────────────────────────
    if args.sanity:
        print(f"\n{'─'*60}")
        print("  SANITY SAMPLES (first 3)")
        print(f"{'─'*60}")
        for r in accepted_records[:3]:
            print(f"\n  Problem     : {r['problem'][:80]}...")
            print(f"  Source      : {r['source']}")
            print(f"  Think tokens: {r['think_tokens']}")
            print(f"  Answer      : {r['answer']}")
            print(f"  Student text: {r['text'][:300]}...")
        print(f"\n✅ Sanity passed — {stats['accepted']} samples")
        print(f"   No MCQ answers, format correct → run full:")
        print(f"   python3 prepare_distill_data.py")
        return

    # ── Save ─────────────────────────────────────────────────────────────────
    if not accepted_records:
        print("\n❌ No records accepted — check filters")
        return

    hf_ds = Dataset.from_list([{"text": r["text"]} for r in accepted_records])
    hf_ds.save_to_disk(str(FINAL_DIR / "distill_dataset"))

    with open(FINAL_DIR / "distill_full.jsonl", "w") as f:
        for r in accepted_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✅ Saved {stats['accepted']:,} examples")
    print(f"   HF dataset : {FINAL_DIR}/distill_dataset")
    print(f"   JSONL      : {FINAL_DIR}/distill_full.jsonl")
    print(f"\n   Next: python3 stage3_train.py")


if __name__ == "__main__":
    main()
