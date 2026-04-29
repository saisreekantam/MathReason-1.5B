"""
Stage 3 — Distillation Trace Generator
────────────────────────────────────────
Server : revanth@172.16.192.168
Path   : ~/nlp/scripts/stage3_distill/generate_traces.py

Dataset : AI-MO/NuminaMath-CoT  (hard sources only — no basic school math)
Teachers:
  GPU 0 → DeepSeek-R1-Distill-Qwen-7B   (medium-hard problems)
  GPU 1 → DeepSeek-R1-Distill-Qwen-14B  (hard-only problems)

DeepSeek-R1-Distill output format (verified from official docs):
  <think>
    ... step by step reasoning, self-correction, backtracking ...
  </think>
  The answer is \\boxed{42}

We convert this → student format:
  <think>
    ... reasoning ...
  </think>
  <solution>
  42
  </solution>

Token limits (based on learnability research for 1.5B models):
  7B  teacher: 200 – 4096 think tokens
  14B teacher: 200 – 8192 think tokens

Usage:
  # Always run sanity first (20 problems, ~5 min)
  CUDA_VISIBLE_DEVICES=0 python3 generate_traces.py --teacher 7b  --sanity
  CUDA_VISIBLE_DEVICES=1 python3 generate_traces.py --teacher 14b --sanity

  # Full generation
  CUDA_VISIBLE_DEVICES=0 python3 generate_traces.py --teacher 7b
  CUDA_VISIBLE_DEVICES=1 python3 generate_traces.py --teacher 14b

  # Merge both into final dataset
  python3 generate_traces.py --merge
"""

import argparse
import json
import os
import re
import random
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# ─── Paths ───────────────────────────────────────────────────────────────────

WORK_DIR        = Path("/home/revanth/nlp")
DATA_CACHE      = WORK_DIR / "data" / "cache"
TRACES_DIR      = WORK_DIR / "data" / "stage3_traces"
FINAL_DIR       = WORK_DIR / "data" / "stage3_final"
STUDENT_CKPT    = WORK_DIR / "checkpoints" / "stage2_sft_merged"


# ─── Teacher models ──────────────────────────────────────────────────────────

TEACHER_MODELS = {
    "7b":  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
}

# ─── Generation settings (DeepSeek official recommendations) ─────────────────
#   - temperature 0.5–0.7 (0.6 recommended)
#   - top_p 0.95
#   - NO system prompt — all instructions in user turn
#   - Force <think>\n at start of generation

TEMPERATURE = 0.6
TOP_P       = 0.95

# ─── Token limits per teacher ────────────────────────────────────────────────
#   Based on: "Small Model Learnability Gap" paper (2025)
#   1.5B student absorbs traces up to ~4K think tokens well
#   14B produces longer traces for harder problems — allowed up to 8K

LIMITS = {
    "7b":  {"min": 200, "max": 4096, "gen_budget": 5000},
    "14b": {"min": 200, "max": 8192, "gen_budget": 10000},
}

# ─── NuminaMath hard sources (skip basic school problems) ────────────────────
#   SFT already covered the easy/medium NuminaMath problems.
#   Distillation should only use reasoning-heavy competition problems.

HARD_SOURCES = [
    "amc_aime",      # AMC 10/12 + AIME
    "olympiads",     # International Math Olympiad + national olympiads
    "aops_forum",    # Art of Problem Solving competition discussions
    "cn_contest",    # Chinese national math competitions
]

# ─── Student system prompt (must match stage2_sft.py exactly) ────────────────

STUDENT_SYSTEM = (
    "You are a mathematical reasoning assistant. "
    "Think step by step inside <think> tags, "
    "then give your final answer inside <solution> tags."
)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", choices=["7b", "14b"], default="7b",
                   help="Which teacher model to use")
    p.add_argument("--sanity", action="store_true",
                   help="Run on only 20 problems to verify pipeline end-to-end")
    p.add_argument("--merge", action="store_true",
                   help="Merge 7B + 14B traces into final training dataset")
    p.add_argument("--n-problems", type=int, default=None,
                   help="Override number of problems (default: 80K for 7b, 25K for 14b)")
    return p.parse_args()


# ─── NuminaMath loading & hard-source filtering ──────────────────────────────

def load_numina_hard(teacher: str, n_target: int, seed: int = 42):
    """
    Load NuminaMath-CoT and filter to hard competition problems only.

    NuminaMath columns: source | problem | solution | messages
    Hard sources: amc_aime, olympiads, aops_forum, cn_contest

    For 7b  → use all hard sources (medium + hard difficulty)
    For 14b → use only olympiads + amc_aime (hardest tier only)
    """
    print(f"\nLoading NuminaMath-CoT (hard subset, teacher={teacher})...")

    ds = load_dataset(
        "AI-MO/NuminaMath-CoT",
        split="train",
        cache_dir=str(DATA_CACHE),
    )
    print(f"  Full dataset: {len(ds):,} problems")

    if teacher == "14b":
        # Hardest tier only for 14B — olympiad + AIME
        use_sources = ["olympiads", "amc_aime"]
    else:
        # All hard sources for 7B
        use_sources = HARD_SOURCES

    ds_hard = ds.filter(
        lambda x: any(
            src in x.get("source", "").lower()
            for src in use_sources
        ),
        num_proc=4,
        desc="Filtering hard sources",
    )
    print(f"  After hard-source filter: {len(ds_hard):,} problems")
    print(f"  Sources used: {use_sources}")

    # Shuffle and take n_target
    ds_hard = ds_hard.shuffle(seed=seed)
    n_actual = min(n_target, len(ds_hard))
    ds_hard = ds_hard.select(range(n_actual))
    print(f"  Selected: {n_actual:,} problems")

    # Show source distribution
    from collections import Counter
    sources = Counter(ds_hard["source"])
    print("  Source distribution:")
    for src, cnt in sources.most_common(10):
        print(f"    {src:<30} {cnt:>6,}")

    return ds_hard


# ─── Answer extraction ───────────────────────────────────────────────────────

def extract_boxed_answer(text: str):
    r"""
    Extract content from \boxed{...} handling nested braces.

    DeepSeek-R1-Distill always puts final answer in \boxed{}.
    Example: "The answer is \boxed{x^2 + 1}" → "x^2 + 1"
    Example: "So \boxed{\frac{1}{2}}"        → "\frac{1}{2}"
    """
    match = re.search(r'\\boxed\{', text)
    if not match:
        # Fallback: last number in the response
        nums = re.findall(r'-?[\d,]+(?:\.\d+)?', text)
        return nums[-1].replace(',', '') if nums else None

    start = match.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1

    if depth != 0:
        return None  # malformed boxed

    return text[start:i - 1].strip()


def extract_think_and_answer(raw_response: str):
    """
    Parse DeepSeek-R1-Distill raw output.

    Expected format:
        <think>
        ... reasoning ...
        </think>
        The answer is \\boxed{42}

    Returns: (think_content, answer) or (None, None)
    """
    # Extract <think>...</think>
    think_match = re.search(r'<think>(.*?)</think>', raw_response, re.DOTALL)
    if not think_match:
        return None, None

    think_content = think_match.group(1).strip()

    # Extract \boxed{} answer from text AFTER </think>
    after_think = raw_response[think_match.end():]
    answer = extract_boxed_answer(after_think)

    # If not found after </think>, try whole response
    if answer is None:
        answer = extract_boxed_answer(raw_response)

    return think_content, answer


# ─── Format conversion ───────────────────────────────────────────────────────

def build_student_training_example(problem: str, think_content: str,
                                   answer: str, student_tokenizer) -> str:
    """
    Convert teacher output → student training format.

    Teacher produced:
        <think> reasoning </think>  \\boxed{answer}

    We produce (wrapped in Qwen2.5 chat template):
        system: You are a mathematical reasoning assistant...
        user:   <problem>
        asst:   <think> reasoning </think>
                <solution> answer </solution>
    """
    assistant_turn = (
        f"<think>\n{think_content}\n</think>\n"
        f"<solution>\n{answer}\n</solution>"
    )

    messages = [
        {"role": "system",    "content": STUDENT_SYSTEM},
        {"role": "user",      "content": problem.strip()},
        {"role": "assistant", "content": assistant_turn},
    ]

    return student_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


# ─── Quality filter ──────────────────────────────────────────────────────────

def is_good_trace(think_content, answer, teacher: str, teacher_tokenizer) -> tuple[bool, str]:
    """
    Filter trace for quality and learnability.
    Returns (keep: bool, reason: str)
    """
    if think_content is None:
        return False, "no_think_tags"

    if answer is None:
        return False, "no_boxed_answer"

    if len(think_content.strip()) < 50:
        return False, "think_too_short_chars"

    think_tokens = len(teacher_tokenizer.encode(think_content))
    lim = LIMITS[teacher]

    if think_tokens < lim["min"]:
        return False, f"too_short_{think_tokens}tok"

    if think_tokens > lim["max"]:
        return False, f"too_long_{think_tokens}tok"

    # Language mixing check — R1 sometimes slips into Chinese
    non_ascii = sum(1 for c in think_content if ord(c) > 127)
    if non_ascii / max(len(think_content), 1) > 0.05:
        return False, "language_mixing"

    # Repetition check — R1 sometimes loops
    words = think_content.split()
    if len(words) > 50:
        # Check if last 30 words repeat heavily
        last30 = " ".join(words[-30:])
        prev30 = " ".join(words[-60:-30]) if len(words) >= 60 else ""
        if prev30 and last30 == prev30:
            return False, "repetition_loop"

    return True, "ok"


# ─── Main generation ─────────────────────────────────────────────────────────

def generate_traces(teacher: str, sanity: bool, n_override=None):
    model_name = TEACHER_MODELS[teacher]
    lim        = LIMITS[teacher]

    # In sanity mode: only 20 problems
    if sanity:
        n_target = 20
    elif n_override:
        n_target = n_override
    else:
        n_target = 80000 if teacher == "7b" else 25000

    print("\n" + "█" * 65)
    print(f"  STAGE 3 — DISTILLATION TRACE GENERATION")
    print(f"  Teacher  : {model_name}")
    print(f"  Problems : {n_target} {'(SANITY)' if sanity else ''}")
    print(f"  Think tok: {lim['min']} – {lim['max']}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("█" * 65)

    # ── Output path ──
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    suffix      = "_sanity" if sanity else ""
    output_path = TRACES_DIR / f"traces_{teacher}{suffix}.jsonl"

    # ── Resume: skip already done problems ──
    done_problems = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    done_problems.add(json.loads(line)["problem"][:80])
                except Exception:
                    pass
        if done_problems:
            print(f"\n  Resuming — {len(done_problems)} already done, skipping...")

    # ── Load teacher model ──
    print(f"\nLoading teacher: {model_name}")
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        trust_remote_code=True,
    )
    teacher_model.eval()
    print(f"  Teacher loaded on: {next(teacher_model.parameters()).device}")

    # ── Load student tokenizer for format conversion ──
    print(f"Loading student tokenizer from: {STUDENT_CKPT}")
    student_tokenizer = AutoTokenizer.from_pretrained(
        str(STUDENT_CKPT), trust_remote_code=True
    )

    # ── Load problems ──
    problems_ds = load_numina_hard(teacher, n_target)

    # ── Stats tracker ──
    stats = {
        "total":            0,
        "accepted":         0,
        "no_think_tags":    0,
        "no_boxed_answer":  0,
        "too_short":        0,
        "too_long":         0,
        "language_mixing":  0,
        "repetition_loop":  0,
        "other_reject":     0,
    }

    print(f"\nStarting generation → {output_path}\n")

    with open(output_path, "a") as f_out:
        for item in tqdm(problems_ds, desc=f"Teacher={teacher}"):
            problem = item.get("problem", "").strip()
            if not problem:
                continue

            # Skip already processed
            if problem[:80] in done_problems:
                continue

            stats["total"] += 1

            # ── Build prompt ──
            # Official DeepSeek recommendation:
            #   - NO system prompt
            #   - Directive in user message: "put final answer within \boxed{}"
            #   - Force <think>\n prefix to ensure reasoning starts immediately
            user_msg = (
                f"{problem}\n\n"
                "Please reason step by step, and put your final answer within \\boxed{}."
            )
            messages = [{"role": "user", "content": user_msg}]

            prompt_text = teacher_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            # Force model into thinking mode immediately
            prompt_text += "<think>\n"

            inputs = teacher_tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=768,   # problem text cap
            ).to("cuda")

            # ── Generate ──
            with torch.no_grad():
                output_ids = teacher_model.generate(
                    **inputs,
                    max_new_tokens=lim["gen_budget"],
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    repetition_penalty=1.05,
                    pad_token_id=teacher_tokenizer.eos_token_id,
                )

            # Decode new tokens only; prepend <think> we injected
            new_tokens   = output_ids[0][inputs["input_ids"].shape[1]:]
            raw_response = "<think>\n" + teacher_tokenizer.decode(
                new_tokens, skip_special_tokens=True
            )

            # ── Parse output ──
            think_content, answer = extract_think_and_answer(raw_response)

            # ── Filter ──
            keep, reason = is_good_trace(
                think_content, answer, teacher, teacher_tokenizer
            )

            if not keep:
                # Map reason to stat bucket
                if "no_think"      in reason: stats["no_think_tags"]   += 1
                elif "no_boxed"    in reason: stats["no_boxed_answer"] += 1
                elif "too_short"   in reason: stats["too_short"]       += 1
                elif "too_long"    in reason: stats["too_long"]        += 1
                elif "language"    in reason: stats["language_mixing"] += 1
                elif "repetition"  in reason: stats["repetition_loop"] += 1
                else:                         stats["other_reject"]    += 1

                if sanity:
                    print(f"  [REJECTED] reason={reason} | {problem[:60]}...")
                continue

            # ── Build student training example ──
            think_tokens  = len(teacher_tokenizer.encode(think_content))
            student_text  = build_student_training_example(
                problem, think_content, answer, student_tokenizer
            )

            stats["accepted"] += 1

            record = {
                "problem":       problem,
                "source":        item.get("source", "unknown"),
                "think_content": think_content,
                "answer":        answer,
                "think_tokens":  think_tokens,
                "teacher":       teacher,
                "text":          student_text,   # ← final SFT training field
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()

            # ── Sanity: print each result ──
            if sanity:
                print(f"\n{'─'*60}")
                print(f"  Problem   : {problem[:80]}...")
                print(f"  Source    : {item.get('source', '?')}")
                print(f"  Think tok : {think_tokens}")
                print(f"  Answer    : {answer}")
                print(f"  Think prev: {think_content[:200]}...")
                print(f"  Student ex: {student_text[:300]}...")

            # ── Progress every 500 ──
            if not sanity and stats["total"] % 500 == 0:
                acc  = stats["accepted"]
                tot  = stats["total"]
                rate = acc / tot * 100
                print(
                    f"  [{tot:>6,}]  accepted={acc:,} ({rate:.1f}%) | "
                    f"too_long={stats['too_long']} | "
                    f"no_ans={stats['no_boxed_answer']} | "
                    f"lang={stats['language_mixing']}"
                )

    # ── Final summary ──
    tot  = max(stats["total"], 1)
    acc  = stats["accepted"]
    print(f"\n{'='*65}")
    print(f"  GENERATION COMPLETE — {teacher.upper()} teacher")
    print(f"{'='*65}")
    print(f"  Total attempted   : {stats['total']:,}")
    print(f"  ✅ Accepted        : {acc:,}  ({acc/tot*100:.1f}%)")
    print(f"  ❌ No <think> tags : {stats['no_think_tags']:,}")
    print(f"  ❌ No \\boxed answer: {stats['no_boxed_answer']:,}")
    print(f"  ❌ Too short       : {stats['too_short']:,}")
    print(f"  ❌ Too long        : {stats['too_long']:,}")
    print(f"  ❌ Language mixing : {stats['language_mixing']:,}")
    print(f"  ❌ Repetition      : {stats['repetition_loop']:,}")
    print(f"\n  Saved to: {output_path}")

    if sanity:
        print(f"\n{'─'*65}")
        print("  SANITY COMPLETE")
        print(f"  Accept rate: {acc/tot*100:.1f}%  (aim for >50%)")
        print(f"  If too_long is high → lower MAX_THINK_TOKENS")
        print(f"  If no_boxed_answer is high → check prompt format")
        print(f"  If looks good → run full generation:")
        print(f"    CUDA_VISIBLE_DEVICES=0 python3 generate_traces.py --teacher 7b")
        print(f"{'─'*65}")

    return output_path


# ─── Merge 7B + 14B traces ───────────────────────────────────────────────────

def merge_traces():
    """
    Merge 7B and 14B traces → final distillation dataset.
    Mix: 80% 7B (medium-hard) + 20% 14B (hardest)
    """
    print("\n" + "█" * 65)
    print("  MERGING TRACES → FINAL DISTILLATION DATASET")
    print("█" * 65)

    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    def load_jsonl(path):
        records = []
        if not path.exists():
            print(f"  ⚠️  Not found: {path}")
            return records
        with open(path) as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
        return records

    t7b  = load_jsonl(TRACES_DIR / "traces_7b.jsonl")
    t14b = load_jsonl(TRACES_DIR / "traces_14b.jsonl")

    print(f"  7B  traces : {len(t7b):,}")
    print(f"  14B traces : {len(t14b):,}")

    if t7b:
        avg7  = sum(r["think_tokens"] for r in t7b)  / len(t7b)
        print(f"  Avg think tokens (7B) : {avg7:.0f}")
    if t14b:
        avg14 = sum(r["think_tokens"] for r in t14b) / len(t14b)
        print(f"  Avg think tokens (14B): {avg14:.0f}")

    # 80/20 mix
    random.seed(42)
    n7  = min(len(t7b),  80000)
    n14 = min(len(t14b), 20000)

    sel7  = random.sample(t7b,  n7)
    sel14 = random.sample(t14b, n14) if t14b else []

    merged = sel7 + sel14
    random.shuffle(merged)

    print(f"\n  Final mix:")
    print(f"    7B  : {n7:,}  (~80%)")
    print(f"    14B : {n14:,}  (~20%)")
    print(f"    Total: {len(merged):,}")

    # Save as HuggingFace dataset (for use in stage3_train.py)
    hf_ds = Dataset.from_list([{"text": r["text"]} for r in merged])
    hf_ds.save_to_disk(str(FINAL_DIR / "distill_dataset"))

    # Save full records as jsonl (for inspection/debugging)
    with open(FINAL_DIR / "distill_full.jsonl", "w") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Token length distribution
    all_tokens = [r["think_tokens"] for r in merged]
    buckets    = {"<500": 0, "500-1K": 0, "1K-2K": 0, "2K-4K": 0, "4K+": 0}
    for t in all_tokens:
        if   t < 500:  buckets["<500"]   += 1
        elif t < 1000: buckets["500-1K"] += 1
        elif t < 2000: buckets["1K-2K"]  += 1
        elif t < 4000: buckets["2K-4K"]  += 1
        else:          buckets["4K+"]    += 1

    print(f"\n  Think token distribution:")
    for bucket, cnt in buckets.items():
        bar = "█" * (cnt * 40 // max(buckets.values()))
        print(f"    {bucket:<10} {cnt:>6,}  {bar}")

    print(f"\n  ✅ Dataset saved  : {FINAL_DIR / 'distill_dataset'}")
    print(f"  ✅ Full JSONL     : {FINAL_DIR / 'distill_full.jsonl'}")
    print(f"\n  Load in stage3_train.py:")
    print(f"    Dataset.load_from_disk('{FINAL_DIR}/distill_dataset')")

    # Show one sample
    if merged:
        s = merged[0]
        print(f"\n  ── Sample entry ──────────────────────────────────────")
        print(f"  Source      : {s['source']}")
        print(f"  Problem     : {s['problem'][:80]}...")
        print(f"  Think tokens: {s['think_tokens']}")
        print(f"  Answer      : {s['answer']}")
        print(f"  Training text preview:\n{s['text'][:400]}...")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.merge:
        merge_traces()
        return

    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    generate_traces(args.teacher, args.sanity, args.n_problems)


if __name__ == "__main__":
    main()
