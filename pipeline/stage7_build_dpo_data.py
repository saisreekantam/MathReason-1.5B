"""
stage7_build_dpo_data.py — DPO Dataset Generator for MathReason-1.5B
=====================================================================
Generates self-play preference pairs (chosen/rejected) from the current
best checkpoint for DPO alignment training.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 stage7_build_dpo_data.py

Outputs:
    ~/nlp/data/dpo_pairs/dpo_pairs_raw.jsonl   — all valid pairs
    ~/nlp/data/dpo_pairs/dpo_pairs_train.jsonl — filtered, ready for DPOTrainer

Resume: safe to Ctrl+C and restart — skips already-processed problems via
        a seen_keys set built from existing output file.
"""

import os, re, json, time, random, logging
from pathlib import Path
from datetime import datetime
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── Config ──────────────────────────────────────────────────────────────────

MODEL_PATH     = os.path.expanduser("~/nlp/checkpoints/gdpo_p2_v1_merged")
# Lineage: stage4d_gdpo → stage5_gap_sft_merged → gdpo_p2_v1_merged (← latest)

OUTPUT_DIR     = Path(os.path.expanduser("~/nlp/data/dpo_pairs"))
RAW_FILE       = OUTPUT_DIR / "dpo_pairs_raw.jsonl"
TRAIN_FILE     = OUTPUT_DIR / "dpo_pairs_train.jsonl"
LOG_FILE       = OUTPUT_DIR / "generation_log.txt"

NUM_GENERATIONS  = 8       # completions per problem
TEMPERATURE      = 0.7
MAX_NEW_TOKENS   = 2048    # must match your training max_length
MAX_PROMPT_TOKENS = 512

# Pair quality filters
MIN_THINK_WORDS  = 50      # skip truncated responses
MAX_THINK_WORDS  = 800     # skip runaway loops (for chosen side)
MIN_PAIRS_TARGET = 500     # stop early if reached (set high to keep going)

# Dataset sizes to sample
GSM8K_N    = 1319   # full test split (unseen problems)
METAMATH_N = 500    # MetaMath MATH variants
NUMINAMATH_GAP_N = 300  # NuminaMath targeted at specific failure modes

# Domain-stratified MATH sampling (L3 + L4)
# Weighted toward diagnosed failure modes from PROGRESS.md diagnostics
MATH_DOMAIN_TARGETS = {
    "Counting & Probability": 200,   # primary failure: brute-force loops (Q03 type)
    "Algebra":                150,   # primary failure: wrong formula selection (Q13 type)
    "Intermediate Algebra":   100,   # harder algebra — AIME-relevant
    "Number Theory":           80,   # GSM8K hard tail
    "Geometry":                50,   # smaller gap
    "Precalculus":             30,   # AIME-relevant
    "Prealgebra":              20,   # low priority — model mostly handles this
}
# Total MATH: ~630 problems across domains

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Logging ─────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE),
    ],
)
log = logging.getLogger(__name__)

# ─── Answer extraction ───────────────────────────────────────────────────────

def extract_answer(text: str) -> str | None:
    """
    Priority-ordered smart extractor — matches eval_math500 logic.
    1. <solution>...</solution> tag
    2. \\boxed{...} last occurrence
    3. First 60% of text: "So/Therefore/Thus/Hence = X"
    4. Last "= X" in first 60%
    5. Last number in full text (fallback)
    """
    # 1. Solution tag
    m = re.search(r"<solution>(.*?)</solution>", text, re.DOTALL | re.IGNORECASE)
    if m:
        val = m.group(1).strip()
        val = re.sub(r"[,$Rs%]", "", val).strip()
        return val if val else None

    # 2. \boxed{...} — last occurrence
    boxes = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxes:
        return boxes[-1].strip()

    # 3-5. Search first 60% of text only (avoid loop region)
    cutoff = int(len(text) * 0.60)
    early = text[:cutoff]

    # 3. Confident conclusion
    m = re.search(
        r"(?:so|therefore|thus|hence)[^.]*?[=:]\s*([\d,.$Rs%\-\/]+)",
        early, re.IGNORECASE
    )
    if m:
        return re.sub(r"[,$Rs%]", "", m.group(1)).strip()

    # 4. Last "= X" in early region
    nums = re.findall(r"=\s*([\d,.\-]+)", early)
    if nums:
        return re.sub(r",", "", nums[-1]).strip()

    # 5. Last number anywhere
    nums = re.findall(r"[\d,.\-]+", text)
    return re.sub(r",", "", nums[-1]).strip() if nums else None


def normalize(val: str) -> str:
    if val is None:
        return ""
    val = val.strip().lower()
    val = re.sub(r"[,$Rs%\s]", "", val)
    # Remove trailing zeros after decimal: 18.00 → 18
    try:
        f = float(val)
        if f == int(f):
            return str(int(f))
        return f"{f:.6f}".rstrip("0")
    except Exception:
        return val


def answers_match(pred: str | None, gt: str) -> bool:
    if pred is None:
        return False
    p, g = normalize(pred), normalize(gt)
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 0.5
    except Exception:
        return False


# ─── Think-word counter ───────────────────────────────────────────────────────

def count_think_words(text: str) -> int:
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return len(m.group(1).split())
    # If no tags, count first 60% of text
    return len(text[:int(len(text) * 0.6)].split())


# ─── Format prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Think through problems step by step inside <think>...</think> tags, "
    "then provide your final answer inside <solution>...</solution> tags."
)

def format_prompt(question: str, tokenizer) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question.strip()},
    ]
    # Use chat template if available, else fallback
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    return f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{question.strip()}\n<|assistant|>\n"


# ─── Dataset loading ──────────────────────────────────────────────────────────

def load_problems() -> list[dict]:
    """Returns list of {"key": str, "question": str, "answer": str}"""
    problems = []

    # 1. GSM8K test split (model hasn't trained on test set)
    log.info("Loading GSM8K test...")
    try:
        gsm = load_dataset("gsm8k", "main", split="test")
        for i, ex in enumerate(gsm):
            answer_raw = ex["answer"]
            # GSM8K answer format: "... #### 42"
            gt = answer_raw.split("####")[-1].strip()
            problems.append({
                "key":      f"gsm8k_test_{i}",
                "question": ex["question"],
                "answer":   normalize(gt),
                "source":   "gsm8k_test",
            })
        log.info(f"  GSM8K: {len(gsm)} problems")
    except Exception as e:
        log.warning(f"  GSM8K load failed: {e}")

    # 2. MATH — domain-stratified (Level 3 + Level 4)
    # Weighted toward Counting & Probability and Algebra — diagnosed failure modes.
    # L4 included alongside L3 to ensure enough variance for strong signal pairs.
    log.info("Loading MATH (domain-stratified L3+L4)...")
    try:
        math_ds = load_dataset("lighteval/MATH", "all", split="train", trust_remote_code=True)

        # Group by subject, keeping only L3 and L4
        by_subject: dict[str, list] = {}
        for ex in math_ds:
            subj = ex.get("type", "Unknown")
            lvl  = ex.get("level", "")
            if lvl not in ("Level 3", "Level 4"):
                continue
            by_subject.setdefault(subj, []).append(ex)

        math_added = 0
        for subject, target_n in MATH_DOMAIN_TARGETS.items():
            pool = by_subject.get(subject, [])
            if not pool:
                log.warning(f"  MATH subject '{subject}' not found — skipping")
                continue
            random.shuffle(pool)
            subject_key = subject.replace(" & ", "_").replace(" ", "_").lower()
            added = 0
            for i, ex in enumerate(pool[:target_n]):
                solution = ex.get("solution", "")
                boxes = re.findall(r"\\boxed\{([^}]+)\}", solution)
                gt = boxes[-1].strip() if boxes else ""
                if not gt:
                    continue
                problems.append({
                    "key":      f"math_{subject_key}_{i}",
                    "question": ex["problem"],
                    "answer":   normalize(gt),
                    "source":   f"math_{subject_key}",
                    "level":    ex.get("level", ""),
                })
                added += 1
            math_added += added
            log.info(f"  MATH {subject:<25}: {added:>3} problems (target {target_n})")

        log.info(f"  MATH total: {math_added} problems across {len(MATH_DOMAIN_TARGETS)} domains")
    except Exception as e:
        log.warning(f"  MATH load failed: {e}")

    # 3. MetaMath MATH variants (not GSM8K type)
    log.info("Loading MetaMathQA MATH variants...")
    try:
        meta = load_dataset("meta-math/MetaMathQA", split="train")
        # Filter to MATH-type (not GSM8K rephrases) — use .filter() not list comp
        math_type = meta.filter(
            lambda ex: ex.get("type", "").startswith("MATH"),
            num_proc=2,
        )
        indices = list(range(len(math_type)))
        random.shuffle(indices)
        indices = indices[:METAMATH_N]
        for i, idx in enumerate(indices):
            ex = math_type[idx]
            answer_raw = ex.get("response", "")
            # MetaMath format: "... The answer is: X"
            m = re.search(r"[Tt]he answer is:?\s*([\d,.\-\/]+)", answer_raw)
            boxes = re.findall(r"\\boxed\{([^}]+)\}", answer_raw)
            if boxes:
                gt = boxes[-1].strip()
            elif m:
                gt = m.group(1).strip()
            else:
                continue
            problems.append({
                "key":      f"metamath_{i}",
                "question": ex["query"],
                "answer":   normalize(gt),
                "source":   "metamath",
            })
        log.info(f"  MetaMath: {sum(1 for p in problems if p['source']=='metamath')} problems")
    except Exception as e:
        log.warning(f"  MetaMathQA load failed: {e}")

    # 4. NuminaMath — gap-targeted competition problems
    # Keywords match the two diagnosed failure modes:
    #   - Counting & Probability gaps (adjacent, circular, without replacement)
    #   - Method selection gaps (work rate, mixture/concentration)
    # Hard competition sources only — these give highest variance completions
    # (2-6/8 correct sweet spot) = strongest DPO signal.
    log.info("Loading NuminaMath gap-targeted problems...")
    try:
        numinamath = load_dataset("AI-MO/NuminaMath-CoT", split="train")

        GAP_KEYWORDS = [
            # Counting / arrangement failures
            "no two adjacent", "not adjacent", "non-adjacent",
            "circular arrangement", "circular permutation",
            "at least one of each", "neither", "at most",
            # Probability failures
            "without replacement", "probability that",
            "expected number", "expected value",
            # Work-rate / method selection failures
            "work rate", "rate of work", "together they",
            "pipe", "fills the tank", "empties",
            # Mixture / concentration failures
            "mixture", "concentration", "solution contains",
            "liters of", "percent alcohol",
        ]

        HARD_SOURCES = {"amc_aime", "olympiads", "cn_contest", "aops_forum"}

        gap_pool = [
            ex for ex in numinamath
            if ex.get("source") in HARD_SOURCES
            and any(kw in ex.get("problem", "").lower() for kw in GAP_KEYWORDS)
        ]
        random.shuffle(gap_pool)
        log.info(f"  NuminaMath gap pool: {len(gap_pool)} candidates")

        added = 0
        for i, ex in enumerate(gap_pool[:NUMINAMATH_GAP_N]):
            solution = ex.get("solution", "")
            boxes = re.findall(r"\\boxed\{([^}]+)\}", solution)
            gt = boxes[-1].strip() if boxes else ""
            if not gt:
                continue
            problems.append({
                "key":      f"numinamath_gap_{i}",
                "question": ex["problem"],
                "answer":   normalize(gt),
                "source":   "numinamath_gap_targeted",
            })
            added += 1
        log.info(f"  NuminaMath gap-targeted: {added} problems")
    except Exception as e:
        log.warning(f"  NuminaMath load failed: {e}")

    log.info(f"Total problems loaded: {len(problems)}")
    random.shuffle(problems)
    return problems


# ─── Resume: load already-seen keys ──────────────────────────────────────────

def load_seen_keys() -> set[str]:
    seen = set()
    if RAW_FILE.exists():
        with open(RAW_FILE) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    seen.add(obj["key"])
                except Exception:
                    pass
    return seen


def count_pairs() -> int:
    if not RAW_FILE.exists():
        return 0
    return sum(1 for _ in open(RAW_FILE))


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model():
    log.info(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    log.info("  Model loaded ✅")
    return model, tokenizer


# ─── Generation ───────────────────────────────────────────────────────────────

@torch.inference_mode()
def generate_completions(
    model, tokenizer, prompt: str, n: int = NUM_GENERATIONS
) -> list[str]:
    """Generate n completions for a single prompt."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_PROMPT_TOKENS,
    ).to(DEVICE)

    prompt_len = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=0.95,
        num_return_sequences=n,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    completions = []
    for out in outputs:
        # Decode only new tokens (strip prompt)
        new_tokens = out[prompt_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        completions.append(text.strip())

    return completions


# ─── Pair building ────────────────────────────────────────────────────────────

def build_pair(question: str, gt_answer: str, completions: list[str]) -> dict | None:
    """
    From N completions, select:
      - chosen:   correct answer, think words in [MIN, MAX], has </solution>
      - rejected: wrong answer, has some reasoning (not empty)

    Returns None if no valid pair can be formed.
    Best signal: problems where 2-6 of 8 completions are correct.
    """
    correct, wrong = [], []

    for c in completions:
        pred = extract_answer(c)
        think_words = count_think_words(c)
        has_solution_tag = "</solution>" in c.lower()

        if answers_match(pred, gt_answer):
            # Chosen must have clean termination and reasonable length
            if has_solution_tag and MIN_THINK_WORDS <= think_words <= MAX_THINK_WORDS:
                correct.append((c, think_words))
        else:
            # Rejected must have some content (not trivially empty)
            if len(c.split()) > 20:
                wrong.append(c)

    # Dead signal: all correct (model already mastered) or all wrong (no chosen)
    if not correct or not wrong:
        return None

    # Prefer shortest correct (avoids verbose but technically correct choices)
    correct.sort(key=lambda x: x[1])
    chosen   = correct[0][0]

    # Prefer longest wrong (captures loop behavior — that's what we want to penalize)
    wrong.sort(key=lambda x: len(x), reverse=True)
    rejected = wrong[0]

    return {
        "chosen":   chosen,
        "rejected": rejected,
        "n_correct": len(correct),
        "n_wrong":   len(wrong),
        "n_total":   len(completions),
    }


# ─── Main loop ────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("stage7_build_dpo_data.py — DPO Dataset Generator")
    log.info(f"Model: {MODEL_PATH}")
    log.info(f"Output: {OUTPUT_DIR}")
    log.info("=" * 60)

    problems = load_problems()
    seen_keys = load_seen_keys()
    existing_pairs = count_pairs()

    log.info(f"Resuming from {existing_pairs} existing pairs | {len(seen_keys)} seen problems")

    model, tokenizer = load_model()

    pair_count = existing_pairs
    skipped_seen   = 0
    skipped_no_pair = 0
    t_start = time.time()

    with open(RAW_FILE, "a") as f_raw:
        for idx, prob in enumerate(problems):
            key = prob["key"]

            if key in seen_keys:
                skipped_seen += 1
                continue

            question  = prob["question"]
            gt_answer = prob["answer"]

            if not gt_answer:
                skipped_no_pair += 1
                continue

            # Progress heartbeat every 20 problems
            if (idx - skipped_seen) % 20 == 0:
                elapsed = time.time() - t_start
                rate = max(idx - skipped_seen - skipped_no_pair, 1)
                log.info(
                    f"[{idx+1}/{len(problems)}] pairs={pair_count} | "
                    f"elapsed={elapsed/60:.1f}m | "
                    f"no_pair_skips={skipped_no_pair}"
                )

            prompt = format_prompt(question, tokenizer)

            try:
                completions = generate_completions(model, tokenizer, prompt)
            except torch.cuda.OutOfMemoryError:
                log.warning(f"OOM on problem {key} — skipping")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                log.warning(f"Generation error on {key}: {e}")
                continue

            pair = build_pair(question, gt_answer, completions)

            if pair is None:
                skipped_no_pair += 1
                seen_keys.add(key)
                # Still log the key so we don't retry on resume
                record = {
                    "key":    key,
                    "source": prob["source"],
                    "status": "no_pair",
                    "n_correct": sum(1 for c in completions if answers_match(extract_answer(c), gt_answer)),
                    "n_total": len(completions),
                }
                f_raw.write(json.dumps(record) + "\n")
                f_raw.flush()
                continue

            record = {
                "key":       key,
                "source":    prob["source"],
                "status":    "pair",
                "prompt":    format_prompt(question, tokenizer),
                "question":  question,
                "gt_answer": gt_answer,
                "chosen":    pair["chosen"],
                "rejected":  pair["rejected"],
                "n_correct": pair["n_correct"],
                "n_wrong":   pair["n_wrong"],
                "n_total":   pair["n_total"],
            }

            f_raw.write(json.dumps(record) + "\n")
            f_raw.flush()
            seen_keys.add(key)
            pair_count += 1

            if pair_count % 50 == 0:
                log.info(f"  ✅ Saved {pair_count} pairs so far")

    log.info(f"\n{'='*60}")
    log.info(f"Generation complete: {pair_count} total pairs")
    log.info(f"Skipped (no valid pair): {skipped_no_pair}")
    log.info(f"Skipped (already seen): {skipped_seen}")

    # ── Post-process: build clean train file ──────────────────────────────────
    log.info("\nBuilding filtered train file...")
    pairs_out = []

    with open(RAW_FILE) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("status") != "pair":
                continue

            chosen   = obj["chosen"]
            rejected = obj["rejected"]

            # Quality filters for training
            think_words_chosen = count_think_words(chosen)

            # Must have solution tag on chosen side
            if "</solution>" not in chosen.lower():
                continue

            # Chosen think length sanity
            if not (MIN_THINK_WORDS <= think_words_chosen <= MAX_THINK_WORDS):
                continue

            # Rejected must be meaningfully different
            if chosen.strip() == rejected.strip():
                continue

            # Prioritize pairs where model is uncertain (2-6 correct)
            # — these have the best gradient signal
            signal_quality = "strong" if 2 <= obj["n_correct"] <= 6 else "weak"

            pairs_out.append({
                "prompt":   obj["prompt"],
                "chosen":   chosen,
                "rejected": rejected,
                "source":   obj["source"],
                "signal":   signal_quality,
                "n_correct": obj["n_correct"],
                "n_total":   obj["n_total"],
            })

    # Sort by signal quality (strong first) then shuffle within groups
    strong = [p for p in pairs_out if p["signal"] == "strong"]
    weak   = [p for p in pairs_out if p["signal"] == "weak"]
    random.shuffle(strong)
    random.shuffle(weak)
    final = strong + weak  # strong signal pairs come first in training

    with open(TRAIN_FILE, "w") as f:
        for p in final:
            f.write(json.dumps(p) + "\n")

    # Stats breakdown
    source_counts = {}
    for p in final:
        source_counts[p["source"]] = source_counts.get(p["source"], 0) + 1

    log.info(f"\n{'='*60}")
    log.info(f"Train file: {TRAIN_FILE}")
    log.info(f"Total DPO pairs: {len(final)}")
    log.info(f"  Strong signal (2-6/8 correct): {len(strong)}")
    log.info(f"  Weak signal:                   {len(weak)}")
    log.info("Source breakdown:")
    for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
        log.info(f"  {src:<20}: {cnt:>5}")

    if len(final) >= MIN_PAIRS_TARGET:
        log.info(f"\n✅ Target of {MIN_PAIRS_TARGET} pairs reached!")
    else:
        log.warning(f"\n⚠️  Only {len(final)} pairs — consider adding more dataset sources")

    log.info(f"\nReady for stage7_dpo_train.py → {TRAIN_FILE}")


if __name__ == "__main__":
    main()
