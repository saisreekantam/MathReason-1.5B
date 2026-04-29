# ══════════════════════════════════════════════════════════════════
# stage5a_build_dataset.py
# Gap-fill SFT — Step 1: Filter source datasets, build problem lists
#
# Run: python stage5a_build_dataset.py
# Output:
#   ~/nlp/data/gap_sft/cat_DEF_r1_traces.jsonl   (R1 traces, ready)
#   ~/nlp/data/gap_sft/cat_ABC_problems.jsonl    (problems needing QwQ traces)
#   ~/nlp/data/gap_sft/stats.json
# ══════════════════════════════════════════════════════════════════
import json
import re
import random
from pathlib import Path
from collections import defaultdict

from datasets import load_dataset

WORK_DIR  = Path("~/nlp").expanduser()
OUT_DIR   = WORK_DIR / "data" / "gap_sft"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)

SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Think step by step inside <think> tags, "
    "then give your final answer inside <solution> tags."
)

# ── Category targets ──────────────────────────────────────────────
CAT_TARGETS = {"D": 600, "E": 700, "F": 400}
ABC_TARGETS = {"A": 1200, "B": 1500, "C": 600}

# ─────────────────────────────────────────────────────────────────
# CATEGORY D — floor / ceil / integer rounding (from OpenR1-Math-220k)
# ─────────────────────────────────────────────────────────────────
FLOOR_CEIL_KEYWORDS = [
    r"\bfloor\b", r"\bceil\b", r"\blfloor\b", r"\brfloor\b",
    r"\blceil\b", r"\brceil\b",
    r"greatest integer", r"integer part",
    r"how many (?:whole |complete )?(?:multiples|groups|times|batches)",
    r"exactly \d+ (?:days|weeks|months)",
    r"minimum number of", r"maximum number of",
    r"\bdivide.*evenly\b", r"\bdivisible\b",
    r"\\left\\lfloor", r"\\right\\rfloor",
]

# CATEGORY E — combinatorics / counting (from OpenR1-Math-220k)
COMBINATORICS_KEYWORDS = [
    r"\barrangement\b", r"\bpermutation\b", r"\bcombination\b",
    r"\bbinom\b", r"\\binom{", r"C\(\d+,\s*\d+\)",
    r"non-adjacent", r"not adjacent", r"no two.*adjacent",
    r"gaps? method", r"gaps? technique",
    r"how many ways", r"in how many ways",
    r"can be arranged", r"can be selected",
    r"choose \d+ from", r"\bselect \d+\b",
    r"circular arrangement", r"necklace",
]

# CATEGORY F — multi-value answers (from OpenR1-Math-220k)
def has_multi_value_answer(answer_str: str) -> bool:
    """True if boxed answer contains a set, pair, or 'and'."""
    if not answer_str:
        return False
    s = answer_str.strip()
    if re.search(r"\\boxed\{[^}]*(?:,|\\text\{ and \}|\\text\{and\}| and )[^}]*\}", s):
        return True
    if re.search(r"\\{.*,.*\\}", s):
        return True
    if re.search(r"\band\b", s, re.IGNORECASE) and re.search(r"\d", s):
        return True
    return False

def matches_any(text: str, patterns: list[str]) -> bool:
    text_lower = text.lower()
    for pat in patterns:
        if re.search(pat, text_lower):
            return True
    return False

def think_token_count(solution: str) -> int:
    m = re.search(r"<think>(.*?)</think>", solution, re.DOTALL)
    if m:
        return len(m.group(1).split())
    return len(solution.split())

def extract_r1_answer(solution: str) -> str | None:
    """Pull the boxed answer from an R1 trace."""
    boxes = re.findall(r"\\boxed\{([^}]*)\}", solution)
    if boxes:
        return boxes[-1].strip()
    sol_m = re.search(r"<solution>(.*?)</solution>", solution, re.DOTALL)
    if sol_m:
        return sol_m.group(1).strip()
    return None

def to_solution_tag(solution: str) -> str:
    """Convert R1's \\boxed{} inside <think> to proper <solution> tag."""
    # Keep the full think block
    think_m = re.search(r"<think>(.*?)</think>", solution, re.DOTALL)
    think_content = think_m.group(1).strip() if think_m else solution.strip()

    # Get the answer
    answer = extract_r1_answer(solution)
    if not answer:
        return None

    return f"<think>\n{think_content}\n</think>\n<solution>{answer}</solution>"

# ── Load OpenR1-Math-220k ─────────────────────────────────────────
print("Loading OpenR1-Math-220k …")
openr1 = load_dataset("open-r1/OpenR1-Math-220k", split="train")
print(f"  Total: {len(openr1):,}")

cat_D, cat_E, cat_F = [], [], []

print("Filtering for categories D, E, F …")
for ex in openr1:
    problem  = ex.get("problem", "") or ex.get("question", "") or ""
    solution = ex.get("solution", "") or ""
    answer   = ex.get("answer",   "") or ""

    # Think token range: 200–800 (keep traces short for surgical SFT)
    n_think = think_token_count(solution)
    if not (150 <= n_think <= 900):
        continue

    # Must convert cleanly to solution-tag format
    formatted = to_solution_tag(solution)
    if not formatted:
        continue

    record = {
        "problem":   problem,
        "solution":  formatted,
        "answer":    answer,
        "source":    "openr1",
    }

    if len(cat_D) < CAT_TARGETS["D"] * 3:  # over-collect, deduplicate later
        if matches_any(problem + " " + solution, FLOOR_CEIL_KEYWORDS):
            cat_D.append(record)

    if len(cat_E) < CAT_TARGETS["E"] * 3:
        if matches_any(problem + " " + solution, COMBINATORICS_KEYWORDS):
            cat_E.append(record)

    if len(cat_F) < CAT_TARGETS["F"] * 3:
        if has_multi_value_answer(answer) or has_multi_value_answer(solution):
            cat_F.append(record)

# Deduplicate by problem text and trim to target
def dedup_trim(items, target):
    seen = set()
    out  = []
    for x in items:
        key = x["problem"][:120]
        if key not in seen:
            seen.add(key)
            out.append(x)
    random.shuffle(out)
    return out[:target]

cat_D = dedup_trim(cat_D, CAT_TARGETS["D"])
cat_E = dedup_trim(cat_E, CAT_TARGETS["E"])
cat_F = dedup_trim(cat_F, CAT_TARGETS["F"])

print(f"  Cat D (floor/ceil)    : {len(cat_D)}")
print(f"  Cat E (combinatorics) : {len(cat_E)}")
print(f"  Cat F (multi-value)   : {len(cat_F)}")

# ── Save D/E/F as full training records ──────────────────────────
def make_training_record(rec):
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": rec["problem"]},
            {"role": "assistant", "content": rec["solution"]},
        ],
        "answer":    rec["answer"],
        "category":  rec.get("category", "DEF"),
        "source":    rec["source"],
    }

def_records = []
for rec in cat_D:
    rec["category"] = "D_floor_ceil"
    def_records.append(make_training_record(rec))
for rec in cat_E:
    rec["category"] = "E_combinatorics"
    def_records.append(make_training_record(rec))
for rec in cat_F:
    rec["category"] = "F_multi_value"
    def_records.append(make_training_record(rec))

random.shuffle(def_records)
def_path = OUT_DIR / "cat_DEF_r1_traces.jsonl"
with open(def_path, "w") as f:
    for r in def_records:
        f.write(json.dumps(r) + "\n")
print(f"\nSaved {len(def_records)} D/E/F records → {def_path}")

# ─────────────────────────────────────────────────────────────────
# CATEGORIES A, B, C — source problems from MetaMathQA + GSM8K
# (traces will be generated by QwQ-32B in stage5b)
# ─────────────────────────────────────────────────────────────────

# CATEGORY A: consumption-language (bakes/uses/eats/gives = subtract)
CONSUMPTION_PATTERNS = [
    r"\b(?:eats?|ate)\b.*\b(?:egg|cookie|apple|fruit|pizza|slice)\b",
    r"\b(?:bakes?|uses?|uses up|used up)\b.*\b(?:egg|cup|gram|lb|oz)\b",
    r"\b(?:gives? away|gave away|donates?|donated)\b",
    r"\b(?:spends?|spent)\b.*\b(?:dollar|\$|money|hour|minute)\b",
    r"\b(?:sells?|sold)\b.*\b(?:remainder|rest|remaining|leftover)\b",
    r"she (?:eats?|bakes?|uses?|gives?|donates?|spends?)",
    r"he (?:eats?|bakes?|uses?|gives?|donates?|spends?)",
    r"they (?:eat|bake|use|give|donate|spend)",
    r"(?:consumed|depleted|used for|reserved for)",
]

# CATEGORY B: multi-step state tracking (restart, multi-segment)
STATE_TRACKING_PATTERNS = [
    r"restart(?:s|ed|ing)?.*(?:download|install|upload|from the beginning)",
    r"from the beginning", r"start(?:s|ed)? over",
    r"(?:first|second|third|then|after that|next).*(?:hour|minute|mile|km)",
    r"(?:traffic|jam|standstill|gridlock)",
    r"(?:segment|leg|stretch) of",
    r"returns? to.*(?:start|beginning|home|base)",
    r"how (?:long|far|much time).*(?:total|altogether|combined|overall)",
    r"remaining.*(?:time|distance|hours?|minutes?|miles?)",
    r"(?:leaves?|departs?|sets? off).*(?:arrives?|reaches?|gets? to)",
    r"two pipes?|three pipes?|pipe [ABC]",
    r"fills? at.*drains? at",
]

# CATEGORY C: interpretation — "each", "profit", "save", "gain"
INTERPRETATION_PATTERNS = [
    r"\beach\b.*(?:train|person|student|worker|car|animal|player)",
    r"\bprofit\b.*(?:made|earn|gain|net)",
    r"\bhow much.*(?:profit|gain|earn|save|saved)\b",
    r"\bmaximize.*profit\b",
    r"\bchoose.*(?:between|among).*(?:higher|better|more profitable)",
    r"\bwhich.*(?:option|choice|plan|investment).*(?:better|best|more)\b",
    r"\bper person\b|\bper student\b|\bper worker\b",
    r"\bhow many.*each\b|\bhow much.*each\b",
    r"\bsplit.*equally\b|\bdivide.*equally\b",
]

print("\nLoading MetaMathQA for A/B/C problem sourcing …")
metamath = load_dataset("meta-math/MetaMathQA", split="train")
print(f"  MetaMathQA total: {len(metamath):,}")

cat_A_probs, cat_B_probs, cat_C_probs = [], [], []

for ex in metamath:
    q = (ex.get("query") or ex.get("input") or "").strip()
    a = (ex.get("response") or ex.get("output") or "").strip()
    if not q or not a or len(q) < 40:
        continue

    # Prefer GSM8K-style (clear arithmetic, not overly symbolic)
    if "####" not in a and "The answer is" not in a:
        continue

    # Extract ground truth
    if "####" in a:
        gt = a.split("####")[-1].strip()
    else:
        m = re.search(r"[Tt]he answer is[:\s]+([\d,.$%]+)", a)
        gt = m.group(1).strip() if m else None
    if not gt:
        continue

    record = {"problem": q, "ground_truth": gt, "source": "metamath"}

    if len(cat_A_probs) < ABC_TARGETS["A"] * 2:
        if matches_any(q, CONSUMPTION_PATTERNS):
            record["category"] = "A_consumption"
            cat_A_probs.append(record)

    if len(cat_B_probs) < ABC_TARGETS["B"] * 2:
        if matches_any(q, STATE_TRACKING_PATTERNS):
            record["category"] = "B_state_tracking"
            cat_B_probs.append(record)

    if len(cat_C_probs) < ABC_TARGETS["C"] * 2:
        if matches_any(q, INTERPRETATION_PATTERNS):
            record["category"] = "C_interpretation"
            cat_C_probs.append(record)

# Also pull from GSM8K test (unseen problems = better eval coverage)
print("Loading GSM8K for additional A/B/C problems …")
gsm8k = load_dataset("openai/gsm8k", "main", split="train")
for ex in gsm8k:
    q = ex.get("question", "").strip()
    a = ex.get("answer",   "").strip()
    if not q or not a:
        continue
    gt_m = re.search(r"####\s*([\d,.$%-]+)", a)
    if not gt_m:
        continue
    gt = gt_m.group(1).strip()
    record = {"problem": q, "ground_truth": gt, "source": "gsm8k"}

    if len(cat_A_probs) < ABC_TARGETS["A"] * 2:
        if matches_any(q, CONSUMPTION_PATTERNS):
            record["category"] = "A_consumption"
            cat_A_probs.append(record)
    if len(cat_B_probs) < ABC_TARGETS["B"] * 2:
        if matches_any(q, STATE_TRACKING_PATTERNS):
            record["category"] = "B_state_tracking"
            cat_B_probs.append(record)
    if len(cat_C_probs) < ABC_TARGETS["C"] * 2:
        if matches_any(q, INTERPRETATION_PATTERNS):
            record["category"] = "C_interpretation"
            cat_C_probs.append(record)

cat_A_probs = dedup_trim(cat_A_probs, ABC_TARGETS["A"])
cat_B_probs = dedup_trim(cat_B_probs, ABC_TARGETS["B"])
cat_C_probs = dedup_trim(cat_C_probs, ABC_TARGETS["C"])

print(f"  Cat A (consumption)   : {len(cat_A_probs)} problems to trace")
print(f"  Cat B (state track)   : {len(cat_B_probs)} problems to trace")
print(f"  Cat C (interpretation): {len(cat_C_probs)} problems to trace")

abc_problems = cat_A_probs + cat_B_probs + cat_C_probs
random.shuffle(abc_problems)
abc_path = OUT_DIR / "cat_ABC_problems.jsonl"
with open(abc_path, "w") as f:
    for r in abc_problems:
        f.write(json.dumps(r) + "\n")
print(f"\nSaved {len(abc_problems)} A/B/C problems → {abc_path}")

# ── Summary ───────────────────────────────────────────────────────
stats = {
    "cat_D": len(cat_D), "cat_E": len(cat_E), "cat_F": len(cat_F),
    "cat_A_problems": len(cat_A_probs),
    "cat_B_problems": len(cat_B_probs),
    "cat_C_problems": len(cat_C_probs),
    "def_ready": len(def_records),
    "abc_need_traces": len(abc_problems),
}
with open(OUT_DIR / "stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("\n── NEXT STEP ──────────────────────────────────────────────")
print(f"  Run: CUDA_VISIBLE_DEVICES=0,1 python stage5b_generate_traces.py")
print(f"  This will generate QwQ-32B traces for the {len(abc_problems)} A/B/C problems.")
