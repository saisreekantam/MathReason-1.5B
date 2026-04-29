"""
eval_prm_best_of_n.py  —  Stage 8C: PRM-Guided Best-of-N Eval on MATH500
══════════════════════════════════════════════════════════════════════════════
Server  : revanth@172.16.192.168
GPU 0   : PRM (Qwen2.5-0.5B + scalar head)
GPU 1   : Generator (stage7_dpo_merged)  ← primary student model

Strategy:
  For each MATH500 problem:
    1. Generate N=8 solutions with the student model (temperature=0.7)
    2. For each solution, split into steps and score each step with PRM
    3. Solution score = min step score  (weakest-link principle)
    4. Select solution with highest min-step-score

Baselines computed in same run:
  - Greedy@1      (single greedy decode)
  - Maj@8         (majority vote, no PRM)
  - PRM Best-of-N (this approach)

Expected gains over Maj@8: +8-12% MATH500 per Math-Shepherd results.

Usage:
  # Both GPUs allocated:
  tmux new -s eval_prm
  conda activate nlp
  CUDA_VISIBLE_DEVICES=0,1 python eval_prm_best_of_n.py

  # Sanity (50 problems):
  CUDA_VISIBLE_DEVICES=0,1 python eval_prm_best_of_n.py --sanity

  # Custom N:
  CUDA_VISIBLE_DEVICES=0,1 python eval_prm_best_of_n.py --n-solutions 16
══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── Paths ────────────────────────────────────────────────────────────────────

WORK_DIR   = Path("~/nlp").expanduser()
GEN_MODEL  = WORK_DIR / "checkpoints" / "stage7_dpo_merged"
GEN_FALLBACK = WORK_DIR / "checkpoints" / "stage4d_gdpo_merged"
PRM_DIR    = WORK_DIR / "checkpoints" / "prm_merged"
RESULTS    = WORK_DIR / "evals" / "prm_best_of_n_gsm8k.json"

SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Think step by step inside <think> tags, "
    "then give your final answer inside <solution> tags."
)

# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="PRM Best-of-N eval on MATH500")
    p.add_argument("--sanity",      action="store_true",
                   help="50 problems — quick test")
    p.add_argument("--n-solutions", type=int,   default=8,
                   help="Solutions to generate per problem")
    p.add_argument("--max-new",     type=int,   default=2048,
                   help="Max new tokens per solution (2048 avoids cutoff)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--gen-model",   type=str,   default="auto")
    p.add_argument("--prm-dir",     type=str,   default=str(PRM_DIR))
    p.add_argument("--resume",      action="store_true",
                   help="Skip already-evaluated problems")
    return p.parse_args()

# ─── Answer extraction ─────────────────────────────────────────────────────────

_SOLUTION_RE = re.compile(r"<solution>(.*?)</solution>", re.DOTALL | re.IGNORECASE)
_BOX_RE      = re.compile(r"\\boxed\{([^}]+)\}")
_NUMBER_RE   = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?")

def extract_answer(text: str) -> Optional[str]:
    """Smart extractor: scan first 60% for <solution> tag, then fallback."""
    cutoff     = max(len(text) * 60 // 100, 200)
    search_zone = text[:cutoff]

    m = _SOLUTION_RE.search(search_zone)
    if m:
        return m.group(1).strip()

    # Try \boxed{} (common in MATH dataset)
    m = _BOX_RE.search(text)
    if m:
        return m.group(1).strip()

    nums = _NUMBER_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None

def normalize(s: str) -> str:
    s = s.strip().lower().replace(",", "").replace(" ", "")
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return f"{f:.6g}"
    except ValueError:
        return s

def verify(pred: Optional[str], gt: str) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(normalize(pred)) - float(normalize(gt))) < 1e-4
    except ValueError:
        return normalize(pred) == normalize(gt)

# ─── Step extraction (same as generator script) ────────────────────────────────

def extract_think_block(text: str) -> str:
    m = re.search(r"<think>(.*?)(?:</think>|$)", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()

def split_into_steps(think_text: str) -> List[str]:
    raw = [s.strip() for s in re.split(r"\n\n+", think_text)]
    return [s for s in raw if len(s) >= 20] or [think_text.strip()]

def build_prefixes(steps: List[str]) -> List[str]:
    prefixes, accumulated = [], ""
    for step in steps:
        accumulated = (accumulated + "\n\n" + step).strip()
        prefixes.append(accumulated)
    return prefixes

# ─── PRM model ────────────────────────────────────────────────────────────────

class ProcessRewardModel(nn.Module):
    def __init__(self, base_model, hidden_size: int):
        super().__init__()
        self.base = base_model
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden   = outputs.hidden_states[-1]
        seq_lens = attention_mask.sum(dim=1) - 1
        last_hidden = hidden[
            torch.arange(hidden.shape[0], device=hidden.device),
            seq_lens,
        ]
        return self.head(last_hidden.float()).squeeze(-1)  # cast bf16→fp32 for head

def load_prm(prm_dir: Path, device: torch.device) -> Tuple:
    """Load merged PRM base + scalar head + tokenizer."""
    config_path = prm_dir / "prm_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"PRM config not found at {config_path}\n"
            "Run train_prm.py first."
        )
    with open(config_path) as f:
        cfg = json.load(f)

    print(f"  Loading PRM base from {prm_dir} ...")
    tok = AutoTokenizer.from_pretrained(str(prm_dir), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        str(prm_dir),
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to(device)

    model = ProcessRewardModel(base, cfg["hidden_size"]).to(device)
    head_path = prm_dir / cfg["head_path"]
    model.head.load_state_dict(torch.load(head_path, map_location=device))
    model.eval()

    print(f"  PRM loaded  (best_val_loss={cfg.get('best_val_loss', '?'):.4f}  "
          f"val_acc={cfg.get('val_acc', '?'):.3f})")
    return model, tok

# ─── Generator ────────────────────────────────────────────────────────────────

def resolve_gen_model(model_arg: str) -> Path:
    if model_arg != "auto":
        p = Path(model_arg)
        if not p.exists():
            raise FileNotFoundError(f"Generator model not found: {p}")
        return p
    if GEN_MODEL.exists():
        print(f"  Generator: {GEN_MODEL}  (stage7_dpo_merged ✅)")
        return GEN_MODEL
    if GEN_FALLBACK.exists():
        print(f"  Generator: {GEN_FALLBACK}  (fallback)")
        return GEN_FALLBACK
    raise FileNotFoundError("No generator model found.")

def load_generator(model_path: Path, device: torch.device) -> Tuple:
    print(f"  Loading generator from {model_path} ...")
    tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return model, tok

def build_prompt(tokenizer, problem: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": problem},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|user|>\n{problem}\n"
            f"<|assistant|>\n"
        )

@torch.inference_mode()
def generate_solutions(
    gen_model, gen_tok, problem: str,
    n: int, max_new: int, temperature: float,
    greedy_device: torch.device,
) -> Tuple[str, List[str]]:
    """Return (greedy_solution, [n sampled solutions])."""
    prompt   = build_prompt(gen_tok, problem)
    enc      = gen_tok(prompt, return_tensors="pt").to(greedy_device)
    inp_len  = enc["input_ids"].shape[1]

    # Greedy (temp=0)
    out_g = gen_model.generate(
        **enc, max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=gen_tok.eos_token_id,
    )
    greedy = gen_tok.decode(out_g[0][inp_len:], skip_special_tokens=True)

    # Sampled
    out_s = gen_model.generate(
        **enc, max_new_tokens=max_new,
        do_sample=True, temperature=temperature, top_p=0.95,
        num_return_sequences=n,
        pad_token_id=gen_tok.eos_token_id,
    )
    sampled = [
        gen_tok.decode(out_s[i][inp_len:], skip_special_tokens=True)
        for i in range(n)
    ]
    return greedy, sampled

# ─── PRM scoring ──────────────────────────────────────────────────────────────

@torch.inference_mode()
def score_solution_with_prm(
    prm_model, prm_tok, problem: str, solution: str,
    prm_device: torch.device, max_len: int = 1024,
) -> float:
    """
    Score a solution via PRM.
    Returns min step score (weakest-link = overall solution quality).
    Returns 0.0 if solution has no parseable steps.
    """
    think   = extract_think_block(solution)
    steps   = split_into_steps(think)
    prefixes = build_prefixes(steps)

    if not prefixes:
        return 0.0

    step_scores = []
    for prefix in prefixes:
        text = problem.strip() + "\n\n" + prefix.strip()
        enc  = prm_tok(
            text,
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        ).to(prm_device)

        score = prm_model(
            enc["input_ids"], enc["attention_mask"]
        ).item()
        step_scores.append(score)

    return min(step_scores)   # weakest-link

# ─── Maj@N voting ────────────────────────────────────────────────────────────

def majority_vote(solutions: List[str]) -> Optional[str]:
    answers = [extract_answer(s) for s in solutions]
    answers = [normalize(a) for a in answers if a is not None]
    if not answers:
        return None
    counter = Counter(answers)
    return counter.most_common(1)[0][0]

# ─── MATH500 loader ───────────────────────────────────────────────────────────

def load_gsm8k(cache_dir: Path) -> List[dict]:
    from datasets import load_dataset
    cache_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("openai/gsm8k", "main", split="test",
                      cache_dir=str(cache_dir))
    problems = []
    for r in ds:
        # GSM8K stores answer as "... #### <number>"
        m = re.search(r"####\s*([\d,]+)", r["answer"])
        if m:
            problems.append({
                "problem": r["question"],
                "answer":  m.group(1).replace(",", ""),
            })
    return problems

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.sanity:
        print("*** SANITY MODE: 50 problems ***\n")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)

    # ── Device allocation ──
    # GPU 0 → PRM (small, 0.5B)
    # GPU 1 → Generator (1.5B student)
    n_gpus = torch.cuda.device_count()
    if n_gpus >= 2:
        prm_device = torch.device("cuda:0")
        gen_device = torch.device("cuda:1")
        print(f"  Dual GPU: PRM→cuda:0  Generator→cuda:1")
    elif n_gpus == 1:
        prm_device = gen_device = torch.device("cuda:0")
        print(f"  Single GPU mode (both models on cuda:0)")
    else:
        prm_device = gen_device = torch.device("cpu")
        print("  WARNING: No GPU found — using CPU (very slow)")

    # ── Load models ──
    gen_path = resolve_gen_model(args.gen_model)
    gen_model, gen_tok = load_generator(gen_path, gen_device)

    prm_dir = Path(args.prm_dir)
    prm_model, prm_tok = load_prm(prm_dir, prm_device)

    # ── Load MATH500 ──
    print("\n  Loading GSM8K test set ...")
    cache_dir = WORK_DIR / "data" / "cache"
    problems  = load_gsm8k(cache_dir)
    if args.sanity:
        problems = problems[:50]
    print(f"  Problems: {len(problems)}")

    # ── Load prior results for resume ──
    done_idx  = set()
    all_results = []
    if args.resume and RESULTS.exists():
        with open(RESULTS) as f:
            all_results = json.load(f).get("per_problem", [])
        done_idx = {r["idx"] for r in all_results}
        print(f"  Resume: {len(done_idx)} already done.")

    # ── Counters ──
    greedy_correct = 0
    maj8_correct   = 0
    prm_bon_correct = 0

    # Recount from already-done
    for r in all_results:
        greedy_correct  += r["greedy_correct"]
        maj8_correct    += r["maj8_correct"]
        prm_bon_correct += r["prm_bon_correct"]

    print(f"\n  Starting eval  |  N={args.n_solutions}  temp={args.temperature}  "
          f"max_new={args.max_new}\n")

    for idx, item in enumerate(problems):
        if idx in done_idx:
            continue

        problem = item["problem"]
        gt      = item["answer"]

        # ── Generate solutions ──
        greedy, sampled = generate_solutions(
            gen_model, gen_tok, problem,
            n=args.n_solutions, max_new=args.max_new,
            temperature=args.temperature, greedy_device=gen_device,
        )

        # ── Greedy@1 ──
        g_pred  = extract_answer(greedy)
        g_corr  = int(verify(g_pred, gt))

        # ── Maj@N ──
        maj_pred = majority_vote(sampled)
        m_corr   = int(verify(maj_pred, gt))

        # ── PRM Best-of-N ──
        scores = [
            score_solution_with_prm(
                prm_model, prm_tok, problem, sol,
                prm_device=prm_device,
            )
            for sol in sampled
        ]
        best_idx  = max(range(len(scores)), key=lambda i: scores[i])
        prm_pred  = extract_answer(sampled[best_idx])
        p_corr    = int(verify(prm_pred, gt))

        greedy_correct  += g_corr
        maj8_correct    += m_corr
        prm_bon_correct += p_corr

        n_done = len(done_idx) + idx + 1 - len([i for i in done_idx if i <= idx])

        # Log
        result_entry = {
            "idx":             idx,
            "greedy_correct":  g_corr,
            "maj8_correct":    m_corr,
            "prm_bon_correct": p_corr,
            "prm_scores":      [round(s, 3) for s in scores],
            "best_prm_score":  round(scores[best_idx], 3),
        }
        all_results.append(result_entry)

        total_done = sum(1 for r in all_results if r["idx"] <= idx)

        print(
            f"[{idx+1:>4}/{len(problems)}]  "
            f"gt={gt[:12]:12s}  "
            f"greedy={'✓' if g_corr else '✗'}  "
            f"maj8={'✓' if m_corr else '✗'}  "
            f"prm={'✓' if p_corr else '✗'}  "
            f"best_prm_score={scores[best_idx]:.3f}"
        )

        # Save incrementally every 10 problems
        if (idx + 1) % 10 == 0:
            n_eval = len(all_results)
            summary = {
                "n_evaluated":     n_eval,
                "greedy_acc":      round(greedy_correct / n_eval * 100, 2),
                "maj8_acc":        round(maj8_correct   / n_eval * 100, 2),
                "prm_bon_acc":     round(prm_bon_correct/ n_eval * 100, 2),
                "prm_gain_vs_maj8": round(
                    (prm_bon_correct - maj8_correct) / n_eval * 100, 2
                ),
            }
            output = {"summary": summary, "per_problem": all_results}
            with open(RESULTS, "w") as f:
                json.dump(output, f, indent=2)

            print(
                f"\n  ── [{idx+1}/{len(problems)}]  "
                f"Greedy={summary['greedy_acc']}%  "
                f"Maj@{args.n_solutions}={summary['maj8_acc']}%  "
                f"PRM_BoN={summary['prm_bon_acc']}%  "
                f"(Δ vs Maj@N = {summary['prm_gain_vs_maj8']:+.1f}%) ──\n"
            )

    # ── Final summary ──
    n_eval = len(all_results)
    final_summary = {
        "model":            str(gen_path),
        "prm":              str(prm_dir),
        "n_problems":       n_eval,
        "n_solutions":      args.n_solutions,
        "greedy_acc":       round(greedy_correct  / n_eval * 100, 2),
        "maj8_acc":         round(maj8_correct    / n_eval * 100, 2),
        "prm_bon_acc":      round(prm_bon_correct / n_eval * 100, 2),
        "prm_gain_vs_greedy": round((prm_bon_correct - greedy_correct) / n_eval * 100, 2),
        "prm_gain_vs_maj8":   round((prm_bon_correct - maj8_correct)   / n_eval * 100, 2),
    }
    output = {"summary": final_summary, "per_problem": all_results}
    with open(RESULTS, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "═" * 60)
    print("  ✅ PRM Best-of-N Eval Complete")
    print(f"     Problems   : {n_eval}")
    print(f"     Greedy@1   : {final_summary['greedy_acc']}%")
    print(f"     Maj@{args.n_solutions:<2}     : {final_summary['maj8_acc']}%")
    print(f"     PRM BoN    : {final_summary['prm_bon_acc']}%")
    print(f"     Δ vs Maj@N : {final_summary['prm_gain_vs_maj8']:+.1f}%")
    print(f"     Results    : {RESULTS}")
    print("═" * 60)

if __name__ == "__main__":
    main()
