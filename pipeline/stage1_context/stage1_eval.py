"""
Stage 1 Evaluation Script
─────────────────────────
Server : revanth@172.16.192.168
Path   : ~/nlp/scripts/stage1_context/stage1_eval.py

Runs two checks:
  1. Perplexity at 4K / 8K / 16K context  (wikitext-2, fast)
  2. Baseline GSM8K accuracy               (50 problems, fast)

Usage:
  python3 stage1_eval.py
  python3 stage1_eval.py --model Qwen/Qwen2.5-1.5B --gsm-samples 100
  python3 stage1_eval.py --skip-perplexity     # only run GSM8K
  python3 stage1_eval.py --skip-gsm            # only run perplexity
"""

import argparse
import re
import math
import json
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─── CLI args ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",           type=str, default="Qwen/Qwen2.5-1.5B",
                   help="HuggingFace model id or local path")
    p.add_argument("--attn-impl",       type=str, default="auto",
                   choices=["auto", "flash_attention_2", "sdpa", "eager"],
                   help="Attention backend. 'auto' tries flash_attn then falls back to sdpa.")
    p.add_argument("--gsm-samples",     type=int, default=50,
                   help="Number of GSM8K test problems to evaluate (max 1319)")
    p.add_argument("--ppl-tokens",      type=int, default=8000,
                   help="Total tokens to use for perplexity eval (keep low for speed)")
    p.add_argument("--lengths",         type=int, nargs="+", default=[4096, 8192, 16384],
                   help="Context lengths to test perplexity at")
    p.add_argument("--skip-perplexity", action="store_true")
    p.add_argument("--skip-gsm",        action="store_true")
    p.add_argument("--save-results",    type=str, default="~/nlp/logs/stage1_eval_results.json",
                   help="Where to save the JSON results")
    return p.parse_args()


# ─── Attention backend ───────────────────────────────────────────────────────

def resolve_attn_impl(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import flash_attn  # noqa
        print("✅ flash_attn found — using flash_attention_2")
        return "flash_attention_2"
    except ImportError:
        print("⚠️  flash_attn not installed — falling back to sdpa")
        return "sdpa"


# ─── Model loading ───────────────────────────────────────────────────────────

def load_model(model_name: str, attn_impl: str):
    print(f"\n{'─'*60}")
    print(f"Loading model : {model_name}")
    print(f"Attn backend  : {attn_impl}")
    print(f"{'─'*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Print key config info
    cfg = model.config
    print(f"\nModel config:")
    print(f"  max_position_embeddings : {cfg.max_position_embeddings}")
    print(f"  hidden_size             : {cfg.hidden_size}")
    print(f"  num_hidden_layers       : {cfg.num_hidden_layers}")
    print(f"  num_attention_heads     : {cfg.num_attention_heads}")
    kv_heads = getattr(cfg, "num_key_value_heads", "n/a")
    print(f"  num_key_value_heads     : {kv_heads}  (GQA)")
    rope = getattr(cfg, "rope_scaling", None)
    print(f"  rope_scaling            : {rope}")
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  parameters              : {total_params:.2f}B")

    device = next(model.parameters()).device
    print(f"  device                  : {device}")

    return model, tokenizer


# ─── 1. Perplexity evaluation ────────────────────────────────────────────────

def eval_perplexity(model, tokenizer, lengths, total_tokens):
    """
    Uses wikitext-2 test split.
    For each context length, chunks the text into blocks of that length
    and computes average perplexity. Ideally perplexity should be stable
    (within ~15%) across all tested lengths.
    """
    print(f"\n{'═'*60}")
    print("PERPLEXITY EVALUATION  (wikitext-2-raw-v1 / test)")
    print(f"{'═'*60}")

    print("Downloading wikitext-2 test split...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Concatenate all text, tokenize once
    full_text = "\n\n".join([ex["text"] for ex in dataset if ex["text"].strip()])
    print(f"Total raw characters: {len(full_text):,}")

    tokens = tokenizer(full_text, return_tensors="pt", truncation=False)["input_ids"][0]
    tokens = tokens[:total_tokens]
    print(f"Tokens being evaluated: {len(tokens):,}")

    results = {}
    device = next(model.parameters()).device

    for ctx_len in lengths:
        if ctx_len > model.config.max_position_embeddings:
            print(f"\n⚠️  Skipping {ctx_len//1024}K — exceeds model max_position_embeddings")
            continue

        print(f"\nEvaluating at context length {ctx_len//1024}K ({ctx_len} tokens)...")

        # Build non-overlapping chunks of size ctx_len
        chunks = [tokens[i : i + ctx_len] for i in range(0, len(tokens) - ctx_len, ctx_len)]
        if not chunks:
            # Not enough tokens for this length — use what we have
            chunks = [tokens]

        total_loss = 0.0
        total_count = 0

        for i, chunk in enumerate(chunks[:5]):  # cap at 5 chunks for speed
            input_ids = chunk.unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_ids, labels=input_ids)
            loss = output.loss.item()
            total_loss += loss
            total_count += 1
            print(f"  chunk {i+1}/{min(len(chunks), 5)} — loss: {loss:.4f}")

        avg_loss = total_loss / total_count
        ppl = math.exp(avg_loss)
        results[ctx_len] = {"loss": round(avg_loss, 4), "perplexity": round(ppl, 2)}
        print(f"  → avg loss: {avg_loss:.4f} | perplexity: {ppl:.2f}")

    # Summary
    print(f"\n{'─'*40}")
    print("Perplexity summary:")
    base_ppl = None
    for ctx_len in sorted(results):
        ppl = results[ctx_len]["perplexity"]
        if base_ppl is None:
            base_ppl = ppl
            ratio = 1.0
        else:
            ratio = ppl / base_ppl
        flag = "✅" if ratio < 1.15 else "⚠️ "
        print(f"  {ctx_len//1024:>3}K → PPL {ppl:>7.2f}   ratio vs {sorted(results)[0]//1024}K: {ratio:.2f}  {flag}")

    if base_ppl:
        max_ratio = max(r["perplexity"] for r in results.values()) / base_ppl
        if max_ratio < 1.15:
            verdict = "✅ PASS — long-context is stable. Proceed to Stage 2."
        elif max_ratio < 1.30:
            verdict = "⚠️  MARGINAL — minor degradation. Acceptable, proceed to Stage 2."
        else:
            verdict = "❌ FAIL — significant perplexity degradation. Run context extension training."
        print(f"\nVerdict: {verdict}")

    return results


# ─── 2. GSM8K baseline evaluation ────────────────────────────────────────────

def extract_answer(text: str) -> str | None:
    """
    Extract the final numeric answer from model output.
    Tries #### pattern first (GSM8K format), then last number in text.
    """
    # GSM8K ground truth uses #### N
    match = re.search(r"####\s*([\d,\-]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()

    # Model might write "The answer is X" or just end with a number
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text.replace(",", ""))
    if numbers:
        return numbers[-1].strip()
    return None


def eval_gsm8k(model, tokenizer, n_samples: int):
    """
    Evaluates on the first n_samples problems from GSM8K test split.
    Uses greedy decoding (temperature=0). This is your Stage 1 baseline number.
    """
    print(f"\n{'═'*60}")
    print(f"GSM8K BASELINE EVALUATION  (greedy, {n_samples} problems)")
    print(f"{'═'*60}")

    print("Downloading GSM8K test split...")
    dataset = load_dataset("gsm8k", "main", split="test")
    dataset = dataset.select(range(min(n_samples, len(dataset))))
    print(f"Evaluating on {len(dataset)} problems...")

    device = next(model.parameters()).device
    correct = 0
    results_detail = []

    for i, item in enumerate(dataset):
        prompt = (
            "Solve the following math problem step by step.\n\n"
            f"Problem: {item['question']}\n\n"
            "Solution:"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,        # greedy
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Extract answers
        predicted = extract_answer(response)
        ground_truth = extract_answer(item["answer"])

        is_correct = (predicted is not None and
                      ground_truth is not None and
                      predicted == ground_truth)
        if is_correct:
            correct += 1

        results_detail.append({
            "problem": item["question"][:80] + "...",
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": is_correct,
        })

        # Print progress every 10 problems
        if (i + 1) % 10 == 0 or (i + 1) == len(dataset):
            acc = correct / (i + 1) * 100
            print(f"  [{i+1:>3}/{len(dataset)}] running accuracy: {acc:.1f}%")

    final_acc = correct / len(dataset) * 100
    print(f"\n{'─'*40}")
    print(f"GSM8K result  : {correct}/{len(dataset)} = {final_acc:.1f}%")
    print(f"Expected range: 60–70% for Qwen2.5-1.5B base (no SFT)")
    print(f"\nThis is your Stage 1 BASELINE — record this number.")
    print(f"After Stage 2 SFT you should see this jump to ~85%.")

    return {
        "accuracy": round(final_acc, 2),
        "correct": correct,
        "total": len(dataset),
        "detail": results_detail,
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    attn_impl = resolve_attn_impl(args.attn_impl)

    print("\n" + "█" * 60)
    print("  STAGE 1 EVALUATION — MathReason-1.5B")
    print("█" * 60)
    print(f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model     : {args.model}")

    # Load model once, reuse for both evals
    model, tokenizer = load_model(args.model, attn_impl)

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "attn_impl": attn_impl,
    }

    # ── Perplexity ──
    if not args.skip_perplexity:
        ppl_results = eval_perplexity(
            model, tokenizer,
            lengths=args.lengths,
            total_tokens=args.ppl_tokens,
        )
        all_results["perplexity"] = ppl_results
    else:
        print("\n⏭  Skipping perplexity eval (--skip-perplexity)")

    # ── GSM8K ──
    if not args.skip_gsm:
        gsm_results = eval_gsm8k(model, tokenizer, n_samples=args.gsm_samples)
        all_results["gsm8k"] = {
            "accuracy": gsm_results["accuracy"],
            "correct": gsm_results["correct"],
            "total": gsm_results["total"],
        }
    else:
        print("\n⏭  Skipping GSM8K eval (--skip-gsm)")

    # ── Save results ──
    save_path = Path(args.save_results).expanduser()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        # Don't save per-problem detail to keep file small
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved to: {save_path}")

    # ── Final summary ──
    print(f"\n{'█'*60}")
    print("  STAGE 1 SUMMARY")
    print(f"{'█'*60}")
    if "perplexity" in all_results:
        for ctx, vals in sorted(all_results["perplexity"].items()):
            print(f"  PPL @ {int(ctx)//1024}K  : {vals['perplexity']}")
    if "gsm8k" in all_results:
        g = all_results["gsm8k"]
        print(f"  GSM8K       : {g['correct']}/{g['total']} = {g['accuracy']}%")
    print(f"{'█'*60}")
    print("\nIf perplexity is stable → proceed to Stage 2 SFT.")
    print("If GSM8K is 60–70%     → expected for base model, no action needed.")


if __name__ == "__main__":
    main()
