# ══════════════════════════════════════════════════════════════════
# stage5b_generate_local.py
# Uses DeepSeek-R1-Distill-Qwen-7B on GPU 0 (fits in ~14GB)
# Same R1 style as Stage 3 — better consistency than QwQ-32B
#
# Run: CUDA_VISIBLE_DEVICES=0 python stage5b_generate_local.py
# ══════════════════════════════════════════════════════════════════
import json, re, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

WORK_DIR = Path("~/nlp").expanduser()
DATA_DIR = WORK_DIR / "data" / "gap_sft"
IN_FILE  = DATA_DIR / "cat_ABC_problems.jsonl"
OUT_FILE = DATA_DIR / "cat_ABC_traced.jsonl"

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Think step by step inside <think> tags, "
    "then give your final answer inside <solution> tags."
)

THINK_RANGE = {
    "A_consumption":    (80,  400),
    "B_state_tracking": (120, 500),
    "C_interpretation": (80,  350),
}

def extract_answer(text):
    if not text: return None
    # 1. explicit <solution> tag
    sol = re.search(r"<solution>(.*?)</solution>", text, re.DOTALL)
    if sol: return sol.group(1).strip()
    # 2. LaTeX boxed
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed: return boxed[-1].strip()
    # 3. conclusion patterns — look at last 600 chars
    tail = text[-600:]
    patterns = [
        r"(?:answer is|value is|result is|equals?)[^\d-]*(-?[\d,]+(?:\.\d+)?)",
        r"x\s*=\s*(-?[\d,]+(?:\.\d+)?)",
        r"n\s*=\s*(-?[\d,]+(?:\.\d+)?)",
        r"therefore[^\d-]*(-?[\d,]+(?:\.\d+)?)",
        r"so[^\d-]*(-?[\d,]+(?:\.\d+)?)",
        r"=\s*\\boxed\{([^}]+)\}",
    ]
    for pat in patterns:
        m = re.search(pat, tail, re.IGNORECASE)
        if m: return m.group(1).replace(",","")
    # 4. fallback: last number in tail
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", tail)
    return nums[-1].replace(",","") if nums else None

def normalize(s):
    if not s: return ""
    s = str(s).strip().replace(",","").replace("$","")
    try:
        f = float(s); return str(int(f)) if f==int(f) else f"{f:.2f}"
    except: return s.lower()

def correct(pred, gt):
    p,g = normalize(pred), normalize(gt)
    if p==g: return True
    try: return abs(float(p)-float(g)) < 0.5
    except: return False

def looping(text):
    w = text.split()
    if len(w) < 40: return False
    ng = [tuple(w[i:i+6]) for i in range(len(w)-6)]
    return (1-len(set(ng))/len(ng)) > 0.40

def to_solution_fmt(response):
    """
    R1-Distill chat template injects <think> into the prompt itself.
    So the response starts mid-think — no opening <think> tag.
    Response format: "{thinking...}\n</think>\n{answer}"
    """
    if "</think>" in response:
        # Normal case: thinking ended, answer follows
        parts        = response.split("</think>", 1)
        think        = parts[0].strip()
        answer_part  = parts[1].strip() if len(parts) > 1 else ""
    elif "<think>" in response:
        # Rare: model re-opened a think tag somehow
        think_m = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        think   = think_m.group(1).strip() if think_m else response.strip()
        answer_part = response
    else:
        # Truncated — no </think> found, treat all as thinking
        think       = response.strip()
        answer_part = response

    ans = extract_answer(answer_part) or extract_answer(response)
    if not ans: return None
    return f"<think>\n{think}\n</think>\n<solution>{ans}</solution>"

def main():
    # Load problems
    problems = [json.loads(l) for l in open(IN_FILE)]
    done = set()
    if OUT_FILE.exists():
        for l in open(OUT_FILE):
            try: done.add(json.loads(l)["messages"][1]["content"][:80])
            except: pass
    remaining = [p for p in problems if p["problem"][:80] not in done]

    print(f"Total: {len(problems)} | Done: {len(done)} | To do: {len(remaining)}")
    if not remaining:
        print("All done!"); return

    print(f"\nLoading {MODEL_ID} on GPU 0 ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},        # force GPU 0 only
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"Loaded. VRAM: ~14GB on GPU 0\n")

    out_f = open(OUT_FILE, "a")
    correct_n, failed_n = 0, 0
    cat_counts = {}
    t0 = time.time()

    print(f"Starting generation loop — {len(remaining)} problems ...\n", flush=True)
    for i, prob in enumerate(remaining, 1):
        problem  = prob["problem"]
        gt       = prob["ground_truth"]
        category = prob.get("category", "A_consumption")
        think_lo, think_hi = THINK_RANGE.get(category, (80,500))

        # Build prompt — R1-Distill uses standard chat template
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": problem},
        ]
        prompt  = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs  = tok(prompt, return_tensors="pt").to(model.device)

        best = None
        for attempt in range(3):
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens = 2048,
                    temperature    = 0.3 + attempt * 0.1,
                    top_p          = 0.9,
                    do_sample      = True,
                    pad_token_id   = tok.eos_token_id,
                )
            response = tok.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            formatted = to_solution_fmt(response)
            if not formatted: continue
            pred = extract_answer(formatted)
            if not correct(pred, gt): continue
            think_m = re.search(r"<think>(.*?)</think>", formatted, re.DOTALL)
            n_think = len(think_m.group(1).split()) if think_m else 0
            if looping(response): continue
            if n_think > think_hi * 1.5: continue
            best = {"formatted": formatted, "think_words": n_think}
            break

        if best:
            record = {
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": problem},
                    {"role": "assistant", "content": best["formatted"]},
                ],
                "ground_truth": gt,
                "category":     category,
                "source":       prob.get("source","metamath"),
                "think_words":  best["think_words"],
            }
            out_f.write(json.dumps(record)+"\n")
            out_f.flush()
            correct_n += 1
            cat_counts[category] = cat_counts.get(category,0) + 1
        else:
            failed_n += 1

        status = "✅" if best else "❌"
        think_w = best["think_words"] if best else 0
        print(f"  [{i:4d}/{len(remaining)}] {status} cat={category[:12]:<12} "
              f"think={think_w:3d}w  correct={correct_n}  failed={failed_n}",
              flush=True)

        if i % 50 == 0:
            elapsed = time.time()-t0
            eta = (len(remaining)-i)/max(i/elapsed,0.01)/60
            print(f"  --- ETA: {eta:.0f}min ---", flush=True)

    out_f.close()
    print(f"\n{'='*55}")
    print(f"  DONE — correct={correct_n} failed={failed_n}")
    for c,n in sorted(cat_counts.items()):
        print(f"    {c:<28}: {n}")
    print(f"  Next: CUDA_VISIBLE_DEVICES=1 python stage5c_gap_sft.py")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()
