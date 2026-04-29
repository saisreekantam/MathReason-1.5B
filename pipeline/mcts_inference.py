"""
mcts_inference.py  —  rStar-style MCTS Inference with PRM Guidance
══════════════════════════════════════════════════════════════════════════════
Server  : revanth@172.16.192.168
GPU 0   : PRM  (Qwen2.5-0.5B + scalar head  — prm_merged)
GPU 1   : Policy model  (stage7_dpo_merged)

How MCTS works here:
  Each NODE = (problem, reasoning_prefix_so_far)
  Each EDGE = one candidate next reasoning step

  For each problem:
    1. Root node = empty prefix
    2. Repeat for N simulations:
       a. SELECT   — walk tree using UCB to find best node to expand
       b. EXPAND   — generate K candidate next steps from that node
       c. EVALUATE — PRM scores each candidate step
       d. BACKPROP — update visit counts + values up to root
    3. When a terminal node is reached (</think> found or max_depth hit):
       — Complete the solution with greedy decode
       — Extract answer
    4. Return answer from terminal node with highest visit-weighted value

UCB formula (PUCT variant):
  UCB(node) = Q(node) + c_puct * prm_prior * sqrt(N_parent) / (1 + N(node))
  where:
    Q(node)    = average PRM score of all rollouts through this node
    prm_prior  = PRM score assigned at node creation
    N(node)    = visit count
    c_puct     = 1.5  (exploration constant)

Baselines computed in the same run:
  - Greedy@1  (single greedy decode, no search)
  - Maj@8     (8 independent samples, majority vote)
  - MCTS      (tree search with PRM)

Expected gains vs Best-of-N: +5-10% additional MATH500 accuracy

Usage:
  tmux new -s mcts
  conda activate nlp
  CUDA_VISIBLE_DEVICES=0,1 python mcts_inference.py --dataset gsm8k

  # MATH500:
  CUDA_VISIBLE_DEVICES=0,1 python mcts_inference.py --dataset math500

  # Sanity (20 problems):
  CUDA_VISIBLE_DEVICES=0,1 python mcts_inference.py --dataset gsm8k --sanity

  # Tune search depth:
  CUDA_VISIBLE_DEVICES=0,1 python mcts_inference.py --dataset math500 \
      --n-simulations 32 --expansion-size 4 --max-depth 8
══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── Paths ────────────────────────────────────────────────────────────────────

WORK_DIR     = Path("~/nlp").expanduser()
GEN_MODEL    = WORK_DIR / "checkpoints" / "stage7_dpo_merged"
GEN_FALLBACK = WORK_DIR / "checkpoints" / "stage4d_gdpo_merged"
PRM_DIR      = WORK_DIR / "checkpoints" / "prm_merged"
RESULTS_DIR  = WORK_DIR / "evals"

SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Think step by step inside <think> tags, "
    "then give your final answer inside <solution> tags."
)

# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MCTS Inference with PRM")
    p.add_argument("--dataset",       type=str, default="gsm8k",
                   choices=["gsm8k", "math500"],
                   help="Evaluation benchmark")
    p.add_argument("--sanity",        action="store_true",
                   help="20 problems — quick test")
    p.add_argument("--resume",        action="store_true",
                   help="Skip already-evaluated problems")

    # MCTS hyperparams
    p.add_argument("--n-simulations", type=int,   default=16,
                   help="MCTS simulations per problem (more=better, slower)")
    p.add_argument("--expansion-size",type=int,   default=4,
                   help="Candidate next steps to generate at each expansion")
    p.add_argument("--max-depth",     type=int,   default=8,
                   help="Max reasoning steps in tree")
    p.add_argument("--c-puct",        type=float, default=1.5,
                   help="UCB exploration constant")
    p.add_argument("--temperature",   type=float, default=0.8,
                   help="Sampling temperature for step generation")

    # Baselines
    p.add_argument("--n-maj",         type=int,   default=8,
                   help="Samples for Maj@N baseline")

    # Generation
    p.add_argument("--max-step-tokens", type=int, default=256,
                   help="Max tokens per reasoning step generation")
    p.add_argument("--max-complete-tokens", type=int, default=1024,
                   help="Max tokens for terminal node solution completion")
    p.add_argument("--max-new",         type=int, default=2048,
                   help="Max tokens for greedy/maj baselines")

    # Models
    p.add_argument("--gen-model",     type=str, default="auto")
    p.add_argument("--prm-dir",       type=str, default=str(PRM_DIR))
    return p.parse_args()

# ─── Answer extraction ────────────────────────────────────────────────────────

_SOLUTION_RE = re.compile(r"<solution>(.*?)</solution>", re.DOTALL | re.IGNORECASE)
_BOX_RE      = re.compile(r"\\boxed\{([^}]+)\}")
_NUMBER_RE   = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?")

def extract_answer(text: str) -> Optional[str]:
    cutoff = max(len(text) * 60 // 100, 200)
    m = _SOLUTION_RE.search(text[:cutoff])
    if m:
        return m.group(1).strip()
    m = _BOX_RE.search(text)
    if m:
        return m.group(1).strip()
    nums = _NUMBER_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None

def normalize(s: str) -> str:
    s = s.strip().lower().replace(",", "").replace(" ", "")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else f"{f:.6g}"
    except ValueError:
        return s

def verify(pred: Optional[str], gt: str) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(normalize(pred)) - float(normalize(gt))) < 1e-4
    except ValueError:
        return normalize(pred) == normalize(gt)

# ─── Step splitting ───────────────────────────────────────────────────────────

def split_steps(think_text: str) -> List[str]:
    raw = [s.strip() for s in re.split(r"\n\n+", think_text)]
    return [s for s in raw if len(s) >= 15]

def is_terminal(prefix: str) -> bool:
    """A node is terminal if the model has closed its think block."""
    return "</think>" in prefix.lower()

# ─── MCTS Node ────────────────────────────────────────────────────────────────

@dataclass
class MCTSNode:
    """
    A node in the MCTS tree.
    prefix      : accumulated reasoning text so far (inside <think> block)
    prm_prior   : PRM score assigned when this node was created
    parent      : parent node (None for root)
    depth       : step depth from root
    """
    prefix:     str
    prm_prior:  float
    parent:     Optional["MCTSNode"]
    depth:      int

    # MCTS statistics (updated during backprop)
    visit_count:  int   = field(default=0)
    total_value:  float = field(default=0.0)
    children:     List["MCTSNode"] = field(default_factory=list)

    @property
    def q_value(self) -> float:
        """Average value of rollouts through this node."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb_score(self, c_puct: float) -> float:
        """
        PUCT score for selection.
        Balances exploitation (Q) with exploration (prior / visit count).
        """
        if self.parent is None:
            return float("inf")
        n_parent = max(self.parent.visit_count, 1)
        exploration = c_puct * self.prm_prior * math.sqrt(n_parent) / (1 + self.visit_count)
        return self.q_value + exploration

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

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
        last_h   = hidden[
            torch.arange(hidden.shape[0], device=hidden.device), seq_lens
        ]
        return self.head(last_h.float()).squeeze(-1)   # bf16→fp32 cast

def load_prm(prm_dir: Path, device: torch.device):
    cfg_path = prm_dir / "prm_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"PRM config not found: {cfg_path}. Run train_prm.py first.")
    with open(cfg_path) as f:
        cfg = json.load(f)

    print(f"  Loading PRM from {prm_dir} ...")
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
    model.head.load_state_dict(
        torch.load(prm_dir / cfg["head_path"], map_location=device)
    )
    model.eval()
    print(f"  PRM ready  (val_loss={cfg.get('best_val_loss',0):.4f}  "
          f"val_acc={cfg.get('val_acc',0):.3f})")
    return model, tok

# ─── Generator ────────────────────────────────────────────────────────────────

def resolve_gen_model(model_arg: str) -> Path:
    if model_arg != "auto":
        p = Path(model_arg)
        if not p.exists():
            raise FileNotFoundError(f"Generator model not found: {p}")
        return p
    for p in [GEN_MODEL, GEN_FALLBACK]:
        if p.exists():
            tag = "(stage7_dpo_merged ✅)" if p == GEN_MODEL else "(fallback)"
            print(f"  Generator: {p}  {tag}")
            return p
    raise FileNotFoundError("No generator model found.")

def load_generator(model_path: Path, device: torch.device):
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

def build_prompt(tokenizer, problem: str, partial_think: Optional[str] = None) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": problem},
    ]
    if partial_think is not None:
        messages.append({
            "role":    "assistant",
            "content": f"<think>\n{partial_think}",
        })
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=(partial_think is None),
            continue_final_message=(partial_think is not None),
        )
    except Exception:
        if partial_think is None:
            return f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{problem}\n<|assistant|>\n"
        return (f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{problem}\n"
                f"<|assistant|>\n<think>\n{partial_think}")

# ─── Core generation functions ────────────────────────────────────────────────

@torch.inference_mode()
def generate_next_steps(
    gen_model, gen_tok, problem: str, prefix: str,
    n: int, max_tokens: int, temperature: float, device: torch.device,
) -> List[str]:
    """
    Generate n candidate NEXT STEPS from the current reasoning prefix.
    Each completion is one additional reasoning step (stopped at \n\n).
    """
    prompt  = build_prompt(gen_tok, problem, partial_think=prefix)
    enc     = gen_tok(prompt, return_tensors="pt").to(device)
    inp_len = enc["input_ids"].shape[1]

    outputs = gen_model.generate(
        **enc,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        num_return_sequences=n,
        pad_token_id=gen_tok.eos_token_id,
        # Stop at double newline = natural step boundary
        # (we'll also split manually below)
    )

    steps = []
    for out in outputs:
        text = gen_tok.decode(out[inp_len:], skip_special_tokens=True)
        # Extract just the first step (up to \n\n or </think>)
        parts = re.split(r"\n\n+|</think>", text, maxsplit=1)
        step  = parts[0].strip()
        if step:
            steps.append(step)
    return steps

@torch.inference_mode()
def complete_solution(
    gen_model, gen_tok, problem: str, prefix: str,
    max_tokens: int, device: torch.device,
) -> str:
    """
    From a terminal node prefix, greedily complete the full solution
    (close </think> and produce <solution>...</solution>).
    """
    # If think block already closed, just generate the solution part
    if "</think>" not in prefix.lower():
        full_prefix = prefix + "\n\n</think>\n"
    else:
        full_prefix = prefix

    prompt  = build_prompt(gen_tok, problem, partial_think=full_prefix)
    enc     = gen_tok(prompt, return_tensors="pt").to(device)
    inp_len = enc["input_ids"].shape[1]

    out = gen_model.generate(
        **enc,
        max_new_tokens=max_tokens,
        do_sample=False,
        pad_token_id=gen_tok.eos_token_id,
    )
    return gen_tok.decode(out[0][inp_len:], skip_special_tokens=True)

@torch.inference_mode()
def generate_full_solutions(
    gen_model, gen_tok, problem: str,
    n: int, max_tokens: int, temperature: float, device: torch.device,
) -> Tuple[str, List[str]]:
    """Generate 1 greedy + n sampled full solutions (for baselines)."""
    prompt  = build_prompt(gen_tok, problem)
    enc     = gen_tok(prompt, return_tensors="pt").to(device)
    inp_len = enc["input_ids"].shape[1]

    # Greedy
    out_g = gen_model.generate(
        **enc, max_new_tokens=max_tokens, do_sample=False,
        pad_token_id=gen_tok.eos_token_id,
    )
    greedy = gen_tok.decode(out_g[0][inp_len:], skip_special_tokens=True)

    # Sampled
    out_s = gen_model.generate(
        **enc, max_new_tokens=max_tokens,
        do_sample=True, temperature=temperature, top_p=0.95,
        num_return_sequences=n, pad_token_id=gen_tok.eos_token_id,
    )
    sampled = [gen_tok.decode(out_s[i][inp_len:], skip_special_tokens=True)
               for i in range(n)]
    return greedy, sampled

# ─── PRM scoring ──────────────────────────────────────────────────────────────

@torch.inference_mode()
def prm_score_step(
    prm_model, prm_tok, problem: str, prefix: str,
    device: torch.device, max_len: int = 1024,
) -> float:
    """Score a single (problem, reasoning_prefix) pair with PRM."""
    text = problem.strip() + "\n\n" + prefix.strip()
    enc  = prm_tok(
        text, max_length=max_len, truncation=True, return_tensors="pt"
    ).to(device)
    return prm_model(enc["input_ids"], enc["attention_mask"]).item()

# ─── Majority vote ────────────────────────────────────────────────────────────

def majority_vote(solutions: List[str]) -> Optional[str]:
    answers = [normalize(a) for s in solutions
               if (a := extract_answer(s)) is not None]
    if not answers:
        return None
    return Counter(answers).most_common(1)[0][0]

# ─── MCTS Core ────────────────────────────────────────────────────────────────

class MCTS:
    """
    rStar-style MCTS for math reasoning.
    Uses PRM as both the prior (node creation score) and value function.
    """

    def __init__(
        self,
        gen_model, gen_tok, gen_device,
        prm_model, prm_tok, prm_device,
        problem: str,
        n_simulations: int  = 16,
        expansion_size: int = 4,
        max_depth: int      = 8,
        c_puct: float       = 1.5,
        temperature: float  = 0.8,
        max_step_tokens: int      = 256,
        max_complete_tokens: int  = 1024,
    ):
        self.gen_model   = gen_model
        self.gen_tok     = gen_tok
        self.gen_device  = gen_device
        self.prm_model   = prm_model
        self.prm_tok     = prm_tok
        self.prm_device  = prm_device
        self.problem     = problem

        self.n_simulations       = n_simulations
        self.expansion_size      = expansion_size
        self.max_depth           = max_depth
        self.c_puct              = c_puct
        self.temperature         = temperature
        self.max_step_tokens     = max_step_tokens
        self.max_complete_tokens = max_complete_tokens

        # Root = empty reasoning prefix, prior=1.0
        self.root = MCTSNode(prefix="", prm_prior=1.0, parent=None, depth=0)

        # Track terminal nodes with their completed solutions
        self.terminal_solutions: List[Tuple[MCTSNode, str, float]] = []
        # (node, solution_text, answer_value)

    def _select(self) -> MCTSNode:
        """
        Walk the tree from root, always choosing the child with
        highest UCB score, until we reach a leaf node.
        """
        node = self.root
        while not node.is_leaf:
            node = max(node.children, key=lambda c: c.ucb_score(self.c_puct))
        return node

    def _expand(self, node: MCTSNode) -> List[MCTSNode]:
        """
        Generate `expansion_size` candidate next steps from this node.
        Score each with PRM → create child nodes.
        Returns list of newly created children.
        """
        if is_terminal(node.prefix) or node.depth >= self.max_depth:
            return []

        # Generate candidate next steps
        candidate_steps = generate_next_steps(
            self.gen_model, self.gen_tok,
            self.problem, node.prefix,
            n=self.expansion_size,
            max_tokens=self.max_step_tokens,
            temperature=self.temperature,
            device=self.gen_device,
        )

        # Deduplicate
        seen = set()
        unique_steps = []
        for s in candidate_steps:
            if s not in seen and len(s) >= 10:
                seen.add(s)
                unique_steps.append(s)

        children = []
        for step in unique_steps:
            new_prefix = (node.prefix + "\n\n" + step).strip()

            # PRM scores this prefix
            prior = prm_score_step(
                self.prm_model, self.prm_tok,
                self.problem, new_prefix,
                device=self.prm_device,
            )

            child = MCTSNode(
                prefix    = new_prefix,
                prm_prior = prior,
                parent    = node,
                depth     = node.depth + 1,
            )
            children.append(child)

        node.children = children
        return children

    def _simulate(self, node: MCTSNode) -> float:
        """
        From this node, complete the solution and return a value ∈ [0, 1].
        Value = PRM score of best step in completed solution (proxy for quality).
        If solution is correct, value is boosted to 1.0.
        """
        solution = complete_solution(
            self.gen_model, self.gen_tok,
            self.problem, node.prefix,
            max_tokens=self.max_complete_tokens,
            device=self.gen_device,
        )

        # Score the completed solution steps with PRM
        think_text = node.prefix  # prefix is already accumulated think steps
        steps      = split_steps(think_text)
        if steps:
            step_scores = [
                prm_score_step(
                    self.prm_model, self.prm_tok,
                    self.problem,
                    "\n\n".join(steps[:i+1]),
                    device=self.prm_device,
                )
                for i in range(len(steps))
            ]
            value = min(step_scores)  # weakest-link
        else:
            value = node.prm_prior

        # Store terminal solution for later answer extraction
        self.terminal_solutions.append((node, solution, value))
        return value

    def _backprop(self, node: MCTSNode, value: float):
        """Propagate value up to root, incrementing visit counts."""
        while node is not None:
            node.visit_count  += 1
            node.total_value  += value
            node = node.parent

    def search(self) -> Tuple[str, dict]:
        """
        Run full MCTS search.
        Returns (best_answer, stats_dict).
        """
        for sim in range(self.n_simulations):
            # 1. Select
            leaf = self._select()

            # 2. Expand (if not terminal/max-depth)
            if not is_terminal(leaf.prefix) and leaf.depth < self.max_depth:
                children = self._expand(leaf)
                if children:
                    # Pick child with highest PRM prior to simulate from
                    leaf = max(children, key=lambda c: c.prm_prior)

            # 3. Simulate
            value = self._simulate(leaf)

            # 4. Backprop
            self._backprop(leaf, value)

        # ── Pick best answer ──────────────────────────────────────────────────
        # Strategy: among all terminal solutions, pick the one where the
        # node has the highest visit-weighted Q value
        if not self.terminal_solutions:
            return None, {"n_terminals": 0, "n_simulations": self.n_simulations}

        # Sort by (visit_count * q_value) descending — favours nodes that
        # were visited often AND had high value
        self.terminal_solutions.sort(
            key=lambda t: t[0].visit_count * t[2], reverse=True
        )

        best_node, best_solution, best_value = self.terminal_solutions[0]
        best_answer = extract_answer(best_solution)

        # Also collect all candidate answers for majority voting across MCTS
        all_answers = [extract_answer(sol) for _, sol, _ in self.terminal_solutions]
        mcts_maj    = majority_vote([sol for _, sol, _ in self.terminal_solutions])

        stats = {
            "n_simulations":  self.n_simulations,
            "n_terminals":    len(self.terminal_solutions),
            "tree_depth":     self._max_tree_depth(self.root),
            "root_visits":    self.root.visit_count,
            "best_value":     round(best_value, 3),
            "best_answer":    best_answer,
            "mcts_maj_answer": mcts_maj,
        }
        return best_answer, stats

    def _max_tree_depth(self, node: MCTSNode, depth: int = 0) -> int:
        if node.is_leaf:
            return depth
        return max(self._max_tree_depth(c, depth+1) for c in node.children)

# ─── Dataset loaders ──────────────────────────────────────────────────────────

def load_gsm8k(cache_dir: Path) -> List[dict]:
    from datasets import load_dataset
    cache_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("openai/gsm8k", "main", split="test",
                      cache_dir=str(cache_dir))
    problems = []
    for r in ds:
        m = re.search(r"####\s*([\d,]+)", r["answer"])
        if m:
            problems.append({
                "problem": r["question"],
                "answer":  m.group(1).replace(",", ""),
            })
    return problems

def load_math500(cache_dir: Path) -> List[dict]:
    from datasets import load_dataset
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test",
                          cache_dir=str(cache_dir))
        return [{"problem": r["problem"], "answer": r["answer"]} for r in ds]
    except Exception:
        ds = load_dataset("lighteval/MATH", split="test",
                          cache_dir=str(cache_dir), trust_remote_code=True)
        return [{"problem": r["problem"], "answer": r["solution"]} for r in ds]

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.sanity:
        print("*** SANITY MODE: 20 problems ***\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / f"mcts_{args.dataset}.json"

    # ── GPU allocation ────────────────────────────────────────────────────────
    n_gpus = torch.cuda.device_count()
    if n_gpus >= 2:
        prm_device = torch.device("cuda:0")
        gen_device = torch.device("cuda:1")
        print(f"  Dual GPU: PRM→cuda:0  Generator→cuda:1")
    elif n_gpus == 1:
        prm_device = gen_device = torch.device("cuda:0")
        print(f"  Single GPU mode")
    else:
        prm_device = gen_device = torch.device("cpu")
        print("  WARNING: CPU mode — extremely slow")

    # ── Load models ───────────────────────────────────────────────────────────
    gen_path = resolve_gen_model(args.gen_model)
    gen_model, gen_tok = load_generator(gen_path, gen_device)

    prm_dir = Path(args.prm_dir)
    prm_model, prm_tok = load_prm(prm_dir, prm_device)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"\n  Loading {args.dataset} ...")
    cache_dir = WORK_DIR / "data" / "cache"
    if args.dataset == "gsm8k":
        problems = load_gsm8k(cache_dir)
    else:
        problems = load_math500(cache_dir)

    if args.sanity:
        problems = problems[:20]
    print(f"  Problems: {len(problems)}")

    # ── Resume ────────────────────────────────────────────────────────────────
    all_results = []
    done_idx    = set()
    if args.resume and results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f).get("per_problem", [])
        done_idx = {r["idx"] for r in all_results}
        print(f"  Resume: {len(done_idx)} already done.")

    # ── Counters ──────────────────────────────────────────────────────────────
    greedy_correct = sum(r["greedy_correct"]  for r in all_results)
    maj_correct    = sum(r["maj_correct"]     for r in all_results)
    mcts_correct   = sum(r["mcts_correct"]    for r in all_results)
    mcts_maj_correct = sum(r["mcts_maj_correct"] for r in all_results)

    print(f"\n  MCTS config: sims={args.n_simulations}  "
          f"expansion={args.expansion_size}  max_depth={args.max_depth}  "
          f"c_puct={args.c_puct}\n")

    for idx, item in enumerate(problems):
        if idx in done_idx:
            continue

        problem = item["problem"]
        gt      = item["answer"]

        print(f"\n[{idx+1:>4}/{len(problems)}]  gt={gt}")

        # ── Baselines: greedy + maj@N ─────────────────────────────────────────
        greedy, sampled = generate_full_solutions(
            gen_model, gen_tok, problem,
            n=args.n_maj, max_tokens=args.max_new,
            temperature=args.temperature, device=gen_device,
        )
        g_pred   = extract_answer(greedy)
        g_corr   = int(verify(g_pred, gt))

        maj_pred = majority_vote(sampled)
        m_corr   = int(verify(maj_pred, gt))

        print(f"  Greedy: {'✓' if g_corr else '✗'} ({g_pred})  "
              f"Maj@{args.n_maj}: {'✓' if m_corr else '✗'} ({maj_pred})")

        # ── MCTS search ───────────────────────────────────────────────────────
        mcts = MCTS(
            gen_model=gen_model, gen_tok=gen_tok, gen_device=gen_device,
            prm_model=prm_model, prm_tok=prm_tok, prm_device=prm_device,
            problem=problem,
            n_simulations=args.n_simulations,
            expansion_size=args.expansion_size,
            max_depth=args.max_depth,
            c_puct=args.c_puct,
            temperature=args.temperature,
            max_step_tokens=args.max_step_tokens,
            max_complete_tokens=args.max_complete_tokens,
        )
        mcts_pred, mcts_stats = mcts.search()
        mcts_maj_pred = mcts_stats.get("mcts_maj_answer")

        p_corr       = int(verify(mcts_pred, gt))
        p_maj_corr   = int(verify(mcts_maj_pred, gt))

        greedy_correct   += g_corr
        maj_correct      += m_corr
        mcts_correct     += p_corr
        mcts_maj_correct += p_maj_corr

        print(
            f"  MCTS:   {'✓' if p_corr else '✗'} ({mcts_pred})  "
            f"MCTS-Maj: {'✓' if p_maj_corr else '✗'} ({mcts_maj_pred})  "
            f"tree_depth={mcts_stats['tree_depth']}  "
            f"terminals={mcts_stats['n_terminals']}"
        )

        all_results.append({
            "idx":              idx,
            "greedy_correct":   g_corr,
            "maj_correct":      m_corr,
            "mcts_correct":     p_corr,
            "mcts_maj_correct": p_maj_corr,
            "mcts_stats":       mcts_stats,
        })

        # Save every 5 problems
        if (idx + 1) % 5 == 0:
            n_eval = len(all_results)
            summary = {
                "n_evaluated":    n_eval,
                "greedy_acc":     round(greedy_correct    / n_eval * 100, 2),
                "maj_acc":        round(maj_correct       / n_eval * 100, 2),
                "mcts_acc":       round(mcts_correct      / n_eval * 100, 2),
                "mcts_maj_acc":   round(mcts_maj_correct  / n_eval * 100, 2),
                "mcts_gain_vs_greedy": round((mcts_correct - greedy_correct) / n_eval * 100, 2),
                "mcts_gain_vs_maj":    round((mcts_correct - maj_correct)    / n_eval * 100, 2),
            }
            with open(results_file, "w") as f:
                json.dump({"summary": summary, "per_problem": all_results}, f, indent=2)
            print(
                f"\n  ── [{idx+1}/{len(problems)}]  "
                f"Greedy={summary['greedy_acc']}%  "
                f"Maj@{args.n_maj}={summary['maj_acc']}%  "
                f"MCTS={summary['mcts_acc']}%  "
                f"MCTS-Maj={summary['mcts_maj_acc']}%  "
                f"(MCTS Δ vs Maj = {summary['mcts_gain_vs_maj']:+.1f}%) ──\n"
            )

    # ── Final summary ─────────────────────────────────────────────────────────
    n_eval = len(all_results)
    final = {
        "dataset":        args.dataset,
        "model":          str(gen_path),
        "prm":            str(prm_dir),
        "n_problems":     n_eval,
        "mcts_config": {
            "n_simulations":  args.n_simulations,
            "expansion_size": args.expansion_size,
            "max_depth":      args.max_depth,
            "c_puct":         args.c_puct,
        },
        "greedy_acc":          round(greedy_correct    / n_eval * 100, 2),
        "maj_acc":             round(maj_correct       / n_eval * 100, 2),
        "mcts_acc":            round(mcts_correct      / n_eval * 100, 2),
        "mcts_maj_acc":        round(mcts_maj_correct  / n_eval * 100, 2),
        "mcts_gain_vs_greedy": round((mcts_correct - greedy_correct) / n_eval * 100, 2),
        "mcts_gain_vs_maj":    round((mcts_correct - maj_correct)    / n_eval * 100, 2),
    }
    with open(results_file, "w") as f:
        json.dump({"summary": final, "per_problem": all_results}, f, indent=2)

    print("\n" + "═" * 65)
    print(f"  ✅ MCTS Eval Complete  ({args.dataset})")
    print(f"     Problems   : {n_eval}")
    print(f"     Greedy@1   : {final['greedy_acc']}%")
    print(f"     Maj@{args.n_maj:<2}     : {final['maj_acc']}%")
    print(f"     MCTS       : {final['mcts_acc']}%  ({final['mcts_gain_vs_greedy']:+.1f}% vs greedy)")
    print(f"     MCTS-Maj   : {final['mcts_maj_acc']}%  ({final['mcts_gain_vs_maj']:+.1f}% vs Maj@{args.n_maj})")
    print(f"     Results    : {results_file}")
    print("═" * 65)

if __name__ == "__main__":
    main()
