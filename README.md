# MathReason-1.5B

> **Training a 1.5B parameter math reasoning model to match 7B baselines via targeted post-training**  
> Group Project · T225-AID 729, Section B · IIIT-Bangalore · April 2026  
> Sreekantam Sai Venkat (IMT2023501) · Revanth (IMT2023118) · Aditya Reddy (IMT2023114) · Santosh Kiran (IMT2023065)

---

## Results

| Model | Params | GSM8K Greedy@1 | GSM8K Maj@8 | pass@5 |
|---|---|---|---|---|
| Qwen2.5-1.5B base *(our start, 2-shot CoT prompting)* | 1.5B | 47% | — | — |
| Qwen2.5-1.5B-Instruct | 1.5B | ~73% | — | — |
| Qwen2.5-3B-Instruct | 3B | 79.1% | — | — |
| Llama-3.1-8B-Instruct | 8B | 82.4% | — | — |
| Qwen2.5-7B-Instruct | 7B | 85.4% | — | — |
| DeepSeek-R1-Distill-1.5B | 1.5B | ~85% | — | — |
| **MathReason-1.5B (ours)** | **1.5B** | **78%** | **82%** | **83%** |

**Key result:** Our 1.5B model matches Llama-3.1-8B-Instruct (82% Maj@8) at 5× smaller size, using only 31K distillation traces — 25× less than DeepSeek-R1-Distill-1.5B (800K traces).

🤗 **Model on HuggingFace:** [`Saivenkat2006/MathReason-1.5B-Stage7-DPO`](https://huggingface.co/Saivenkat2006/MathReason-1.5B-Stage7-DPO)

---

## What This Project Is

We start from `Qwen2.5-1.5B` — a **general-purpose base with no math specialization** — and apply a 6-stage post-training pipeline to teach it mathematical reasoning. All benchmark gains come entirely from our pipeline, not from pretraining advantages.

The core thesis: **a well-designed post-training pipeline can substitute for large-scale data** in building capable math reasoning models.

```
Qwen2.5-1.5B (general base, 47% GSM8K)
    │
    ├── Stage 1 — Context Extension      (4K → 16K tokens)
    ├── Stage 2 — SFT                    (teach reasoning format)
    ├── Stage 3 — CoT Distillation       (31K DeepSeek-R1 traces)
    ├── Stage 4 — GRPO                   (binary RL, termination learning)
    ├── Stage 4D — GDPO                  (multi-reward RL, loop suppression)
    ├── Stage 5 — Gap-Fill SFT           (targeted failure mode fixes)
    ├── Stage 6 — DPO Alignment          (loop rejection pairs)
    └── Stage 7 — PRM                    (process reward model)
                                          ↓
                              MathReason-1.5B (82% Maj@8)
```

---

## Repository Structure

```
MathReason-1.5B/
├── pipeline/
│   ├── stage1_context/               # Context extension scripts
│   ├── stage2_sft/                   # Supervised fine-tuning
│   ├── stage3_distill/               # CoT distillation from R1 traces
│   ├── stage4a_grpo/                 # GRPO Phase A (binary reward)
│   ├── stage4b_grpo/                 # GRPO Phase B (harder problems)
│   ├── stage4c_fullgrpo/             # Full-parameter GRPO
│   ├── stage4d_gdpo/                 # GDPO multi-reward RL
│   ├── stage5a_build_dataset.py      # Gap-fill dataset construction
│   ├── stage5b_generate_local.py     # Local teacher generation
│   ├── stage5c_gap_sft.py            # Gap-fill SFT training
│   ├── stage6_gdpo_phase2.py         # GDPO Phase 2
│   ├── stage7_build_dpo_data.py      # DPO pair construction
│   ├── stage7_dpo_train.py           # DPO training
│   ├── train_orm.py                  # Outcome reward model training
│   ├── train_prm.py                  # Process reward model training
│   ├── generate_mc_step_labels.py    # Monte Carlo step labeling
│   ├── eval_prm_best_of_n.py         # PRM Best-of-N evaluation
│   └── mcts_inference.py             # MCTS inference
├── evals/
│   ├── mcts_gsm8k.json               # MCTS eval results
│   └── prm_best_of_n_gsm8k.json      # PRM Best-of-N results
├── docs/
│   ├── PROGRESS.md                   # Concise milestone tracker
│   └── PROGRESS_FULL.md              # Full annotated pipeline log
└── README.md
```

---

## Pipeline — Stage by Stage

### Stage 1 — Context Extension (`stage1_context/`)

Extends the model's context window from 4K to 16K tokens using **YaRN RoPE scaling** (zero-shot NTK — no continued pretraining required). This is a prerequisite for Stage 3 distillation, where DeepSeek-R1 reasoning traces average 2,471 tokens of thinking plus problem context.

**Why needed:** Standard Qwen2.5-1.5B has a 4K context limit. R1-style reasoning chains regularly exceed 2,000 tokens. Without this stage, long reasoning traces get truncated mid-solution.

---

### Stage 2 — Supervised Fine-Tuning (`stage2_sft/`)

Teaches the model the reasoning format: `<think>...</think><solution>...</solution>`. Everything downstream depends on consistent tag usage — GRPO reward functions, answer extractors, and DPO pairs all assume this structure.

**Dataset:** ~50K examples from MetaMathQA + GSM8K train + MATH Level 1–3  
**Method:** LoRA (rank=32) + SFTTrainer, 3 epochs, LR=2e-5  
**Critical setting:** `label_smoothing_factor=0.0` — non-negotiable. Any smoothing value adds `log(vocab_size) ≈ 11.9` per-token loss, causing loss explosion (we hit train loss=44.45 in an earlier run by forgetting this).

**Output:** `<think>` compliance 100%, `<solution>` compliance 97%, GSM8K 70% (up from 47%)

---

### Stage 3 — CoT Distillation (`stage3_distill/`)

The most impactful stage. Transfers long chain-of-thought reasoning patterns from **DeepSeek-R1-671B** into our 1.5B model via 31K curated traces from `open-r1/OpenR1-Math-220k`.

**Dataset curation:** Filtered for hard sources only (olympiads, AMC/AIME, AoPS forum, Chinese contests). Easy problems that the SFT model already solves contribute near-zero distillation signal — keeping them wastes training budget.

**Dataset breakdown:**
- GSM8K hard subset: 4.2K traces
- MATH Level 2–4: 9.8K traces
- NuminaMath olympiads: 8.1K traces
- AMC/AIME: 4.6K traces
- OpenR1-misc: 4.3K traces

**Local teacher:** Where R1-671B traces were unavailable, we generated locally using `DeepSeek-R1-Distill-Qwen-7B` on GPU 0. Both share the R1 reasoning format, avoiding style inconsistency.

**Output:** GSM8K ~85% (real, measured with smart extractor), MATH500 ~55%

> **The Extraction Bug:** Naive "last number" extractors reported 48% GSM8K after distillation — a 3× undercount. The model computes the correct answer inside `<think>` at token ~200–300, then re-verifies and loops. The loop region often ends with a different number. The **smart extractor** scans the first 60% of the response and finds the `<solution>` tag before the loop region. Real accuracy was ~85%.

---

### Stage 4 — GRPO (`stage4a_grpo/`, `stage4b_grpo/`, `stage4c_fullgrpo/`)

Reinforcement learning with Group Relative Policy Optimization to improve answer consistency and correct termination behavior.

**What GRPO cannot do:** Expand the capability ceiling. GRPO refines behaviors the model learned during distillation — it cannot teach techniques the model has never seen. The ceiling is set by distillation data volume.

**The reward (binary — nothing else):**
```python
reward = 1.0 if verify_answer(predicted, ground_truth) else 0.0
```

We ran three reward designs. The first two failed:

| Run | Reward Design | Result |
|---|---|---|
| v1 | HAPO length reward (weight=0.3) | Regression 70% → 64.5% |
| v2 | 5-component: correct + format + termination + anti-loop + length | MATH500 collapsed 55% → 12% |
| **v3** | **Binary only** | **Stable, working** |

The 5-component collapse is explained by conflicting gradients on a 1.5B model. Each reward component pushes weights in a different direction; the model cannot reconcile them and collapses to a degenerate policy.

**Dataset difficulty calibration:** Problems are filtered to a **25–55% expected pass rate**. Below 25% = zero positive rollouts = dead gradient. Above 55% = near-zero reward variance = no useful advantage signal. GSM8K (85%+ accuracy at this stage) and DeepMath Level 5–9 (<5% pass rate) were both excluded.

**DAPO asymmetric clipping** (`ε_low=0.20, ε_high=0.28`): Higher upper clip preserves high-entropy reasoning tokens ("wait", "however", "let me reconsider") that act as reasoning pivots. Without this, entropy collapse kills exploratory reasoning.

---

### Stage 4D — GDPO (`stage4d_gdpo/`)

Extends GRPO with targeted rewards for the core failure mode: **model finds correct answer → unnecessary re-verification → wrong answer overwrites correct one**.

**Three reward signals (normalized independently, then summed):**
1. **Correctness** (0/1) — same binary signal as GRPO
2. **Termination bonus** (0–0.3) — fires when `</think><solution>` appears within 200 tokens of the correct answer appearing in the think block
3. **Efficiency penalty** — length-based penalty on unnecessarily long responses

GDPO normalizes each reward independently before summing, which preserves distinct gradient signals. Standard multi-reward GRPO sums raw rewards, allowing one dominant signal to drown others.

---

### Stage 5 — Gap-Fill SFT (`stage5a_build_dataset.py`, `stage5b_generate_local.py`, `stage5c_gap_sft.py`)

Targeted SFT to fix three specific failure modes identified through diagnostic evaluation after Stage 3.

**`stage5a_build_dataset.py`** — Constructs the gap-fill training dataset by sampling problems of each identified failure type from existing datasets (MATH, GSM8K hard) and formatting them for local generation.

**`stage5b_generate_local.py`** — Runs `DeepSeek-R1-Distill-Qwen-7B` on GPU 0 to generate correct reference solutions for the gap-fill problems. Filters for verified-correct outputs only (1,592 examples passed filtering).

**`stage5c_gap_sft.py`** — LoRA SFT (rank=32, LR=2e-5, 3 epochs) on the 1,592 generated traces.

**Three gaps targeted:**

| Failure | Example | Fix |
|---|---|---|
| Floor/ceiling arithmetic | Outputs 128.43 instead of ⌊900/7⌋ = 128 | 1K traces with explicit `floor(n/k)` notation |
| Multi-value answers | `<solution>12</solution>` when answer is "12 and 8" | 1K traces showing `<solution>12 and 8</solution>` |
| Counting/combinatorics | Missing "place group first, count gaps" technique | 2K traces showing gap-placement method |

---

### Stage 6 — DPO Alignment (`stage7_build_dpo_data.py`, `stage7_dpo_train.py`)

Direct Preference Optimization to surgically suppress loop failure modes that GDPO reduced but did not eliminate.

**`stage7_build_dpo_data.py`** — Constructs 807 preference pairs from the model's own diagnostic outputs. Three loop types each get dedicated rejection examples:

- **Re-verification spiral** — chosen: stops after first correct answer; rejected: continues into second-method re-verification that overwrites correct answer
- **"Wait, no" catastrophic loop** — chosen: confident single-method solution; rejected: "Wait, no, that's not possible" × 45 verbatim repetitions
- **Infinite repetition trap** — chosen: advances computation each paragraph; rejected: same problem summary paragraph repeated 6–8 times

**`stage7_dpo_train.py`** — LoRA DPO training (rank=32, LR=5e-7, 3 epochs, β=0.1). `label_smoothing=0.0` enforced throughout.

---

### Stage 7 — PRM (`train_prm.py`, `generate_mc_step_labels.py`, `eval_prm_best_of_n.py`)

Process Reward Model: a step-level evaluator that scores whether each reasoning step is on the correct track.

**`generate_mc_step_labels.py`** — Generates step-level training labels using **Monte Carlo rollout labeling** (Math-Shepherd approach). For each reasoning step prefix in a solution, samples K=8 completions from that point forward and checks how many reach the correct final answer. The step's label is 1 (good) if ≥30% of rollouts are correct, 0 (bad) at the first drop. Generated 9,880 step records from 807 problems (9,702 good / 178 bad).

> The 98.2% good-step ratio reflects the final model's strength at Stage 6 — most reasoning steps are already on the correct track. The PRM is most useful for the 17% of problems where the model is inconsistent (greedy wrong but Maj@8 correct).

**`train_prm.py`** — Trains `Qwen2.5-0.5B` + 2-layer scalar head (hidden→256→1, sigmoid) on the step labels. LoRA rank=16, LR=2e-4, 3 epochs, batch=8. Final: val loss=0.1054, val acc=98.2%.

**`eval_prm_best_of_n.py`** — Evaluates PRM Best-of-N on the GSM8K and MATH500 test sets. Generates N solutions per problem, scores each step of each solution with the PRM, and selects the solution where the minimum step score (the weakest step) is highest — i.e., the solution with no obviously wrong steps.

**`mcts_inference.py`** — Lightweight beam search guided by PRM scores. At each reasoning step, generates 4 continuations, scores them with the PRM, keeps the top 2, and expands again. Simpler than full MCTS but captures most of the benefit for models under 7B.

---

### ORM (`train_orm.py`)

Outcome Reward Model trained on solution-level preference pairs (Bradley-Terry pairwise loss). Base: `Qwen2.5-0.5B`. Achieved 92.5% validation ranking accuracy but ORM-weighted Maj@8 underperformed raw Maj@8 by ~3.5% — attributed to distribution mismatch between DPO training pairs and GSM8K eval distribution.

---

## Key Lessons

**1. RLVR is a refinement tool, not a capability builder.**  
GRPO cannot teach the model techniques it has never seen — only distillation can raise the capability ceiling. Our pass@5 ceiling (83%) is determined entirely by 31K distillation traces, not by RL. The 28-point gap vs. DeepSeek-R1-Distill-1.5B (91% ceiling) is explained by 25× less data.

**2. Binary reward is essential for RL stability.**  
Multi-component reward (5 signals) caused catastrophic MATH500 collapse from 55% to 12%. Two clean signals (correctness + termination) outperform five-tier schemes in practice.

**3. Dataset difficulty calibration is critical.**  
Target 25–55% pass rate for healthy GRPO signal. GSM8K at 85%+ accuracy = dead gradient. DeepMath Level 5–9 at <5% pass rate = dead gradient. Both were excluded.

**4. Evaluation extraction is as important as training.**  
A naive "last number" extractor reported 13.3% MATH500 at GRPO step 200 — 5× lower than the real ~60%+. Wrong extraction numbers cause false regression alarms and wrong training decisions. Always use a loop-aware smart extractor.

**5. Loop failure is the dominant failure mode for distilled reasoning models.**  
Correct answer found → unnecessary re-verification → wrong answer overwrites correct one. This single pattern accounts for the majority of incorrect outputs. DPO rejection pairs are the surgical fix; distillation quality partially reduces loops as a side effect.

**6. `label_smoothing=0.0` is mandatory throughout.**  
Any smoothing value adds approximately `log(vocab_size)` per-token loss during SFT. With a 150K vocabulary this is ~11.9 nats — enough to make loss explode and training useless. We learned this the hard way (11 hours of wasted compute).

---

## Hardware & Environment

| Component | Spec |
|---|---|
| GPUs | 2× RTX 6000 Ada (48 GB each, 96 GB total) |
| Server | `revanth@172.16.192.168` |
| CUDA / Driver | 13.1 / 590.48.01 |
| Framework | PyTorch 2.10 + HuggingFace TRL 0.29.0 |
| Attention kernel | SDPA (Flash Attention 2 unavailable — blocked by server glibc) |
| Precision | bfloat16 throughout |
| GPU allocation | GPU 1 trains; GPU 0 runs teacher inference concurrently |

**TRL 0.29.0 API notes (for reproducibility):**
- Use `processing_class` not `tokenizer` in GRPOTrainer/DPOTrainer
- Use `eval_strategy` not `evaluation_strategy`
- `reward_weights` lives in `GRPOConfig`, not the trainer constructor
- `DPOConfig` has no `max_prompt_length` — set via tokenizer max_length

---

## Datasets Used

| Stage | Dataset | Size | Source |
|---|---|---|---|
| SFT | MetaMathQA, GSM8K train, MATH L1-3 | ~50K | HuggingFace |
| Distillation | OpenR1-Math-220k (filtered) | 31K | `open-r1/OpenR1-Math-220k` |
| GRPO | MetaMath + MATH L2-4 + NuminaMath | 13,404 | HuggingFace |
| Gap-fill | Generated via R1-Distill-7B | 1,592 | Local generation |
| DPO | Model diagnostic outputs | 807 pairs | Self-generated |
| PRM | MATH train + GSM8K hard | 9,880 labels | MC rollout labeling |

---

## Reproducing Results

```bash
# Clone repo
git clone https://github.com/Saivenkat2006/MathReason-1.5B.git
cd MathReason-1.5B

# Install dependencies
pip install torch transformers trl==0.29.0 peft datasets accelerate

# Run pipeline in order (each stage takes the previous stage's merged checkpoint as input)
# Stage 2 — SFT
CUDA_VISIBLE_DEVICES=1 python pipeline/stage2_sft/stage2_sft.py

# Stage 3 — Distillation
CUDA_VISIBLE_DEVICES=1 python pipeline/stage3_distill/stage3_distill_train.py

# Stage 4A — GRPO
CUDA_VISIBLE_DEVICES=1 python pipeline/stage4a_grpo/stage4a_grpo.py

# Stage 4D — GDPO
CUDA_VISIBLE_DEVICES=1 python pipeline/stage4d_gdpo/stage4d_gdpo.py

# Stage 5 — Gap-fill
CUDA_VISIBLE_DEVICES=0 python pipeline/stage5b_generate_local.py   # teacher on GPU 0
CUDA_VISIBLE_DEVICES=1 python pipeline/stage5c_gap_sft.py          # train on GPU 1

# Stage 6 — DPO
CUDA_VISIBLE_DEVICES=1 python pipeline/stage7_dpo_train.py

# PRM (after Stage 6 is complete)
CUDA_VISIBLE_DEVICES=0 python pipeline/generate_mc_step_labels.py  # ~3 hours
CUDA_VISIBLE_DEVICES=0 python pipeline/train_prm.py                 # ~1 hour
CUDA_VISIBLE_DEVICES=0 python pipeline/eval_prm_best_of_n.py        # evaluation
```

> All training scripts support `--sanity` flag for quick smoke tests (100 samples, 5 steps).  
> All long-running scripts should be run inside `tmux` for disconnect resilience.

---

## References

1. DeepSeek-AI et al. (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* arXiv:2501.12948
2. Anonymous (2025). *Reinforcement Learning vs. Distillation: Understanding Accuracy and Capability in LLM Reasoning.* NeurIPS MATH-AI Workshop 2025
3. Shao et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* arXiv:2402.03300
4. Guan et al. (2025). *rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking.* arXiv:2501.04519
5. Wang et al. (2024). *Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations.* ACL 2024
6. Liao et al. (2025). *DeepScaleR: Surpassing O1-Preview with a 1.5B Model by Scaling RL.* Technical Report
7. Qwen Team (2024). *Qwen2.5 Technical Report.* arXiv:2412.15115
