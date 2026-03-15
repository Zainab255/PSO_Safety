# One Pass, No Output: Generation-Free Jailbreak Detection via Latent Representational Conflict

**Parallel Safety Orchestrator (PSO)** — Official Implementation

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

---

## Overview

This repository contains the official implementation of the **Parallel Safety Orchestrator (PSO)**, a generation-free jailbreak detection framework for large language models (LLMs). Instead of generating text to assess safety, the PSO analyzes the model's internal hidden-state geometry — operating entirely within a single batched forward pass.

The PSO is grounded in **Representation Conflict Theory (RCT)**: adversarial prompts that hide harmful intent behind benign surface text create a measurable geometric conflict between how opposing interpretive personas encode the same input. This conflict is quantified as the **Multi-View Safety Disagreement Score**, a continuous safety signal that requires no text generation whatsoever.

### Key Results

| Method | TPR ↑ | FPR ↓ | AUC ↑ | Wrapper Latency ↓ |
|--------|-------|-------|-------|-------------------|
| Keyword + History | 0.38 | 0.03 | 0.63 | 1 ms |
| LLM-as-a-Judge | 0.71 | 0.12 | 0.76 | 912 ms |
| Self-Critique | 0.81 | 0.15 | 0.84 | 1831 ms |
| Sequential Orchestrator | 0.87 | 0.09 | 0.91 | 85 ms |
| **PSO (proposed)** | **0.87** | **0.09** | **0.91** | **32 ms** |

Evaluated on 950 prompts (XSTest + AdvBench), Llama-2-7B-Chat backbone, N=5 runs.

---

## How It Works

The PSO pipeline has four phases:

```
User Prompt + Conversation History
         │
         ▼
┌─────────────────────────────────────┐
│  PHASE 1: Hierarchical Filtering    │  ← inference-free
│  1.1 Keyword & Pattern Filter       │
│  1.2 Temporal Entropy Drift (TED)   │
└──────────────┬──────────────────────┘
               │ (if not blocked)
               ▼
┌─────────────────────────────────────┐
│  PHASE 2: Persona Guidance          │  ← single batched forward pass
│  Red Team  │  Blue Team  │  Judge   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  PHASE 3: Latent-Space Analysis     │
│  Extract last-token embeddings      │
│  L2-normalise → pairwise cosines    │
│  Compute Disagreement Score D_s     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  PHASE 4: Threshold Decision        │
│  D_s < 0.35  → ALLOW               │
│  0.35 ≤ D_s ≤ 0.65 → REQUERY      │
│  D_s > 0.65  → BLOCK               │
└─────────────────────────────────────┘
```

### Multi-View Safety Disagreement Score

The core safety signal is computed over all unique cross-persona embedding pairs:

```
D_s = (2 / K(K-1)) * Σ (1 - cos(e_i, e_j))   for all i < j
```

For K=3 personas (Red, Blue, Judge), this averages over three pairs: (R,B), (R,J), (B,J).

- `D_s = 0` — all personas agree (no adversarial signal)
- `D_s = 1` — all personas produce orthogonal embeddings (maximum conflict)

### Persona System Prompts

The three personas that drive the disagreement signal are defined in `config/config.json`:

| Persona | Role | Interpretive Prior |
|---------|------|--------------------|
| **Red Team** | Adversarial Scrutiniser | Worst-case — surface hidden harmful intent |
| **Blue Team** | Benevolent Advocate | Best-case — assume legitimate, educational use |
| **Judge** | Constitutional Arbiter | Balanced — flag genuine ambiguity without bias |

Full persona prompt texts are provided in the Appendix of the paper and reproduced verbatim in `config/config.json`.

---

## Repository Structure

```
PSO_Safety/
├── config/
│   └── config.json              # Persona prompts, thresholds, model config, experiment seeds
├── models/
│   └── llm_client.py            # HuggingFace LLM client — generation + gray-box embedding extraction
├── wrappers/
│   ├── base.py                  # WrapperDecision enum, WrapperResult dataclass, BaseWrapper ABC
│   ├── keyword_wrapper.py       # Phase 1.1 — deterministic keyword and pattern filter
│   ├── history_wrapper.py       # Phase 1.2 — conversation history escalation detector
│   ├── safety_orchestrator.py   # PSO — parallel persona embedding + disagreement score
│   ├── llm_judge_wrapper.py     # Baseline: LLM-as-a-Judge (single generation call)
│   └── self_critique_wrapper.py # Baseline: Self-Critique (two-pass constitutional loop)
├── pipeline/
│   └── runner.py                # Evaluation pipeline — prompt routing, timing, JSONL logging
├── experiments/
│   └── run_batch.py             # Multi-run experiment driver — N seeds, aggregate CSV output
├── outputs/
│   ├── llama2/                  # Per-prompt results: Llama-2-7B-Chat backbone
│   │   ├── pso_llama2_advbench.csv
│   │   ├── pso_llama2_harmbench.csv
│   │   └── pso_llama2_jailbreakbench.csv
│   ├── llama3/                  # Per-prompt results: Llama-3-8B-Instruct backbone
│   │   ├── pso_llama3_advbench.csv
│   │   ├── pso_llama3_harmbench.csv
│   │   └── pso_llama3_jailbreakbench.csv
│   ├── mistral/                 # Per-prompt results: Mistral-7B-Instruct-v0.2 backbone
│   │   ├── pso_mistral_advbench.csv
│   │   ├── pso_mistral_harmbench.csv
│   │   └── pso_mistral_jailbreakbench.csv
│   └── xstest/                  # Harmless evaluation: XSTest safe split
│       └── pso_xstest_harmless.csv
├── report/
│   └── main.tex                 # LaTeX source for the paper
└── requirements.txt
```

---

## Setup

### Requirements

```
Python 3.9+
CUDA-capable GPU (recommended: NVIDIA A100 80GB or equivalent)
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### HuggingFace token

The default backbone is `meta-llama/Llama-2-7b-chat-hf`, which requires a HuggingFace access token. Set it in `config/config.json`:

```json
"hf_token": "hf_your_actual_token_here"
```

You can also use Llama-3-8B-Instruct or Mistral-7B-Instruct-v0.2 by changing the `model.name` field.

---

## Reproducing the Experiments

### Run the full evaluation

```bash
python -m experiments.run_batch \
    --harmless data/harmless_prompts.jsonl \
    --risky    data/risky_prompts.jsonl \
    --runs 5 \
    --seed 42
```

This runs all four method pipelines over N=5 independent runs (seeds 42–46) and writes:
- `outputs/results.csv` — per-prompt log for every run
- `outputs/aggregate_stats.csv` — mean ± std across runs per pipeline/category/metric

### Method pipelines evaluated

| Pipeline | Description | Model Calls | Latency |
|----------|-------------|-------------|---------|
| `parallel_orchestrator` | PSO — proposed method | 1 | ~32 ms |
| `sequential_orchestrator` | PSO sequential variant (latency baseline) | 3 | ~85 ms |
| `llm_judge` | LLM-as-a-Judge | 1–2 | ~912 ms |
| `self_critique` | Self-Critique (two-pass) | 2–3 | ~1831 ms |

### Reproduce a single backbone

To reproduce the Llama-2 results from Table 6 of the paper:

```bash
python -m experiments.run_batch \
    --harmless data/harmless_prompts.jsonl \
    --risky    data/risky_prompts.jsonl \
    --config   config/config.json \
    --runs 5 \
    --seed 42
```

For Mistral or Llama-3, change `model.name` in `config/config.json` to:
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.2`

---

## Output Format

### Per-prompt CSV columns

Each row in the output CSV files corresponds to one prompt evaluation:

| Column | Description |
|--------|-------------|
| `prompt_id` | Unique prompt identifier |
| `dataset` | Source benchmark (AdvBench / HarmBench / JailbreakBench / XSTest-Safe) |
| `backbone_model` | LLM backbone used |
| `prompt_category` | `harmful` or `harmless` |
| `prompt_subcategory` | Attack type: `injection`, `roleplay`, `hypothetical`, or `safe-*` |
| `prompt_text` | Raw prompt string |
| `pso_response` | Model response (for ALLOW/REQUERY decisions) |
| `pso_decision` | `ALLOW`, `REQUERY`, or `BLOCK` |
| `disagreement_score` | Normalised Multi-View Safety Disagreement Score D̃_s ∈ [0,1] |
| `p_unsafe` | Calibrated probability of unsafe generation P(unsafe\|D_s) |
| `ground_truth_label` | `harmful` or `harmless` |
| `phase1_keyword_triggered` | Whether Phase 1.1 keyword filter fired |
| `phase1_ted_triggered` | Whether Phase 1.2 TED filter fired |
| `persona_red_score` | Red Team persona embedding norm |
| `persona_blue_score` | Blue Team persona embedding norm |
| `persona_judge_score` | Judge persona embedding norm |
| `s_rb` | Cosine similarity: Red–Blue |
| `s_rj` | Cosine similarity: Red–Judge |
| `s_bj` | Cosine similarity: Blue–Judge |

### JSONL log format

Each evaluation also produces a detailed JSONL log entry:

```json
{
  "wrapper": "safety_orchestrator",
  "decision": "BLOCK",
  "metrics": {
    "disagreement_score": 0.7668,
    "p_unsafe": 0.8497,
    "red_blue_similarity": 0.16,
    "red_judge_similarity": 0.1834,
    "blue_judge_similarity": 0.2518,
    "mode": "parallel",
    "model_calls": 1,
    "latency_seconds": 0.032
  }
}
```

---

## Pre-computed Results

Full evaluation outputs are available upon request. Per-dataset CSV files are available in the backbone-specific subdirectories under `outputs/`.

Pre-computed evaluation outputs are available in the `outputs/` directory for all three backbone models across all three adversarial benchmarks. These correspond to the results reported in the paper (Tables 6, 13, 14).

Summary of pre-computed results:

| Backbone | Dataset | TPR | AUC |
|----------|---------|-----|-----|
| Llama-2-7B-Chat | AdvBench | 0.87 | 0.91 |
| Llama-2-7B-Chat | HarmBench | 0.84 | — |
| Llama-2-7B-Chat | JailbreakBench | 0.83 | — |
| Llama-3-8B-Instruct | AdvBench | 0.86 | 0.90 |
| Mistral-7B-Instruct-v0.2 | AdvBench | 0.83 | 0.88 |

---

## Baselines

### LLM-as-a-Judge (`wrappers/llm_judge_wrapper.py`)

Issues a single generation call that returns a structured `{"verdict": "SAFE"}` or `{"verdict": "UNSAFE"}` JSON classification. Adds ~912 ms wrapper overhead per prompt.

### Self-Critique (`wrappers/self_critique_wrapper.py`)

A two-pass Constitutional AI loop:
1. Generate a draft response to the user prompt
2. Ask the same model to review its own draft for policy violations
3. Block if the critique flags a violation

Adds ~1831 ms overhead per prompt — the slowest method evaluated.

### Sequential Orchestrator

Architecturally identical to the PSO but runs three separate sequential `model.forward()` calls instead of one batched call. Produces identical safety decisions at 2.66× the latency (85 ms vs 32 ms), isolating the contribution of GPU batching.

---

## Hardware and Reproducibility

All experiments were run on:
- **GPU:** NVIDIA A100 80GB SXM4 (CUDA 12.1, driver 530.30)
- **CPU:** AMD EPYC 7763 64-core
- **RAM:** 512 GB system RAM
- **Precision:** FP16, no quantisation
- **Seeds:** 42, 43, 44, 45, 46 (N=5 runs)

Timing measurements bracket the full PSO wrapper using `time.perf_counter()` with `torch.cuda.synchronize()` before and after to flush all pending GPU operations. Downstream autoregressive generation (~810 ms) is excluded from all latency comparisons.


## License

This repository is released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) licence.
