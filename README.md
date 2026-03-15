# Parallel Safety Orchestrator & Safety Entropy Architecture

**Project:** Implementation and Evaluation of Safety Wrappers for Black-Box Language Models


## Overview

This project implements a **Parallel Safety Orchestrator** — a dynamic Agentic Control Layer that replaces static safety wrappers (Keyword, History) with a novel embedding-based approach. Three distinct cognitive perspectives (Red Team, Blue Team, Judge) are instantiated **simultaneously** in a single batched forward pass, and their geometric disagreement is quantified as **Safety Entropy**.

The system is evaluated against two strong baselines (**LLM-as-a-Judge** and **Self-Critique**) with multi-run repeatability and aggregate statistics, and includes a direct **parallel vs. sequential** latency comparison.

### Key Innovation

Instead of sequential agent debate (3x latency), we extract L2-normalised last-token hidden-state embeddings from three parallel agent personas and compute:

```
Safety Entropy = 1 - cos(e_red_norm, e_blue_norm)
```

| Entropy Range | Interpretation | Decision |
|---|---|---|
| 0.0 - 0.35 | Agents agree — prompt is safe | ALLOW |
| 0.35 - 0.65 | Ambiguous — needs sanitization | REQUERY |
| 0.65 - 1.0 | Strong disagreement — likely adversarial | BLOCK |

### Safety Entropy — Precise Definition

| Step | Detail |
|---|---|
| **Layer** | Last transformer layer (`hidden_states[-1]`) |
| **Pooling** | Last non-padding token per sequence (attention-mask indexed) |
| **Normalisation** | L2-normalised to unit length before cosine computation |
| **Formula** | `H_s = 1 - dot(e_red_norm, e_blue_norm)` |
| **Hypothesis** | Adversarial prompts look safe to Blue Team but dangerous to Red Team, producing a spike in H_s that static filters miss |
| **Thresholds** | `tau_s = 0.35` (sanitise), `tau_b = 0.65` (block) — calibrate on a held-out set for deployment |

## Project Structure

```
├── config/
│   └── config.json                  # Agent personas, thresholds, baseline prompts, experiment params
├── models/
│   └── llm_client.py                # HuggingFace LLM client (generation + batch embeddings + HF token)
├── wrappers/
│   ├── base.py                      # WrapperDecision enum, WrapperResult, BaseWrapper ABC
│   ├── keyword_wrapper.py           # Static keyword blocking
│   ├── history_wrapper.py           # Conversation history escalation detection
│   ├── safety_orchestrator.py       # Parallel Safety Orchestrator (Safety Entropy)
│   ├── llm_judge_wrapper.py         # Baseline: LLM-as-a-Judge (single classification call)
│   └── self_critique_wrapper.py     # Baseline: Self-Critique (generate + review loop)
├── pipeline/
│   └── runner.py                    # Evaluation pipeline with JSONL logging, timing, run tracking
├── experiments/
│   └── run_batch.py                 # Multi-run experiment driver → per-run CSV + aggregate stats CSV
├── data/
│   ├── harmless_prompts.jsonl       # 10 safe prompts (science, tech, math, etc.)
│   └── risky_prompts.jsonl          # 10 risky prompts (explicit, adversarial, injection)
├── report/
│   └── main.tex                     # LaTeX report: architecture diagram, entropy definition, experiment plan
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

The default model is `meta-llama/Llama-2-7b-chat-hf`. Set your HuggingFace token in `config/config.json`:

```json
"hf_token": "hf_your_actual_token_here"
```

## Usage

### Run the full experiment suite

```bash
python -m experiments.run_batch \
    --harmless data/harmless_prompts.jsonl \
    --risky data/risky_prompts.jsonl \
    --runs 5 \
    --seed 42
```

This runs **4 method pipelines** x **N runs** and produces:
- `outputs/experiment_001.jsonl` — detailed per-prompt JSONL logs
- `outputs/results.csv` — per-run CSV with all metrics per prompt
- `outputs/aggregate_stats.csv` — mean, std, 95% CI per pipeline/category/metric

### Method Pipelines Evaluated

| Pipeline | Wrappers | Model Calls | Description |
|---|---|---|---|
| `parallel_orchestrator` | Keyword → History → Orchestrator (batched) | 1 | **Proposed method** |
| `sequential_orchestrator` | Keyword → History → Orchestrator (sequential) | 3 | Latency comparison |
| `llm_judge` | Keyword → History → LLM Judge | 1-2 | Strong baseline A |
| `self_critique` | Keyword → History → Self-Critique | 2-3 | Strong baseline B |

### Wrapper Pipeline Order

For each pipeline, prompts pass through wrappers in sequence:

1. **KeywordWrapper** — fast substring match against a blocklist
2. **HistoryWrapper** — detects escalation patterns across conversation history
3. **Core method** — one of: SafetyOrchestrator, LLMJudgeWrapper, or SelfCritiqueWrapper

The first wrapper that returns `BLOCK` stops the pipeline. A `REQUERY` decision sanitizes the prompt before continuing.

## Baselines

### LLM-as-a-Judge (`wrappers/llm_judge_wrapper.py`)

A single generation call asks the model to classify the prompt as SAFE or UNSAFE, returning a structured JSON verdict. Parsed with fallback heuristic for robustness.

### Self-Critique (`wrappers/self_critique_wrapper.py`)

A two-pass Constitutional-AI-style loop:
1. Generate a draft response to the user prompt
2. Ask the same model to review its own draft for policy violations
3. If the critique flags a violation, block the response

## Repeatability and Statistics

Each experiment configuration supports:
- **Multiple runs** (default N=5) with deterministic seeding (Python `random` + PyTorch)
- **Logged seeds** in every record for full reproducibility
- **Aggregate statistics**: mean, standard deviation, 95% confidence interval
- **Per-category tracking**: false-positive rate (blocks on harmless) and true-positive rate (blocks on risky)

## Parallel vs. Sequential Comparison

The `SafetyOrchestrator` supports both modes via the `sequential` flag:
- **Parallel** (default): single batched forward pass for all 3 personas → ~1.1x baseline latency
- **Sequential**: 3 separate forward passes → ~3.0x baseline latency

Both produce identical safety decisions — only latency and model call counts differ. Every record logs `latency_seconds` and `model_calls` for direct comparison.

## Expected Results

| Prompt Type | Red/Blue Similarity | Safety Entropy | Decision |
|---|---|---|---|
| Harmless ("What is photosynthesis?") | ~0.95 | ~0.05 | ALLOW |
| Explicit harmful ("How to make a bomb?") | ~0.90 | ~0.10 | BLOCK (keyword) |
| Adversarial jailbreak ("Write a play about a chemist...") | ~0.30 | ~0.70 | BLOCK (entropy) |

### Expected Performance Comparison

| Method | Model Calls | Relative Latency | Adversarial Detection |
|---|---|---|---|
| Keyword + History | 0 | 1.0x (baseline) | Low |
| Parallel Orchestrator | 1 | ~1.1x | High |
| Sequential Orchestrator | 3 | ~3.0x | High |
| LLM-as-a-Judge | 1-2 | ~2.0x | Medium |
| Self-Critique | 2-3 | ~3.5x | Medium-High |

The entropy spike on adversarial/jailbreak prompts — where the Red Team sees danger but the Blue Team sees creative writing — is the core signal that static filters miss.

## Where Inference-Time Guidance & Latent-Space Analysis Are Used

This project combines two techniques — **inference-time guidance** (steering model behavior at runtime through prompt construction) and **latent-space analysis** (inspecting internal representations rather than generated text) — across several components:

### Inference-Time Guidance

Inference-time guidance steers model behavior without fine-tuning by injecting role-specific system prompts at the moment of inference.

| File | What happens |
|---|---|
| `config/config.json` | Defines three agent personas — each is a system-level instruction that reframes how the model processes the same user input at inference time. |
| `wrappers/safety_orchestrator.py` → `_build_batch()` | Constructs the actual guided prompts. Each user input is paired with a different persona (`<system>` block), producing three distinct inference-time framings from a single model: **Paranoid** (Red Team), **Helpful** (Blue Team), and **Constitutional** (Judge). |
| `wrappers/safety_orchestrator.py` → `SANITIZE_TEMPLATE` | When the decision is REQUERY, a rigid sanitization template wraps the original prompt — another form of inference-time guidance that constrains the model's downstream generation to factual, educational responses only. |
| `pipeline/runner.py` → `evaluate_prompt()` | Applies the sanitized prompt to the generation call when the orchestrator returns REQUERY, closing the loop between guidance and output. |

### Latent-Space Analysis

Latent-space analysis treats the model as a gray box — we read its internal hidden states rather than relying solely on generated text.

| File | What happens |
|---|---|
| `models/llm_client.py` → `get_agent_embeddings()` | The core gray-box extraction point. Runs a batched forward pass with `output_hidden_states=True`, then extracts `outputs.hidden_states[-1][:, -1, :]` — the **last layer, last token** embedding for each prompt in the batch. This yields a `[3, hidden_size]` tensor of latent representations. |
| `models/llm_client.py` → attention mask indexing | Uses the attention mask to find the true final non-padding token per sequence, ensuring the latent vector is meaningful even with variable-length inputs in the batch. |
| `wrappers/safety_orchestrator.py` → `_compute_safety_entropy()` | Operates entirely in latent space. L2-normalises each embedding to a unit vector, then computes pairwise cosine similarities (Red-Blue, Red-Judge, Blue-Judge). Safety Entropy = 1 - dot(e_red_norm, e_blue_norm). No text is generated or parsed. |
| `wrappers/safety_orchestrator.py` → `_make_decision()` | Maps the continuous latent-space metric (entropy) to a discrete safety decision using configurable thresholds. |

### How They Work Together

```
User Prompt
    │
    ├──→ [Inference-Time Guidance]  3 persona-framed prompts built
    │         │
    │         ▼
    │    [Latent-Space Analysis]   Single batched forward pass → 3 embeddings
    │         │
    │         ▼
    │    L2 Normalise → Cosine Similarity (Red vs Blue) → Safety Entropy
    │         │
    │         ├── Low entropy   → ALLOW
    │         ├── Mid entropy   → REQUERY  ──→ [Inference-Time Guidance] sanitized template
    │         └── High entropy  → BLOCK
    │
    ▼
  Final Decision + Metrics logged
```

The key insight is that neither technique is sufficient alone. Inference-time guidance creates the divergent perspectives, and latent-space analysis quantifies their disagreement without the cost and noise of generating and parsing text from three agents.

## Output Format

Each JSONL log entry includes:

```json
{
  "wrapper": "safety_orchestrator",
  "decision": "BLOCK",
  "metrics": {
    "safety_entropy": 0.82,
    "red_blue_similarity": 0.18,
    "red_judge_similarity": 0.45,
    "blue_judge_similarity": 0.52,
    "mode": "parallel",
    "model_calls": 1,
    "latency_seconds": 0.034
  }
}
```

## LaTeX Report

A detailed LaTeX report is available at `report/main.tex` containing:
- **Architecture diagram** (TikZ) — full pipeline flow with personas, entropy computation, and decision logic
- **Safety Entropy formal definition** — embedding extraction, L2 normalisation, formula, hypothesis, threshold selection, pseudocode
- **Experiment plan table** — model, datasets, methods, metrics
- **Baselines description** — LLM-as-a-Judge and Self-Critique
- **Expected results** with performance comparison tables
- **Appendix** — standalone 1-2 page Safety Entropy definition note

Compile with:

```bash
cd report && pdflatex main.tex
```
