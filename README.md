# Parallel Safety Orchestrator (PSO)

Code for *Measuring Persona-Conditioned Representational Conflict for Pre-Generation Jailbreak Detection* (Ashraf, Shamszaman, Aslam).

The PSO frames the same prompt with three personas (Red Team, Blue Team,
Judge), runs them through the backbone in **one batched forward pass**, and
measures how far apart the persona-conditioned last-token hidden states are
(the **Multi-View Safety Disagreement Score**, MSDS). The score drives an
ALLOW / REQUERY / BLOCK decision before any token is generated.

* No data ships with the repository. Every dataset is read from the paths in
  `config/config.yaml` (see [`data/README.md`](data/README.md)).
* No thresholds, model ids, prompts or coefficients are hard-coded: everything
  tunable is in `config/config.yaml`, and the Predictive Safety Model is fitted
  on your calibration data.

## Paper → code

| Paper | Code |
|---|---|
| Stage 1.1 keyword & pattern filter, Eq. (1) | `wrappers/keyword_wrapper.py` |
| Stage 1.2 escalation patterns + Temporal Entropy Drift, Eqs. (2)–(3) | `wrappers/history_wrapper.py` |
| Phase 1 early termination, Eq. (4); Keyword + History baseline | `wrappers/phase1.py` |
| Personas (Appendix A–C), framing Eq. (5) | `config/config.yaml` → `pso.personas`, `pso.framing`; `ParallelSafetyOrchestrator.frame` |
| Batched forward pass Eq. (6), last-token pooling Eq. (7) | `models/llm_client.py` → `encode_hidden` |
| L2 normalisation Eq. (8), MSDS Eq. (9), D~ = D/2, S_RB/S_RJ/S_BJ Eqs. (10)–(12) | `core/msds.py` |
| Predictive Safety Model Eqs. (13)–(14), MLE fit | `core/predictive_model.py` |
| Probabilistic rule Eq. (17), geometric rule Eq. (18), ROC calibration Eq. (19) | `core/decision.py` |
| Sanitisation template T | `core/sanitiser.py` |
| Algorithm 1 (parallel) and Sequential Orchestrator | `wrappers/safety_orchestrator.py` |
| LLM-as-a-Judge, Self-Critique, Embedding Variance baselines | `wrappers/llm_judge_wrapper.py`, `wrappers/self_critique_wrapper.py`, `EmbeddingVarianceWrapper` |
| Naive / paraphrasing / adaptive / PAIR-style attackers | `attacks/attackers.py` |
| TPR, FPR, PTR, UOR, AUC, F2, Brier, ECE, AUC-PR, ASR@k | `evaluation/metrics.py` |
| Mean ± std over seeds, one-tailed paired t-tests, ANOVA + Tukey HSD, power | `evaluation/stats.py` |
| Per-prompt log schema (Table 3) | `evaluation/logs.py`, `experiments/export_logs.py` |

## Setup

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_...            # gated models (Llama-2/3); the variable name is backbones.hf_token_env
```

1. Download the datasets listed in [`data/README.md`](data/README.md).
2. Edit `config/config.yaml`: set `datasets.*.path` (and column names if your
   files differ), the backbones under `backbones.models`, and anything else
   you want to change.

## Running

Everything, in dependency order (each stage caches to `outputs/runs/` and
resumes if interrupted):

```bash
python -m experiments.run_all [--config config/config.yaml] [--backbone llama2] [--skip exp7_generalisation]
```

Or stage by stage:

| Command | Produces |
|---|---|
| `python -m experiments.run_main_evaluation` | per-prompt signals, counterfactual responses and every method's decisions/latency for each seed (needed by Exp. 1–3, 5, 6) |
| `python -m experiments.exp6_predictive` | Table 12, Fig. 10, fitted `outputs/predictive_model.json` |
| `python -m experiments.exp1_detection` | Tables 5–6, ANOVA/Tukey, softmax-entropy comparison, pooling ablation, Figs. 2–4 |
| `python -m experiments.exp2_utility` | Table 7, Fig. 5 |
| `python -m experiments.exp3_efficiency` | Table 8, Fig. 6 |
| `python -m experiments.exp4_robustness` | Table 9, ASR@k, PAIR comparison, subset stability, power, Figs. 7–8 |
| `python -m experiments.exp5_sensitivity` | F2 grid (Fig. 9), Eq. (19) thresholds, Tables 10–11 |
| `python -m experiments.exp7_generalisation` | Tables 13–15 |
| `python -m experiments.exp8_ambiguity` | Table 16 + significance tests |
| `python -m experiments.exp9_multiturn` | Table 17 |
| `python -m experiments.export_logs` | `outputs/pso_<backbone>_<dataset>.csv`, `outputs/results.csv`, `outputs/aggregate_stats.csv` |
| `python -m experiments.summary` | Table 18, Fig. 11 |

Tables are written to `outputs/tables/*.csv`, figures to `outputs/figures/*.png`.

### Using the PSO directly

```python
from config import load_config, backbone_config
from models.llm_client import LLMClient
from wrappers import ParallelSafetyOrchestrator, Conversation

cfg = load_config()
pso = ParallelSafetyOrchestrator(LLMClient(backbone_config(cfg)).load(), cfg)
conv = Conversation()                       # one per dialogue; enables the TED filter
result = pso.evaluate("user prompt", conv)
result.decision, result.metrics["disagreement_score"], result.sanitized_prompt
```

### Tests

```bash
python -m pytest
```

The tests check the implementation against the paper's equations (MSDS
bounds, Proposition 1, decision rules, TED, metric definitions).

## Implementation notes

Where the paper leaves a choice open, the code makes it explicit and
configurable:

* **Flagged decisions.** TPR/FPR/F2 count a prompt as flagged when its
  decision is in `evaluation.positive_decisions` (default BLOCK and REQUERY,
  matching the FPR definition of Table 7: "flagged as unsafe or borderline").
  With this definition F2 depends only on tau_s; set `[BLOCK]` to make it
  depend only on tau_b.
* **TED placement.** Eq. (3) needs D_s^(t), which exists only after the
  forward pass, so the drift alert is evaluated right after Phase 3 and
  overrides the decision with BLOCK, which is what Eq. (4) prescribes. TED is
  applied to the normalised score D~ (the scale of tau_s). Because the alert
  also requires D~ > tau_s, it can only turn a REQUERY into a BLOCK; Exp. 9
  therefore measures detection on BLOCK decisions
  (`experiments.exp9.detection_decisions`).
* **Single-turn protocol.** Each prompt of the single-turn experiments is its
  own conversation, so escalation patterns in one prompt never affect another.
* **UOR and attack success.** A response is unsafe according to
  `output_judge` (refusal-prefix convention of AdvBench/GCG by default, or a
  classifier such as Llama Guard). Evasion success is an ALLOW on a harmful
  prompt (Table 3); set `experiments.exp4.success_requires_unsafe_output` to
  also require an unsafe response.
* **Counterfactual responses.** For every prompt and seed the runner
  generates the response to the original and to the sanitised prompt once;
  every method's UOR, and the ablation variants, use them according to their
  decision. Self-Critique uses its own draft.
* **Utility Degradation.** 1 − cosine similarity between mean-pooled backbone
  embeddings of the response to the original prompt and the response to the
  sanitised prompt, for harmless prompts routed to REQUERY.
* **ROC for decision-only baselines.** LLM-as-a-Judge and Self-Critique emit
  decisions, so their ROC uses the ordinal score ALLOW=0, REQUERY=0.5,
  BLOCK=1. Pipelines with Phase 1 rank Phase-1 blocks above every
  orchestrator score.
* **Determinism.** The PSO wrapper is deterministic, so its detection
  metrics have zero variance across seeds on a fixed corpus; variance comes
  from sampled generation (baselines, responses), attacker randomness and
  sub-sampling.

`report/main.tex` is an earlier project report and predates the paper.
