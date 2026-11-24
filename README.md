# Parallel Safety Orchestrator (PSO)

Code, data-construction scripts and analysis for *Is latent representational conflict a
pre-generation safety signal for large language models? A pre-registered test and a diverse
single-view probe*
(Ashraf, Shamszaman, Aslam; submitted to Computers & Security).

The PSO frames the same prompt with three personas (Red Team, Blue Team,
Judge), runs them through the backbone in **one batched forward pass**, and
measures how far apart the persona-conditioned last-token hidden states are
(the **Multi-View Safety Disagreement Score**, MSDS, normalised as D~ = D/2).
The score drives an ALLOW / REQUERY / BLOCK decision before any token is generated.

## Main finding (read this first)

The paper has **one negative result, one positive result and a development of the positive one into a guard**. The negative one is a
**pre-specified test of the persona signal**. The hypothesis was that
concealed (framed) jailbreaks give *larger* persona disagreement than benign prompts, by
more than matched neutral prefixes do. On Llama-2-7B-Chat, Llama-3-8B-Instruct,
Mistral-7B-Instruct-v0.2 and Qwen2.5-7B-Instruct:

* jailbreaks disagree **less** than benign prompts (calibration AUC of D~ 0.14–0.43);
* neutral prefixes produce a gap of the same sign, and rewording the personas changes the
  sign of the persona × prompt interaction (Exp. 10, difference-in-differences with family bootstrap);
* prompt length and source dataset explain most of the remaining difference;
* the pre-specified detector detects only what its Phase-1 keyword filter detects;
* the sign-reversed score (`msds_dircal`, exploratory, Exp. 11) reaches test AUC 0.57–0.86,
  whereas a linear probe on one view of the bare prompt (Exp. 12) reaches 0.997–0.999.

The positive one is the single-view probe. In a **pre-registered template control**
(`PREREGISTRATION_exp13.md`, Exp. 13) with harmless requests wrapped in exactly the same
jailbreak templates, the probe still ranks harmful above harmless requests (AUC 0.91–0.98 on all
four backbones), so harmful intent is linearly readable before generation beyond the template.
Calibrated on untemplated harmless prompts it flags 31–43 % of templated harmless requests, so it
fails the control's operating criterion: hidden-state detectors need templated harmless prompts
in their calibration data.
An exploratory follow-up (`PREREGISTRATION_exp14.md`, Exp. 14) calibrates the probe with templated
harmless prompts: on held-out families it detects 98–99 % of framed jailbreaks while flagging 2–5 %
of framed harmless requests, meeting its criterion on three of four backbones.

Pre-registered tests on independent data (`PREREGISTRATION_exp15.md`, Exp. 15) show the limit of a probe
calibrated on one corpus: it misses GCG jailbreaks, flags most OR-Bench-Hard prompts and a score-aware
attacker evades it on 60 % of goals.

**From probe to guard (Exp. 16–18, exploratory, each with a dated pre-registration).**

* Exp. 16 (`exp16_diverse_probe.py`): nine source groups; leave-one-source-out (unseen sources are
  detected poorly); operator augmentation applied equally to harmful and harmless prompts gives the
  *diverse probe* (held-out AUC 0.991; strict score-aware ASR@10 0.04 on Llama-2).
* Exp. 17 (`exp17_coverage.py`): few-shot coverage curves (10 GCG families lift GCG detection from
  3–14 % to 94–100 %), seven held-out attack operators, near-duplicate control, replication on four
  backbones (AUC 0.991–0.992), probe-OR-Llama-Guard combination (`tools/combination_ci.py`).
* Exp. 18a (`exp18_encodings.py`): base64 / ROT13 / leetspeak / hex; leave-one-encoding-out
  augmentation fails, a rule-based canonicaliser lifts AUC to 0.99–1.00 on every backbone.
* Exp. 18b (`exp18_attack_update.py`): three rounds of harvesting evasions with harmless twins;
  strict score-aware ASR@10 0.115 → 0.055 on Llama-2.

`PREREGISTRATION_exp*.md` state every criterion before the run; the paper reports all of them
(Table "Every success criterion"), met or not. Hard harmless prompts (OR-Bench-Hard, benign
JailbreakBench twins) remain the main source of false alarms. The probe is a research artefact, not
a validated production defence.

**Settings of the reported run.** All paper numbers come from Kaggle T4×2 runs with
`config/config_fast.yaml` (3 seeds, 50 attacked goals per seed, 1,000 bootstrap resamples, 30-prompt
latency subset after 20 warm-up passes); `config/config.yaml` holds the larger defaults. The weekly
GPU quota ran out before the 7B HarmBench output classifier, the PAIR-style attacker, the third
score-aware seed and the first-evasion response judging could run; responses were re-judged on CPU
with Llama Guard 3-1B (`output_judge.rejudge_classifier: llama_guard`), and attack success counts an
ALLOW only when the variant is not a paraphraser refusal (`exp4_asr_validity.csv`).
`exp7_generalisation` finished only for Llama-2 on AdvBench, in-the-wild and HarmBench (Self-Critique
generation took the 11 h budget); its metrics were computed from the per-seed evaluation files and
are in `outputs/tables/exp7_pipeline_llama2.csv`. Score-level transfer for all four backbones is in
the `exp12` transfer table.

* No data ships with the repository; `data/prepare_data.py` downloads and builds it.
  `experiments.export_release` writes the released prompt sets and per-prompt results.
* No thresholds, model ids, prompts or coefficients are hard-coded: everything
  tunable is in `config/config.yaml`; thresholds and the Predictive Safety Model are
  fitted on the calibration split only.

## Paper → code

| Paper | Code |
|---|---|
| §3.2 score: last-token pooling, L2 normalisation, MSDS, D~, pairwise similarities | `models/llm_client.py` → `encode_hidden`; `core/msds.py` |
| §3.3 controls: identity check, neutral and paraphrased prefixes, DiD, same-goal pairs, length OLS, layer curves | `experiments/exp10_prefix_controls.py` |
| §3.4 PSO Phase 1 keyword/pattern filter, escalation patterns, TED | `wrappers/keyword_wrapper.py`, `wrappers/history_wrapper.py`, `wrappers/phase1.py` |
| §3.4 personas and framing | `config/config.yaml` → `pso.personas`, `pso.framing`; neutral/paraphrased prefixes under `experiments.exp10` |
| §3.4 thresholds τs/τb on the calibration split; direction calibration | `core/calibration.py`, `core/decision.py`, `experiments/exp11_direction.py` |
| §3.4 Predictive Safety Model (logistic fit, Brier, ECE) | `core/predictive_model.py`, `experiments/exp6_predictive.py` |
| Algorithm 1 (parallel) and Sequential Orchestrator, sanitiser | `wrappers/safety_orchestrator.py`, `core/sanitiser.py` |
| §4.3 baselines: keyword, LLM-judge, Self-Critique, embedding variance | `wrappers/` |
| §4.3 baselines: linear probes, perplexity, HiddenDetect-style, SmoothLLM, Llama Guard 3 | `experiments/exp12_baselines.py` |
| §4.4 output judge (HarmBench classifier) for UOR / attack success | `experiments/rejudge_outputs.py`, `evaluation/output_judge.py` |
| §5 adaptive attackers incl. score-aware | `attacks/attackers.py`, `experiments/exp4_robustness.py` |
| Metrics and statistics | `evaluation/metrics.py`, `evaluation/stats.py` |
| Every figure, from the logged CSVs | `experiments/make_figures.py` |
| LaTeX rows of the backbone, detector and transfer tables | `tools/make_tables.py` |
| §5 template control, template-aware probe, independent data | `experiments/exp13_*.py`, `exp14_template_aware_probe.py`, `exp15_independent_and_adaptive.py` |
| §6 diverse probe, LOSO, re-attack | `experiments/exp16_diverse_probe.py` |
| §6 coverage, operators, near-duplicates, combination | `experiments/exp17_coverage.py`, `tools/combination_ci.py` |
| §6 canonicaliser, encodings | `experiments/exp18_encodings.py` (`Canonicaliser`) |
| §6 update from evasions | `experiments/exp18_attack_update.py` |
| §6 figures and tables, radar figures | `tools/make_figs_exp16_18.py` |

## Revision notes (re-run protocol)

These changes make the experiments runnable on a Kaggle T4×2 within the weekly GPU quota and
close methodological gaps; the paper's tables must be regenerated from the new outputs.

* **Data** — `python data/prepare_data.py` downloads XSTest, AdvBench, HarmBench, JBB-Behaviors and
  the in-the-wild jailbreak prompts (Shen et al.), and builds:
  `advbench_framed.csv` (500 AdvBench goals, each wrapped in one hypothetical / roleplay / injection
  framing; 167/166/167, four templates per framing), `advbench_direct.csv` (the same goals unframed,
  matched control), `jailbreak_templates.csv` (templates ≤ 300 words with the placeholder), and
  `ambiguous_prompts.csv` (150 harmless-but-ambiguous prompts written for the study).
* **Calibration / test split** — prompts are assigned to a 30 % calibration split by a hash of their
  family (AdvBench goal), so direct and framed versions of a goal share a split.
  `tau_s`, `tau_b`, the embedding-variance threshold and the TED drift threshold are selected on the
  calibration split (`core/calibration.py`, Eq. 17 with λ = 0.5; `tau_b` = highest-TPR block threshold
  with ≤ 2 % harmless blocks); all reported TPR/FPR/UOR/AUC/F2 use the test split. The fixed defaults
  0.35 / 0.65 do not match the score scale of real backbones.
* **Speed** — batched left-padded generation (`LLMClient.generate_batch`, OOM fallback),
  batched multi-prompt scoring, hidden states read without the LM head, Self-Critique reusing the
  seed's sampled response as its draft, LLM-Judge/Self-Critique decisions batched with wall-clock
  latency timed unbatched on a fixed 50-prompt subset, attacks run in lockstep across prompts
  (`attacks.attackers.run_lockstep`) with feedback-independent paraphrases shared across defences,
  chunk checkpoints for responses. Attack experiments use 3 seeds.
* **Experiment 10** (`experiments/exp10_prefix_controls.py`) — the Section III-E protocol:
  identity control, matched neutral prefixes, persona paraphrases, baseline-adjusted score (Eq. 8),
  layer-wise curves, difference-in-differences Δ (Eq. 6) with family bootstrap, framed-vs-direct
  matched contrast, and a length-controlled regression.
* **Kaggle** — `kaggle/PSO_Kaggle.ipynb` + `kaggle/kaggle_runner.py` run stages A–D with a time
  budget and resume from the previous version's outputs.

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
| `python -m experiments.exp10_prefix_controls` | identity check, DiD with family bootstrap, same-goal pairs, length OLS, layer curves |
| `python -m experiments.exp11_direction` | direction-calibrated (exploratory) variant |
| `python -m experiments.exp12_baselines` | probes, perplexity, HiddenDetect-style, SmoothLLM, Llama Guard, transfer |
| `python -m experiments.exp13_template_control --backbone <bb>` | template control of the probe (paper Table 8; forward passes only) |
| `python -m experiments.exp13_llama_guard` | the same control for Llama Guard 3-1B (last row of paper Table 8; runs on CPU) |
| `python -m experiments.exp14_template_aware_probe --backbone <bb>` | template-aware probe, exploratory (paper Table 9; reuses the Exp. 13 states) |
| `python -m experiments.exp15_independent_and_adaptive --backbone <bb>` | independent data (PAIR, GCG, OR-Bench) and adaptive attacks on the probes |
| `python -m experiments.exp16_diverse_probe --backbone <bb>` | leave-one-source-out, diverse (augmented) probe, re-attack |
| `python -m experiments.exp17_coverage --backbone <bb>` | few-shot coverage, held-out operators, near-duplicates, per-prompt scores |
| `python tools/combination_ci.py outputs/ outputs/tables/` | probe OR Llama Guard with family-bootstrap intervals |
| `python -m experiments.exp18_encodings --backbone <bb>` | encodings: LOEO, all-encodings, canonicaliser |
| `python -m experiments.exp18_attack_update --backbone llama2` | attack–update loop (resumes from its round logs) |
| `python tools/make_figs_exp16_18.py outputs/tables paper/figs paper/` | Figs. coverage, operators, encodings, update, both radars; LaTeX rows of Tables for Exp. 16–18 |
| `python -m experiments.rejudge_outputs` | re-judges every generated response with the HarmBench classifier |
| `python -m experiments.make_figures` | all paper figures as PDF from `outputs/` |
| `python tools/make_tables.py outputs/tables paper/` | LaTeX table rows used by the paper |
| `python -m experiments.export_release --out release/` | released prompt sets, split assignments, per-prompt results |

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

