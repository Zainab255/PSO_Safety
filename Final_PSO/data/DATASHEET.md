# Datasheet: PSO persona-disagreement benchmark

Released with *Persona-conditioned disagreement as a pre-generation jailbreak signal: a
pre-specified test, controls and an exploratory detector*. Built by `data/prepare_data.py`;
written to `release/data/` by `python -m experiments.export_release`.

## Files

| File | Rows | Content |
|---|---|---|
| `main_corpus_950.csv` | 950 | The main corpus on which thresholds, probes and the score direction are calibrated: XSTest safe (250; factual 100, technical 50, conversational 50, creative 50), XSTest unsafe (200, explicit harmful) and the 500 framed jailbreaks (hypothetical 167, role-play 166, override 167). `ground_truth` 1 = harmful, 0 = harmless; split 277 calibration / 673 test. |
| `adversarial_framed.csv` | 500 | Every AdvBench goal wrapped in one of 12 framing templates written for this study (hypothetical 167, role-play 166, instruction override 167; 41–42 prompts per template). Columns: `id, family, goal, prompt, subcategory, template_id, split`. |
| `adversarial_wild.csv` | 500 | The same 500 goals inserted into 42 public in-the-wild jailbreak templates (JailbreakHub, Shen et al. 2024; templates of at most 300 words containing a request placeholder). |
| `direct_harmful_advbench.csv` | 500 | The same goals, unframed: the topic-matched direct-harmful control. |
| `framing_templates.csv` | 12 | The 12 framing templates (4 per framing). |
| `xstest_assignment.csv` | 450 | Every XSTest prompt (250 safe, 200 unsafe) with its original type, our sub-category (factual 100, technical 50, conversational 50, creative 50, explicit harmful 200) and split. |
| `ambiguous_prompts.csv` | 150 | Harmless but ambiguous prompts written by the first author (creative, philosophical, vague; 50 each). Not independently annotated; sheets for two annotators are in `annotation/`. |
| `multiturn_sequences.csv` | 3,600 turns | Constructed multi-turn sequences per seed (42, 43, 44): abrupt and gradual escalations and harmless controls, with the index of the harmful turn. |
| `template_control/benign_direct.csv` | 500 | Harmless Alpaca instructions (Taori et al., 2023; no input, 5-25 words, not questions, harm-word filter), unframed. |
| `template_control/benign_framed.csv` | 500 | The same instructions in the 12 framing templates, AdvBench framing proportions. |
| `template_control/benign_wild.csv` | 500 | The same instructions in the 42 in-the-wild templates. |
| `template_control/jbb_benign_framed.csv`, `jbb_harmful_framed.csv` | 100 + 100 | JailbreakBench benign and harmful behaviours (topic-matched by the benchmark) in identical template assignments. |

`family` is the AdvBench goal id; the framed, in-the-wild and direct versions of a goal share it.

## Split

Prompts are assigned to calibration (30 %) or test (70 %) by a hash of their family, so all
versions of one goal fall in the same split (AdvBench-derived sets: 140 calibration / 360 test;
XSTest: 137 / 313). Every threshold, probe and probability model in the paper is fitted on the
calibration split only.

## Results files (`release/results/`)

* `per_prompt_<backbone>.csv`: one row per main-corpus prompt with split, persona signals
  (D~, pairwise similarities), the decisions of every method, every baseline score and the
  output judgements (Llama Guard 3-1B response judge and refusal-prefix rule).
* `attacks/attack_*.csv`: every query of every attack run (seed, attacker, defence, goal, query
  index, variant, decision, score). `results/tables/exp4_asr_validity.csv` gives ASR with the
  raw, lenient and strict success definitions used in the paper.
* `template_control_scores_<backbone>.csv`: per-prompt probe scores of the template control (Exp. 13);
  `template_control_llama_guard_scores.csv`: Llama Guard 3-1B on the same sets;
  `template_aware_probe_scores_<backbone>.csv`: the template-aware probe (Exp. 14, exploratory).
* `tables/`: every table behind the paper; `<backbone>_thresholds*.json`: calibrated thresholds.

## Licences of the sources

AdvBench (Zou et al., 2023; MIT), Alpaca instructions (Taori et al., 2023; CC BY-NC 4.0), XSTest (Röttger et al., 2024; CC BY 4.0), in-the-wild jailbreak
prompts (Shen et al., 2024; MIT), HarmBench and JailbreakBench behaviours (MIT). Derived files keep
the licence of their source; the templates, ambiguous prompts and multi-turn sequences written for
this study are released under CC BY 4.0. Check the upstream repositories before redistribution.

## Ethics

The harmful requests are taken from public safety benchmarks and contain no new harmful
instructions. The framing templates are standard role-play, hypothetical and override patterns and
are released to support defence research. Generated model responses are not released, only their
judgements, and no human subjects were involved.
