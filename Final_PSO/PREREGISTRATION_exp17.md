# Pre-registration: Experiment 17, coverage of the diverse probe

Written on 2026-11-21 (UTC), after the Exp. 16 results were known and before Exp. 17 was run.
Code: `experiments/exp17_coverage.py`. Reported as exploratory, because its design follows Exp. 16.

## Question
Exp. 16 showed that a probe trained on diverse sources detects held-out AdvBench well and resists the
covered black-box attackers (strict score-aware ASR@10 0.04), but fails on sources it has never seen
(GCG 0.11-0.12, PAIR 0.15, OR-Bench toxic 0.27; OR-Bench-Hard flagged 0.71). We read this as a
coverage limit. Exp. 17 tests that reading: how many examples of a new source does the probe need,
how does it handle attack operators it has not seen, and does the pattern hold on all four backbones?

## Analyses (fixed now)
1. **Few-shot coverage curve.** For each held-out group (H3 HarmBench, H4 JailbreakBench incl. PAIR and
   GCG, H5 OR-Bench toxic, B3 OR-Bench-Hard, B4 JBB benign), train on all other groups plus the prompts
   of k calibration families of the held-out group, k = 0, 5, 10, 25, 50 (capped at the number of
   calibration families), five random draws for k > 0; evaluate on the held-out group's test families.
   Layer as in Exp. 16; C fixed to the k = 0 fold's value; threshold by the lambda = 0.5 rule.
2. **Held-out operators.** The augmented diverse probe of Exp. 16 is evaluated on 150 test-family
   AdvBench goals and 150 test-family Alpaca instructions rewritten with seven operators absent from its
   training data: base64, ROT13, leetspeak, character insertion, word splitting, payload splitting and
   five new hand-written templates. A JailbreakHub template and the plain goal are reported as
   in-coverage references.
3. **Near-duplicate control.** Harmful test families whose prompt has token Jaccard >= 0.6 with a
   harmful calibration prompt are listed; in-distribution results are reported with and without them.
4. **Replication** of 1-3 on Llama-2 (primary), Llama-3, Mistral and Qwen2.5.
5. **Offline** (from saved per-prompt scores): probe-OR-Llama-Guard combination and family-level
   bootstrap intervals (2,000 resamples).

## Criteria (fixed now)
* Coverage: mean detection over the test families of H3-H5 at k = 25 >= 0.80, and flag rate on B3 and
  B4 at k = 25 <= 0.30 (Llama-2).
* Operators: detection >= 0.80 with benign flag rate <= 0.20 on at least four of the seven held-out
  operators (Llama-2).
* Near-duplicates: removing them changes no in-distribution detection or flag rate by more than 0.03.
* Replication: augmented-probe test AUC >= 0.95 on all four backbones.
* Combination: probe OR Llama Guard detects at least as much as Llama Guard on every independent
  harmful set, with a false-positive rate on XSTest-safe no more than 0.05 above Llama Guard's.
Llama-2 is primary; all results are reported whether or not the criteria are met.
