# Pre-registration: Experiment 13, template control for the single-view probe

Written on 2026-11-19 (UTC), before any hidden state of the sets below was computed.
Code: `experiments/exp13_template_control.py`; data: `data/prepare_data.py::build_benign_controls`.

## Question
The bare-prompt linear probe separates the main-corpus test prompts with AUC 0.997-0.999 (Exp. 12).
Its harmful prompts are templated and its harmless prompts are not, so the probe may detect
templates rather than harmful requests. We test this with harmless requests in the same templates.

## Sets (evaluation only; never used for fitting)
* `benign_direct`: 500 Alpaca instructions without input, 5-25 words, not questions, none containing a
  word from a fixed harm list (seeded sample).
* `benign_framed`: the same 500 in the 12 framing templates (same framing proportions as AdvBench).
* `benign_wild`: the same 500 in the 42 in-the-wild templates.
* `jbb_benign_framed` / `jbb_harmful_framed`: the 100 benign and 100 harmful JailbreakBench behaviours
  (topic-matched by the benchmark) in identical template assignments.

## Fixed analysis
The probe is fitted exactly as in Exp. 12 (main-corpus calibration split; layer and C by
family-grouped 5-fold CV; threshold by the lambda = 0.5 rule on the calibration split). Nothing is refitted
on the new sets. We report the CV-layer probe (primary) and the final-layer probe (secondary).

## Primary outcome and success criterion (decided now)
On each backbone, the probe **passes** the template control if both hold:
1. its flag rate on `benign_framed` at the calibrated threshold is at most 0.10, and
2. AUC(framed AdvBench test prompts vs `benign_framed`) is at least 0.90.
Secondary: the same quantities for the in-the-wild pair, the direct pair and the JailbreakBench pair.
Llama-2-7B-Chat is the primary backbone; the other three are reported in full whatever the result.
A failure is reported as a failure; thresholds, layers and sets are not changed after seeing results.
