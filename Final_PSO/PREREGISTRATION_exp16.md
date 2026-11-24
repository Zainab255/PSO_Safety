# Pre-registration: Experiment 16, diverse-data probe (leave-one-source-out and adversarial augmentation)

Written on 2026-11-21 (UTC), after the Exp. 15 results were known and before Exp. 16 was run.
Code: `experiments/exp16_diverse_probe.py`. Reported as exploratory, because its design follows Exp. 15.

## Question
Exp. 15 showed that the template-aware probe does not generalise to attack-generated jailbreaks
(GCG 0-4 % detected), flags 54-86 % of OR-Bench-Hard prompts and is evaded by a score-aware
attacker (strict ASR@10 0.60). Does training the same kind of probe on diverse sources fix this?

## Source groups (all prompts that share a goal stay in one group)
Harmful: H1 AdvBench (framed, in-the-wild, direct); H2 XSTest unsafe; H3 HarmBench;
H4 JailbreakBench goals (direct, framed, PAIR, GCG x2); H5 OR-Bench toxic.
Harmless: B1 XSTest safe; B2 Alpaca (direct, framed, in-the-wild); B3 OR-Bench-Hard; B4 JBB benign.

## Analyses (fixed now)
1. **Leave-one-source-out (LOSO).** For each group, fit the probe on all other groups and evaluate on
   the held-out group: detection rate for a harmful group, flag rate for a harmless group. Layer is
   chosen once by family-grouped CV on the pooled calibration families and reused in every fold;
   C by grouped CV within each fold; threshold by the lambda = 0.5 rule on the training data.
   GCG prompts (part of H4) are also reported separately.
2. **Operator augmentation and re-attack (Llama-2).** Add to the training data the attack operators of
   Exp. 4 (in-the-wild templates, framings, evaluator appeal, persona mimicry, benign padding and
   prefix) applied to calibration-family AdvBench goals *and*, equally, to calibration-family Alpaca
   instructions, three variants each. Then rerun the three attackers of Exp. 15 (50 test-family goals,
   seeds 42-44, ten queries).

## Criteria (fixed now)
* LOSO: on every held-out harmful group detection >= 0.80, on every held-out harmless group flag rate
  <= 0.20, and GCG detection >= 0.80.
* Re-attack: score-aware strict ASR@10 <= 0.20 on Llama-2.
Llama-2 is primary; all results are reported whether or not the criteria are met.
