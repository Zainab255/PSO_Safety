# Ambiguity-set annotation (two independent annotators)

The 150 prompts in `data/raw/ambiguous_prompts.csv` were written for this study by the first
author. To report inter-annotator agreement, two people other than the author label them
**independently** (no discussion until both are done):

1. Annotator A fills `annotator_A.csv`, annotator B fills `annotator_B.csv` (same items, shuffled,
   no category shown). Do not open `key_do_not_share_with_annotators.csv`.
2. For every item answer:
   * **Q1 harmless intent** — `yes` if the most plausible intent is harmless, `no` otherwise.
   * **Q2 could be read as harmful** — `yes` if a reasonable reader could also see a harmful reading.
   * **Q3 category** — `creative`, `philosophical`, `vague` or `other`.
3. Run `python annotation/agreement.py annotation/annotator_A.csv annotation/annotator_B.csv`.
   It prints Cohen's kappa and raw agreement per question and against the intended categories,
   and lists items where either annotator answered Q1 = `no` (to be removed or reported).
