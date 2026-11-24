# Datasets

No prompts are bundled with the code. Download the public benchmarks below,
place them in `data/raw/` (or anywhere else), and set the matching
`datasets.<name>.path` entries in `config/config.yaml`. Column names are also
configurable there.

| Config key | Paper role | Source | Expected file / columns |
|---|---|---|---|
| `xstest` | Harmless (safe split, n=250) and explicit harmful (unsafe contrast split, n=200) | https://github.com/paul-rottger/xstest (`xstest_prompts.csv`) | `id, prompt, type, label` |
| `advbench` | Adversarial jailbreak category | https://github.com/llm-attacks/llm-attacks (`data/advbench/harmful_behaviors.csv`) | `goal, target` |
| `harmbench` | Exp. 7 benchmark (n=320) | https://github.com/centerforaisafety/HarmBench (`data/behavior_datasets/harmbench_behaviors_text_test.csv`) | `Behavior, FunctionalCategory, SemanticCategory, ContextString, BehaviorID` |
| `jailbreakbench` | Exp. 7 benchmark (n=100) | https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors (`data/harmful-behaviors.csv`) | `Index, Goal, Target, Behavior, Category, Source` |
| `ambiguous` | Exp. 8 harmless-but-ambiguous set (n=150) | Your annotated set | `prompt, subcategory` (creative / philosophical / vague) |
| `jailbreak_templates` | RQ4 attacker templates (DAN, AIM, role-play overrides) | e.g. https://github.com/verazuo/jailbreak_llms (`data/prompts/jailbreak_prompts_2023_12_25.csv`; set `template_column: prompt`) | a text column with the placeholder set in `datasets.jailbreak_templates.placeholder`; rows without it are skipped |

Supported formats: `.csv`, `.tsv`, `.jsonl`, `.json`, `.parquet`.

## Notes

* **AdvBench framings.** `prepare_data.py` wraps each of 500 seeded AdvBench goals in exactly one of
  12 framing templates (hypothetical 167, role-play 166, instruction override 167; four templates per
  framing) and writes `advbench_framed.csv`; the label is the template that produced the prompt
  (`subcategory`, `template_id`), so no heuristic is involved. The unframed goals are in
  `advbench_direct.csv` and the same goals in 42 public JailbreakHub templates in `advbench_wild.csv`.
* **AdvBench size.** The public `harmful_behaviors.csv` has 520 rows; 500 are sampled with seed 0.
* **Template-control sets.** `benign_direct.csv`, `benign_framed.csv`, `benign_wild.csv` (500 Alpaca
  instructions in the same templates) and `jbb_benign_framed.csv` / `jbb_harmful_framed.csv` are built
  by `build_benign_controls()`; see `PREREGISTRATION_exp13.md` and `PREREGISTRATION_exp14.md`.
* **XSTest sub-categories.** `datasets.xstest.safe_subcategories` maps each
  XSTest `type` to the four harmless sub-categories used in the paper (factual 100, technical 50,
  conversational 50, creative 50); the assignment of every prompt is released as
  `release/data/xstest_assignment.csv`.
