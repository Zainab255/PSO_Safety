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

* **AdvBench sub-categories.** The paper labels AdvBench prompts as
  hypothetical / roleplay / injection with a marker heuristic
  (`datasets.advbench.subcategory_heuristic`). The raw `goal` strings of the
  public file contain none of the markers, so every prompt falls into the
  default class. If you evaluated framed versions of the prompts, point
  `datasets.advbench.path` / `prompt_column` at that file, or supply your
  released labels through `datasets.advbench.category_column`.
* **AdvBench size.** The public `harmful_behaviors.csv` has 520 rows; the
  paper reports n=500.
* **XSTest sub-categories.** `datasets.xstest.safe_subcategories` maps each
  XSTest `type` to the sub-categories of Tables 5/7/16. Replace it with your
  released assignments.
