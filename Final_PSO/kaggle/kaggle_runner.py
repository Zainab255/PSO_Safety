"""Resumable Kaggle driver for the PSO experiments.

Used by kaggle/PSO_Kaggle.ipynb. Each stage is a list of experiment modules; every
module caches its intermediate CSVs under outputs/runs/, so a stage interrupted by
the time budget resumes where it stopped in the next notebook version (attach the
previous version's output as an input and it is restored automatically).
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

STAGES = {
    # Llama-2: main corpus, calibration, detection, utility, latency, ablations, ambiguity,
    # multi-turn drift, prefix controls / layer-wise analysis.
    "A": ["run_main_evaluation", "exp6_predictive", "exp1_detection", "exp2_utility", "exp3_efficiency",
          "exp5_sensitivity", "exp8_ambiguity", "exp9_multiturn", "exp10_prefix_controls"],
    # Llama-2: single-view probe and published baselines; naive / paraphrase / adaptive / score-aware / PAIR attacks.
    "B": ["exp12_baselines", "exp4_robustness"],
    # Benchmarks for Llama-2, then Llama-3, Mistral and Qwen2.5 (main corpus + benchmarks + attacks),
    # with the prefix controls and single-view probes repeated on every backbone.
    "C": ["exp7_generalisation", ("exp10_prefix_controls", "llama2"),   # llama2: adds the in-the-wild set
          ("exp10_prefix_controls", "llama3"), ("exp12_baselines", "llama3"),
          ("exp10_prefix_controls", "mistral"), ("exp12_baselines", "mistral"),
          ("exp10_prefix_controls", "qwen"), ("exp12_baselines", "qwen")],
    # HarmBench-classifier re-judging of every response, then tables rebuilt from the caches,
    # per-prompt logs, summary, exploratory direction analysis and all figures from the logs.
    "D": ["rejudge_outputs", "exp1_detection", "exp2_utility", "exp5_sensitivity", "exp4_robustness",
          "export_logs", "summary", "exp11_direction"],
}

# Extra stages run in a second notebook in parallel with A/B (same caches, merged afterwards):
# persona analysis, prefix controls and single-view probes for the other backbones.
EXTRA = {
    # Exp. 13 template control (forward passes only; about 15 min per backbone on a T4).
    "T": [("exp13_template_control", b) for b in ("llama2", "llama3", "mistral", "qwen")],
    # Exp. 14 template-aware probe (exploratory; reuses the Exp. 13 hidden states when present).
    "P": [("exp14_template_aware_probe", b) for b in ("llama2", "llama3", "mistral", "qwen")],
    # Exp. 15 independent data (all backbones) and adaptive attacks on the template-aware probe (Llama-2).
    "Q": [("exp15_independent_and_adaptive", b) for b in ("llama2", "llama3", "mistral", "qwen")],
    # Exp. 17 coverage of the diverse probe (forward passes only; all backbones).
    "X": [("exp17_coverage", b) for b in ("llama2", "llama3", "mistral", "qwen")],
    # Benchmarks and attacks for all four backbones (Exp. 7) + the Llama-2 in-the-wild prefix controls;
    # run on a second Kaggle account in parallel with A/B/L.
    "C7": ["exp7_generalisation", ("exp10_prefix_controls", "llama2")],
    "L": [("exp12_baselines", "llama3"), ("exp10_prefix_controls", "llama3"),
          ("exp12_baselines", "mistral"), ("exp10_prefix_controls", "mistral"),
          ("exp12_baselines", "qwen"), ("exp10_prefix_controls", "qwen")],
}


def _free_backbone_cache(repo: Path, bb: str) -> None:
    """Delete a non-primary backbone's weights from the HF cache once its modules are done."""
    if os.environ.get("PSO_DELETE_MODEL_CACHE") != "1":
        return
    import yaml
    cfg = yaml.safe_load((repo / "config" / "config.yaml").read_text())
    if bb == cfg["backbones"]["primary"]:
        return
    name = cfg["backbones"]["models"][bb]["name_or_path"].replace("/", "--")
    hub = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"
    shutil.rmtree(hub / f"models--{name}", ignore_errors=True)


def restore_previous_outputs(repo: Path) -> None:
    """Copy outputs/ from any attached input (a previous version of this notebook)."""
    target = repo / "outputs"
    for z in sorted(glob.glob("/kaggle/input/**/pso_outputs.zip", recursive=True)):
        print(f"Restoring previous outputs from {z}")
        target.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(z, target)
    # Kaggle extracts uploaded zips, so the cache can also arrive as a folder holding runs/<backbone>/.
    for runs in sorted(glob.glob("/kaggle/input/**/runs", recursive=True)):
        if not any(Path(runs).glob("*/main")) and not any(Path(runs).glob("*/thresholds.json")):
            continue
        src = Path(runs).parent
        if src.resolve() == target.resolve():
            continue
        print(f"Restoring previous outputs from {src}")
        shutil.copytree(src, target, dirs_exist_ok=True)


def run_stage(repo: Path, stage: str, budget_hours: float, backbone: str = "llama2", fast: bool = False,
              deadline: float = None) -> bool:
    """Run one stage; True when every module finished."""
    deadline = deadline or time.time() + budget_hours * 3600
    logs = repo / "outputs" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    entries = {**STAGES, **EXTRA}[stage]
    for pos, entry in enumerate(entries):
        module, bb = (entry, backbone) if isinstance(entry, str) else entry
        # Stages C and L share per-backbone modules, so a module finished in either counts for both.
        key = "CL" if stage in ("C", "L", "C7") else stage
        done_flag = repo / "outputs" / "logs" / f"DONE_{key}_{module}_{bb}"
        if done_flag.exists():
            print(f"{module} ({bb}) already done.")
            continue
        remaining = deadline - time.time()
        if remaining < 15 * 60:
            print(f"Stopping before {module}: less than 15 minutes of budget left.")
            return False
        print(f"\n===== {module} [{bb}] (budget left {remaining / 3600:.1f} h) =====", flush=True)
        log = open(logs / f"{module}.log", "a")
        cmd = [sys.executable, "-m", f"experiments.{module}", "--backbone", bb]
        if fast:
            cmd += ["--config", str(repo / "config" / "config_fast.yaml")]
        proc = subprocess.Popen(cmd,
                                cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        # Enforce the budget even when the module prints nothing for hours (long attack loops):
        # a timer terminates it at the deadline; cached chunks are kept and the next version resumes.
        import threading
        timed_out = threading.Event()

        def _stop():
            timed_out.set()
            print("Time budget reached; stopping (cached progress is kept).", flush=True)
            proc.terminate()
        watchdog = threading.Timer(max(0.0, deadline - time.time()), _stop)
        watchdog.daemon = True
        watchdog.start()
        try:
            for line in proc.stdout:
                log.write(line)
                if any(k in line for k in ("[INFO] experiments", "responses for", "Calibrated", "Error",
                                           "Traceback", "OOM", "Wrote", "complete")):
                    print(line.rstrip()[:300], flush=True)
                if time.time() > deadline:
                    print("Time budget reached; stopping (cached progress is kept).", flush=True)
                    proc.terminate()
                    proc.wait(timeout=60)
                    return False
            proc.wait()
        finally:
            watchdog.cancel()
            log.close()
        if timed_out.is_set():
            return False
        if proc.returncode != 0:
            print(f"{module} failed with exit code {proc.returncode}; see outputs/logs/{module}.log")
            tail = (logs / f"{module}.log").read_text().splitlines()[-40:]
            print("\n".join(tail))
            return False
        done_flag.write_text(time.ctime())
        nxt = entries[pos + 1] if pos + 1 < len(entries) else None
        nxt_bb = None if nxt is None else (backbone if isinstance(nxt, str) else nxt[1])
        if nxt_bb != bb:
            _free_backbone_cache(repo, bb)
    print(f"\nStage {stage} complete.")
    (repo / "outputs" / f"STAGE_{stage}_DONE").write_text(time.ctime())
    return True


def status(repo: Path) -> None:
    """Print which cached results exist."""
    out = repo / "outputs"
    for d in sorted((out / "runs").glob("*/*")) if (out / "runs").exists() else []:
        print(f"{d.relative_to(out)}: {len(list(d.glob('*.csv')))} csv")
    tables = sorted(p.name for p in (out / "tables").glob("*.csv")) if (out / "tables").exists() else []
    print(f"tables: {len(tables)}")


def run_all(repo: Path, budget_hours: float, fast: bool = False) -> None:
    """Run stages A-D in order, skipping finished ones, until the time budget is used."""
    deadline = time.time() + budget_hours * 3600
    for stage in STAGES:
        if (repo / "outputs" / f"STAGE_{stage}_DONE").exists():
            print(f"Stage {stage} already complete.")
            continue
        if not run_stage(repo, stage, budget_hours, fast=fast, deadline=deadline):
            print(f"Stage {stage} not finished in this run; the next run resumes it.")
            return
    print("ALL STAGES COMPLETE")
