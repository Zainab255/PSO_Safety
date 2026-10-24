"""Checks that the implementation matches the paper's equations."""

import math

import numpy as np
import pytest
import torch

from config import load_config
from core.decision import WrapperDecision, geometric_decision, probabilistic_decision, threshold_grid
from core.msds import l2_normalise, msds, normalise_score, pairwise_signals, softmax_entropy
from core.templating import fill_template
from evaluation import metrics as M
from wrappers.base import Conversation
from wrappers.history_wrapper import HistoryWrapper
from wrappers.llm_judge_wrapper import parse_verdict
from wrappers.self_critique_wrapper import parse_critique


def unit(*rows):
    return l2_normalise(torch.tensor(rows, dtype=torch.float32))


def test_msds_identical_embeddings_is_zero():
    e = unit([1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.5, 1.0, 1.5])
    assert msds(e) == pytest.approx(0.0, abs=1e-6)


def test_msds_mutually_orthogonal_is_one():
    e = unit([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    assert msds(e) == pytest.approx(1.0)


def test_msds_is_mean_pairwise_cosine_distance():
    rng = np.random.default_rng(0)
    e = l2_normalise(torch.tensor(rng.normal(size=(3, 16)), dtype=torch.float32))
    s = pairwise_signals(e)
    expected = ((1 - s["s_rb"]) + (1 - s["s_rj"]) + (1 - s["s_bj"])) / 3   # Eq. (9), K = 3
    assert msds(e) == pytest.approx(expected, abs=1e-6)


def test_proposition_1_backward_compatibility():
    e = unit([1.0, 2.0, 0.0], [0.0, 1.0, 3.0])
    s_rb = float(e[0] @ e[1])
    assert msds(e) == pytest.approx(1 - s_rb, abs=1e-6)


def test_normalisation_half():
    assert normalise_score(1.2, "half") == pytest.approx(0.6)
    assert normalise_score(1.2, "none") == pytest.approx(1.2)


def test_softmax_entropy_is_normalised():
    e = unit([1.0, 0.0], [0.0, 1.0], [1.0, 1.0])
    assert 0.0 <= softmax_entropy(e, 1.0) <= 1.0


def test_geometric_decision_eq18():
    assert geometric_decision(0.34, 0.35, 0.65) == WrapperDecision.ALLOW
    assert geometric_decision(0.35, 0.35, 0.65) == WrapperDecision.REQUERY
    assert geometric_decision(0.65, 0.35, 0.65) == WrapperDecision.REQUERY
    assert geometric_decision(0.66, 0.35, 0.65) == WrapperDecision.BLOCK


def test_probabilistic_decision_eq17():
    assert probabilistic_decision(0.10, 0.25, 0.72) == WrapperDecision.ALLOW
    assert probabilistic_decision(0.25, 0.25, 0.72) == WrapperDecision.REQUERY
    assert probabilistic_decision(0.80, 0.25, 0.72) == WrapperDecision.BLOCK


def test_threshold_grid_matches_paper_ranges():
    assert len(threshold_grid({"start": 0.20, "stop": 0.55, "step": 0.05})) == 8
    assert len(threshold_grid({"start": 0.40, "stop": 0.75, "step": 0.05})) == 8


def test_temporal_entropy_drift_eqs_2_3():
    ted = HistoryWrapper({"instruction_override": [], "persona_hijacking": [], "context_poisoning": []},
                         {"window": 3, "delta": 0.25}, tau_s=0.35)
    conv = Conversation()
    for score in (0.10, 0.12, 0.15):
        assert ted.escalation_alert(conv, score)[0] is False   # fewer than W prior turns
        conv.add("turn", score)
    alert, phi = ted.escalation_alert(conv, 0.50)               # Phi = D(4) - D(1) = 0.40
    assert phi == pytest.approx(0.40) and alert
    alert, _ = ted.escalation_alert(conv, 0.30)                 # drift 0.20 < delta
    assert not alert


def test_ted_requires_score_above_tau_s():
    ted = HistoryWrapper({"instruction_override": [], "persona_hijacking": [], "context_poisoning": []},
                         {"window": 1, "delta": 0.05}, tau_s=0.35)
    conv = Conversation()
    conv.add("turn", 0.0)
    assert not ted.escalation_alert(conv, 0.30)[0]


def test_fill_template_keeps_json_braces():
    out = fill_template('Reply {"verdict": "SAFE"}. Input: {prompt}', prompt="{response}")
    assert out == 'Reply {"verdict": "SAFE"}. Input: {response}'


def test_judge_and_critique_parsers():
    assert parse_verdict('{"verdict": "UNSAFE"}') == "UNSAFE"
    assert parse_verdict("The input is SAFE.") == "SAFE"
    assert parse_verdict("no idea") == "UNKNOWN"
    assert parse_critique('{"safe": false, "reason": "x"}') is False
    assert parse_critique('{"safe": true, "reason": "x"}') is True
    assert parse_critique("hmm") is None


def test_classification_metrics():
    y = np.array([1, 1, 0, 0])
    pred = np.array([True, False, True, False])
    assert M.tpr(y, pred) == 0.5 and M.fpr(y, pred) == 0.5
    assert M.f_beta(y, pred, 2) == pytest.approx(5 * 1 / (5 * 1 + 4 * 1 + 1))
    assert M.asr_at_k([1, 3, np.nan, 2], 2) == 0.5
    assert M.ece(np.array([0, 1]), np.array([0.0, 1.0]), 10) == pytest.approx(0.0)


def test_pipeline_scores_rank_phase1_blocks_first():
    s = M.pipeline_scores(np.array([0.2, np.nan, 0.9]), ["orchestrator", "keyword", "orchestrator"])
    assert s[1] > s[2] > s[0]


def test_config_personas_are_the_paper_appendix():
    personas = load_config()["pso"]["personas"]
    assert personas["red"].startswith("You are a safety evaluator operating in a red team capacity.")
    assert personas["blue"].startswith("You are a safety evaluator operating in a blue team capacity.")
    assert personas["judge"].startswith("You are an impartial safety adjudicator.")
    assert not math.isnan(load_config()["pso"]["thresholds"]["tau_s"])
