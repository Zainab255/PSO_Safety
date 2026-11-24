"""Multi-View Safety Disagreement Score and related geometric signals.

Implements Section III-E/F of the paper:

    e_hat_k = e_k / ||e_k||_2                                   (Eq. 8)
    s_ij    = e_hat_i^T e_hat_j                                  (Def. 5)
    D_s^(K) = 2 / (K(K-1)) * sum_{i<j} (1 - s_ij)                (Eq. 9)
    D~_s^(K) = D_s^(K) / 2                                       (normalised)

plus the two comparison signals reported in the paper: the full-matrix
softmax entropy (Section III-F-1) and the Embedding Variance baseline
(Table 6, ‡).
"""

from __future__ import annotations

import math
from typing import Dict, Sequence

import torch
import torch.nn.functional as F

# Order in which persona embeddings are stacked throughout the code base.
PERSONA_ORDER = ("red", "blue", "judge")


def l2_normalise(embeddings: torch.Tensor) -> torch.Tensor:
    """Eq. (8): project each row onto the unit hypersphere."""
    return F.normalize(embeddings.float(), p=2, dim=-1)


def similarity_matrix(unit_embeddings: torch.Tensor) -> torch.Tensor:
    """Def. 5: K x K matrix of cosine similarities s_ij."""
    return unit_embeddings @ unit_embeddings.T


def msds(unit_embeddings: torch.Tensor) -> float:
    """Eq. (9): mean cosine distance over the K(K-1)/2 unique persona pairs."""
    k = unit_embeddings.shape[0]
    if k < 2:
        raise ValueError("MSDS needs at least two persona embeddings.")
    sims = similarity_matrix(unit_embeddings)
    iu = torch.triu_indices(k, k, offset=1)
    return float((1.0 - sims[iu[0], iu[1]]).mean())


def normalise_score(d: float, mode: str) -> float:
    """Map D_s^(K) to the decision scale D~_s^(K).

    ``half`` is the paper's definition D~ = D / 2 in [0, 1]; ``none`` keeps D.
    """
    if mode == "half":
        return d / 2.0
    if mode == "none":
        return d
    raise ValueError(f"Unknown normalisation mode '{mode}' (expected 'half' or 'none').")


def pairwise_signals(unit_embeddings: torch.Tensor, order: Sequence[str] = PERSONA_ORDER) -> Dict[str, float]:
    """Eqs. (10)-(12): S_RB, S_RJ, S_BJ."""
    idx = {name: i for i, name in enumerate(order)}
    sims = similarity_matrix(unit_embeddings)
    return {
        "s_rb": float(sims[idx["red"], idx["blue"]]),
        "s_rj": float(sims[idx["red"], idx["judge"]]),
        "s_bj": float(sims[idx["blue"], idx["judge"]]),
    }


def softmax_entropy(unit_embeddings: torch.Tensor, temperature: float) -> float:
    """Shannon entropy of a softmax over the full K x K similarity matrix.

    Normalised by log(K^2) so the value lies in [0, 1]. This is the
    diagonal-confounded alternative the paper argues against (Section III-F-1).
    """
    k = unit_embeddings.shape[0]
    logits = similarity_matrix(unit_embeddings).flatten() / temperature
    p = torch.softmax(logits, dim=0)
    h = -(p * torch.log(p)).sum()
    return float(h / math.log(k * k))


def embedding_variance(raw_embeddings: torch.Tensor) -> float:
    """Embedding Variance baseline: variance across personas, averaged over dimensions."""
    return float(raw_embeddings.float().var(dim=0, unbiased=False).mean())


def score_embeddings(raw_embeddings: torch.Tensor, normalisation: str, entropy_temperature: float) -> Dict[str, float]:
    """All geometric signals for one prompt from its [K, d] persona embeddings."""
    unit = l2_normalise(raw_embeddings)
    d_raw = msds(unit)
    out = {
        "msds_raw": d_raw,
        "disagreement_score": normalise_score(d_raw, normalisation),
        "softmax_entropy": softmax_entropy(unit, entropy_temperature),
        "embedding_variance": embedding_variance(raw_embeddings),
    }
    out.update(pairwise_signals(unit))
    return out
