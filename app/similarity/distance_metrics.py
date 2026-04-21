"""
ECHO FORGE — Distance Metrics
Multiple similarity/distance functions for comparing structural fingerprints.

Each metric captures a different aspect of pattern similarity:
- Cosine: directional alignment in feature space
- Euclidean: absolute magnitude difference
- DTW: time-warped structural alignment
- Composite: weighted combination
"""

import numpy as np
from scipy.spatial.distance import cosine as cosine_dist, euclidean
from scipy.signal import correlate


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two feature vectors.
    Returns value in [0, 1] where 1 = identical direction.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def euclidean_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Euclidean distance converted to similarity score in [0, 1].
    Uses exponential decay: sim = exp(-dist / dim).
    """
    dist = euclidean(a, b)
    dim = len(a)
    # Scale by sqrt(dim) so the metric is dimensionality-invariant
    return float(np.exp(-dist / (np.sqrt(dim) + 1e-10)))


def dtw_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Simplified Dynamic Time Warping similarity.
    Uses a fast approximation via cross-correlation for high-dimensional vectors.

    For true DTW on raw time series, use dtw_timeseries().
    """
    # Normalize both vectors
    a_norm = (a - np.mean(a)) / (np.std(a) + 1e-10)
    b_norm = (b - np.mean(b)) / (np.std(b) + 1e-10)

    # Cross-correlation based similarity
    corr = correlate(a_norm, b_norm, mode="full")
    max_corr = np.max(corr)
    # Normalize by vector lengths
    norm_factor = np.sqrt(np.sum(a_norm**2) * np.sum(b_norm**2)) + 1e-10
    sim = max_corr / norm_factor

    return float(np.clip(sim, 0.0, 1.0))


def dtw_timeseries(s: np.ndarray, t: np.ndarray) -> float:
    """
    True DTW distance for 1D time series.
    Returns a similarity score in [0, 1].

    Uses O(n*m) dynamic programming — suitable for moderate-length series.
    """
    n, m = len(s), len(t)
    if n == 0 or m == 0:
        return 0.0

    # Normalize series
    s = (s - np.mean(s)) / (np.std(s) + 1e-10)
    t = (t - np.mean(t)) / (np.std(t) + 1e-10)

    # DTW cost matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (s[i - 1] - t[j - 1]) ** 2
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1],
            )

    distance = np.sqrt(dtw_matrix[n, m]) / max(n, m)
    return float(np.exp(-distance))


def composite_similarity(
    a: np.ndarray,
    b: np.ndarray,
    cosine_weight: float = 0.5,
    euclidean_weight: float = 0.3,
    dtw_weight: float = 0.2,
) -> dict:
    """
    Compute weighted composite similarity from multiple metrics.

    Returns
    -------
    dict with keys: composite, cosine, euclidean, dtw
    """
    cos_sim = cosine_similarity(a, b)
    euc_sim = euclidean_similarity(a, b)
    dtw_sim = dtw_similarity(a, b)

    composite = (
        cosine_weight * cos_sim
        + euclidean_weight * euc_sim
        + dtw_weight * dtw_sim
    )

    return {
        "composite": float(composite),
        "cosine": cos_sim,
        "euclidean": euc_sim,
        "dtw": dtw_sim,
    }
