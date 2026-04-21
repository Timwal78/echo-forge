"""
ECHO FORGE — Matching Engine
Searches historical pattern database for structurally similar echoes.

Supports:
- Cross-asset matching (compare current ticker against all instruments)
- Time-scale invariance (match patterns across different timeframes)
- Configurable top-N retrieval
- Approximate nearest neighbor for large databases
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.config import SimilarityConfig
from app.encoder.feature_vector import FeatureVector
from app.similarity.distance_metrics import composite_similarity


@dataclass
class EchoMatch:
    """A single matched historical pattern."""
    feature_vector: FeatureVector
    similarity_scores: dict  # composite, cosine, euclidean, dtw
    rank: int

    @property
    def composite_score(self) -> float:
        return self.similarity_scores.get("composite", 0.0)

    @property
    def ticker(self) -> str:
        return self.feature_vector.ticker

    @property
    def timeframe(self) -> str:
        return self.feature_vector.timeframe

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "timeframe": self.timeframe,
            "start_time": self.feature_vector.start_time.isoformat(),
            "end_time": self.feature_vector.end_time.isoformat(),
            "similarity": round(self.composite_score, 4),
            "cosine": round(self.similarity_scores.get("cosine", 0.0), 4),
            "euclidean": round(self.similarity_scores.get("euclidean", 0.0), 4),
            "forward_return": round(self.feature_vector.forward_return, 4),
            "max_drawdown": round(self.feature_vector.max_drawdown, 4),
            "time_to_resolution": self.feature_vector.time_to_resolution,
            "outcome_label": self.feature_vector.outcome_label,
            "rank": self.rank,
        }


class PatternMatcher:
    """
    Core matching engine. Given a query feature vector, finds the
    most structurally similar historical patterns.
    """

    def __init__(self, config: Optional[SimilarityConfig] = None):
        self.config = config or SimilarityConfig()

    def find_matches(
        self,
        query: np.ndarray,
        candidates: list[FeatureVector],
        top_n: Optional[int] = None,
        exclude_ticker: Optional[str] = None,
        cross_asset: Optional[bool] = None,
    ) -> list[EchoMatch]:
        """
        Find top-N structurally similar patterns.

        Parameters
        ----------
        query : np.ndarray
            Feature vector of the current pattern.
        candidates : list[FeatureVector]
            Historical pattern database.
        top_n : int, optional
            Number of matches to return.
        exclude_ticker : str, optional
            Exclude self-matches (same ticker, overlapping time).
        cross_asset : bool, optional
            If False, only match within same ticker.

        Returns
        -------
        list[EchoMatch]
            Ranked matches, highest similarity first.
        """
        top_n = top_n or self.config.top_n_matches
        cross_asset = cross_asset if cross_asset is not None else self.config.cross_asset_enabled

        scored = []
        for candidate in candidates:
            # Filter by cross-asset setting
            if not cross_asset and exclude_ticker and candidate.ticker != exclude_ticker:
                continue

            scores = composite_similarity(
                query,
                candidate.vector,
                cosine_weight=self.config.cosine_weight,
                euclidean_weight=self.config.euclidean_weight,
                dtw_weight=self.config.dtw_weight,
            )

            if scores["composite"] >= self.config.min_similarity_threshold:
                scored.append((candidate, scores))

        # Sort by composite similarity descending
        scored.sort(key=lambda x: x[1]["composite"], reverse=True)

        # Build ranked results
        matches = []
        for rank, (fv, scores) in enumerate(scored[:top_n], start=1):
            matches.append(EchoMatch(
                feature_vector=fv,
                similarity_scores=scores,
                rank=rank,
            ))

        return matches

    def find_matches_batch(
        self,
        query: np.ndarray,
        candidate_matrix: np.ndarray,
        candidate_metadata: list[FeatureVector],
        top_n: Optional[int] = None,
    ) -> list[EchoMatch]:
        """
        Vectorized matching against a pre-computed matrix.
        Much faster for large databases.

        Parameters
        ----------
        query : np.ndarray
            Shape (D,) query vector.
        candidate_matrix : np.ndarray
            Shape (N, D) matrix of candidate vectors.
        candidate_metadata : list[FeatureVector]
            Metadata for each row in the matrix.
        """
        top_n = top_n or self.config.top_n_matches

        # Vectorized cosine similarity
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        norms = np.linalg.norm(candidate_matrix, axis=1, keepdims=True) + 1e-10
        normed = candidate_matrix / norms
        cosine_sims = normed @ query_norm

        # Vectorized euclidean similarity
        diffs = candidate_matrix - query
        dists = np.linalg.norm(diffs, axis=1)
        dim = query.shape[0]
        euc_sims = np.exp(-dists / (np.sqrt(dim) + 1e-10))

        # Composite (skip DTW for batch — too expensive)
        adjusted_cosine_w = self.config.cosine_weight + self.config.dtw_weight * 0.5
        adjusted_euc_w = self.config.euclidean_weight + self.config.dtw_weight * 0.5
        composites = adjusted_cosine_w * cosine_sims + adjusted_euc_w * euc_sims

        # Get top-N indices
        threshold_mask = composites >= self.config.min_similarity_threshold
        valid_indices = np.where(threshold_mask)[0]

        if len(valid_indices) == 0:
            return []

        valid_scores = composites[valid_indices]
        sorted_order = np.argsort(-valid_scores)[:top_n]
        top_indices = valid_indices[sorted_order]

        matches = []
        for rank, idx in enumerate(top_indices, start=1):
            fv = candidate_metadata[idx]
            matches.append(EchoMatch(
                feature_vector=fv,
                similarity_scores={
                    "composite": float(composites[idx]),
                    "cosine": float(cosine_sims[idx]),
                    "euclidean": float(euc_sims[idx]),
                    "dtw": 0.0,  # skipped in batch mode
                },
                rank=rank,
            ))

        return matches
