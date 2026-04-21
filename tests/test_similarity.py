"""
ECHO FORGE — Similarity Engine Tests
Tests for distance metrics, matching, and ranking.
"""

import numpy as np
import pytest
from datetime import datetime

from app.similarity.distance_metrics import (
    cosine_similarity, euclidean_similarity, dtw_similarity,
    composite_similarity,
)
from app.similarity.matcher import PatternMatcher, EchoMatch
from app.similarity.ranker import MatchRanker
from app.encoder.feature_vector import FeatureVector


def _make_feature_vector(
    ticker: str = "TEST",
    seed: int = 42,
    forward_return: float = 0.05,
    max_drawdown: float = -0.02,
) -> FeatureVector:
    np.random.seed(seed)
    return FeatureVector(
        vector=np.random.randn(42),
        ticker=ticker,
        timeframe="1h",
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 1, 2),
        window_size=60,
        forward_return=forward_return,
        max_drawdown=max_drawdown,
        max_runup=abs(forward_return) + 0.01,
        time_to_resolution=10,
        outcome_label="mixed_regime",
    )


class TestDistanceMetrics:
    def test_cosine_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_cosine_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_euclidean_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert euclidean_similarity(a, a) == pytest.approx(1.0)

    def test_euclidean_distant(self):
        a = np.zeros(10)
        b = np.ones(10) * 100
        sim = euclidean_similarity(a, b)
        assert 0 < sim < 0.5

    def test_dtw_similar_signals(self):
        a = np.sin(np.linspace(0, 2 * np.pi, 50))
        b = np.sin(np.linspace(0.1, 2 * np.pi + 0.1, 50))
        sim = dtw_similarity(a, b)
        assert sim > 0.5

    def test_composite_returns_all_keys(self):
        a = np.random.randn(20)
        b = np.random.randn(20)
        result = composite_similarity(a, b)
        assert "composite" in result
        assert "cosine" in result
        assert "euclidean" in result
        assert "dtw" in result

    def test_composite_bounded(self):
        a = np.random.randn(20)
        b = np.random.randn(20)
        result = composite_similarity(a, b)
        assert -0.1 <= result["composite"] <= 1.1


class TestMatcher:
    def test_find_matches(self):
        query = np.random.randn(42)
        candidates = [_make_feature_vector(seed=i) for i in range(50)]

        matcher = PatternMatcher()
        matches = matcher.find_matches(query, candidates, top_n=10)
        assert len(matches) <= 10
        assert all(isinstance(m, EchoMatch) for m in matches)

    def test_matches_sorted_by_similarity(self):
        query = np.random.randn(42)
        candidates = [_make_feature_vector(seed=i) for i in range(50)]

        matcher = PatternMatcher()
        matches = matcher.find_matches(query, candidates, top_n=10)

        scores = [m.composite_score for m in matches]
        assert scores == sorted(scores, reverse=True)

    def test_cross_asset_matching(self):
        # Use a candidate's own vector as query to guarantee matches
        candidates = []
        for i, ticker in enumerate(["TSLA", "AMC", "SPY", "BTC-USD"]):
            for j in range(10):
                candidates.append(_make_feature_vector(
                    ticker=ticker, seed=i * 100 + j
                ))

        # Query is similar to first candidate (add small noise)
        np.random.seed(0)
        query = candidates[0].vector + np.random.randn(42) * 0.1

        from app.config import SimilarityConfig
        config = SimilarityConfig(min_similarity_threshold=0.1)
        matcher = PatternMatcher(config=config)
        matches = matcher.find_matches(query, candidates, top_n=10, cross_asset=True)
        assert len(matches) > 0

    def test_batch_matching(self):
        query = np.random.randn(42)
        candidates = [_make_feature_vector(seed=i) for i in range(100)]
        matrix = np.vstack([c.vector for c in candidates])

        matcher = PatternMatcher()
        matches = matcher.find_matches_batch(query, matrix, candidates, top_n=10)
        assert len(matches) <= 10


class TestRanker:
    def test_rerank_preserves_count(self):
        query = np.random.randn(42)
        candidates = [_make_feature_vector(seed=i) for i in range(30)]
        matcher = PatternMatcher()
        matches = matcher.find_matches(query, candidates, top_n=10)

        ranker = MatchRanker()
        reranked = ranker.rerank(matches)
        assert len(reranked) == len(matches)

    def test_confidence_bounded(self):
        query = np.random.randn(42)
        candidates = [_make_feature_vector(seed=i) for i in range(30)]
        matcher = PatternMatcher()
        matches = matcher.find_matches(query, candidates, top_n=10)

        ranker = MatchRanker()
        conf = ranker.compute_confidence(matches)
        assert 0 <= conf <= 1

    def test_echo_type_classification(self):
        ranker = MatchRanker()
        # Test with no matches
        assert ranker.classify_echo_type([]) == "insufficient_data"
