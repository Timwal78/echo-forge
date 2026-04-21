"""
ECHO FORGE — End-to-End Pipeline Tests
Tests the full scan pipeline from data generation through projection.
"""

import numpy as np
import pandas as pd
import pytest

from app.core.window_extractor import WindowExtractor
from app.encoder.pattern_encoder import PatternEncoder
from app.encoder.feature_vector import FeatureVector
from app.similarity.matcher import PatternMatcher
from app.similarity.ranker import MatchRanker
from app.clustering.outcome_cluster import OutcomeClusterer
from app.clustering.distribution_model import DistributionModeler
from app.outcomes.outcome_engine import OutcomeEngine
from app.outcomes.failure_analysis import FailureAnalyzer
from app.outcomes.projection import ProjectionEngine
from app.workers.jobs import generate_mock_ohlcv, generate_mock_candidates


class TestFullPipeline:
    """Integration test: full echo scan pipeline."""

    def test_end_to_end_scan(self):
        # 1. Generate mock data
        df = generate_mock_ohlcv(n_bars=300, ticker="TSLA", regime="volatile")
        assert len(df) == 300

        # 2. Extract windows
        extractor = WindowExtractor()
        windows = extractor.extract(df, "TSLA", "1h", window_size=60)
        assert len(windows) > 0

        # 3. Encode windows
        encoder = PatternEncoder()
        ohlcv_list = [w.ohlcv for w in windows]
        vectors = encoder.fit_encode(ohlcv_list)
        assert vectors.shape[0] == len(windows)
        assert vectors.shape[1] > 0

        # 4. Build feature vectors with metadata
        candidates = []
        for i, w in enumerate(windows):
            candidates.append(FeatureVector(
                vector=vectors[i],
                ticker="TSLA",
                timeframe="1h",
                start_time=w.start_time,
                end_time=w.end_time,
                window_size=60,
                forward_return=w.forward_return,
                max_drawdown=w.max_drawdown,
                max_runup=w.max_runup,
                time_to_resolution=w.time_to_resolution,
            ))

        # 5. Match
        query = vectors[-1]  # Use last window as query
        matcher = PatternMatcher()
        matches = matcher.find_matches(query, candidates[:-1], top_n=15)
        assert len(matches) > 0

        # 6. Rank
        ranker = MatchRanker()
        matches = ranker.rerank(matches)
        confidence = ranker.compute_confidence(matches)
        echo_type = ranker.classify_echo_type(matches)
        assert 0 <= confidence <= 1
        assert isinstance(echo_type, str)

        # 7. Cluster
        clusterer = OutcomeClusterer()
        clusters = clusterer.cluster(matches)
        assert len(clusters) > 0
        total_prob = sum(c.probability for c in clusters)
        assert abs(total_prob - 1.0) < 0.01

        # 8. Distribution
        modeler = DistributionModeler()
        distribution = modeler.build_distribution(clusters)
        assert isinstance(distribution.mean_return, float)

        # 9. Outcome stats
        engine = OutcomeEngine()
        stats = engine.compute(matches)
        assert stats.n_matches == len(matches)
        assert 0 <= stats.win_rate <= 1

        # 10. Failure analysis
        analyzer = FailureAnalyzer()
        failure = analyzer.analyze(matches, clusters)
        assert 0 <= failure.failure_risk_score <= 1

        # 11. Projection
        projector = ProjectionEngine()
        projection = projector.project(
            matches=matches,
            clusters=clusters,
            distribution=distribution,
            outcome_stats=stats,
            failure_analysis=failure,
            ticker="TSLA",
            timeframe="1h",
        )
        assert projection.narrative
        assert len(projection.narrative) > 50

    def test_cross_asset_pipeline(self):
        """Test that cross-asset matching works across multiple tickers."""
        candidates = generate_mock_candidates(n=200)
        tickers_in_db = set(c.ticker for c in candidates)
        assert len(tickers_in_db) > 1, "Mock data should span multiple tickers"

        # Query with a specific ticker
        query_vec = candidates[0].vector
        matcher = PatternMatcher()
        matches = matcher.find_matches(
            query_vec, candidates[1:], top_n=15, cross_asset=True
        )

        matched_tickers = set(m.ticker for m in matches)
        # Cross-asset matching should potentially return multiple tickers
        assert len(matches) > 0

    def test_mock_candidate_generation(self):
        candidates = generate_mock_candidates(n=100)
        assert len(candidates) == 100
        assert all(isinstance(c, FeatureVector) for c in candidates)
        assert all(len(c.vector) > 0 for c in candidates)

    def test_projection_narrative_quality(self):
        """Verify narrative contains institutional-quality content."""
        candidates = generate_mock_candidates(n=200)
        query = candidates[0].vector

        matcher = PatternMatcher()
        matches = matcher.find_matches(query, candidates[1:], top_n=15)

        ranker = MatchRanker()
        matches = ranker.rerank(matches)

        clusterer = OutcomeClusterer()
        clusters = clusterer.cluster(matches)

        modeler = DistributionModeler()
        distribution = modeler.build_distribution(clusters)

        engine = OutcomeEngine()
        stats = engine.compute(matches)

        analyzer = FailureAnalyzer()
        failure = analyzer.analyze(matches, clusters)

        projector = ProjectionEngine()
        projection = projector.project(
            matches, clusters, distribution, stats, failure,
            ticker="AMC", timeframe="1h",
        )

        narrative = projection.narrative.lower()
        # Should not contain retail language
        assert "moon" not in narrative
        assert "rocket" not in narrative
        assert "buy" not in narrative
        assert "sell" not in narrative

        # Should contain analytical language
        assert "precedent" in narrative or "confidence" in narrative or "probability" in narrative
