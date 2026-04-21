"""
ECHO FORGE — Core Module Tests
Tests for window extraction, feature building, and normalization.
"""

import numpy as np
import pandas as pd
import pytest

from app.core.window_extractor import WindowExtractor, PatternWindowData
from app.core.feature_builder import FeatureBuilder
from app.core.normalization import FeatureNormalizer, NormMethod, TimeWarpNormalizer


def _make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate test OHLCV data."""
    np.random.seed(seed)
    base = 100
    returns = np.random.normal(0.001, 0.02, n)
    closes = base * np.exp(np.cumsum(returns))
    highs = closes * (1 + np.abs(np.random.normal(0, 0.008, n)))
    lows = closes * (1 - np.abs(np.random.normal(0, 0.008, n)))
    opens = np.roll(closes, 1)
    opens[0] = base
    volumes = np.random.lognormal(10, 0.5, n)
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volumes,
    }, index=pd.date_range("2024-01-01", periods=n, freq="h"))


class TestWindowExtractor:
    def test_extract_returns_windows(self):
        df = _make_ohlcv(200)
        extractor = WindowExtractor()
        windows = extractor.extract(df, "TEST", "1h", window_size=60)
        assert len(windows) > 0
        assert all(isinstance(w, PatternWindowData) for w in windows)

    def test_window_sizes_correct(self):
        df = _make_ohlcv(200)
        extractor = WindowExtractor()
        windows = extractor.extract(df, "TEST", "1h", window_size=40)
        for w in windows:
            assert len(w.ohlcv) == 40

    def test_forward_outcomes_populated(self):
        df = _make_ohlcv(200)
        extractor = WindowExtractor()
        windows = extractor.extract(df, "TEST", "1h", window_size=60)
        for w in windows:
            assert isinstance(w.forward_return, float)
            assert isinstance(w.max_drawdown, float)
            # max_drawdown is typically <= 0, but can be 0 or slightly positive
            # due to floating point on flat data
            assert w.max_drawdown <= 0.01

    def test_extract_single(self):
        df = _make_ohlcv(100)
        extractor = WindowExtractor()
        window = extractor.extract_single(df, "TEST", "1h", window_size=60)
        assert window is not None
        assert len(window.ohlcv) == 60

    def test_insufficient_data_returns_empty(self):
        df = _make_ohlcv(30)
        extractor = WindowExtractor()
        windows = extractor.extract(df, "TEST", "1h", window_size=60)
        assert len(windows) == 0


class TestFeatureBuilder:
    def test_build_returns_array(self):
        df = _make_ohlcv(100)
        builder = FeatureBuilder()
        features = builder.build(df)
        assert isinstance(features, np.ndarray)
        assert len(features) > 0

    def test_no_nans_in_output(self):
        df = _make_ohlcv(100)
        builder = FeatureBuilder()
        features = builder.build(df)
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))

    def test_consistent_dimension(self):
        builder = FeatureBuilder()
        f1 = builder.build(_make_ohlcv(100, seed=1))
        f2 = builder.build(_make_ohlcv(100, seed=2))
        assert len(f1) == len(f2)

    def test_different_data_different_features(self):
        builder = FeatureBuilder()
        f1 = builder.build(_make_ohlcv(100, seed=1))
        f2 = builder.build(_make_ohlcv(100, seed=99))
        assert not np.allclose(f1, f2)


class TestNormalization:
    def test_zscore_normalizer(self):
        vectors = np.random.randn(50, 20)
        norm = FeatureNormalizer(method=NormMethod.ZSCORE)
        result = norm.fit_transform(vectors)
        # After z-score, mean should be ~0, std ~1
        assert abs(np.mean(result)) < 0.1
        assert abs(np.std(result) - 1.0) < 0.1

    def test_combined_normalizer_bounded(self):
        vectors = np.random.randn(50, 20)
        norm = FeatureNormalizer(method=NormMethod.COMBINED)
        result = norm.fit_transform(vectors)
        # Combined (sigmoid) should be in (0, 1)
        assert np.all(result > 0)
        assert np.all(result < 1)

    def test_minmax_normalizer(self):
        vectors = np.random.randn(50, 20)
        norm = FeatureNormalizer(method=NormMethod.MINMAX)
        result = norm.fit_transform(vectors)
        assert np.all(result >= -0.01)  # small tolerance
        assert np.all(result <= 1.01)

    def test_time_warp_resampling(self):
        warper = TimeWarpNormalizer(canonical_length=50)
        series = np.random.randn(30)
        resampled = warper.resample(series)
        assert len(resampled) == 50

    def test_time_warp_identity(self):
        warper = TimeWarpNormalizer(canonical_length=50)
        series = np.random.randn(50)
        resampled = warper.resample(series)
        assert np.allclose(series, resampled)
