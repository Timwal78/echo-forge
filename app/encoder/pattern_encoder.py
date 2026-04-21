"""
ECHO FORGE — Pattern Encoder
Orchestrates the full encoding pipeline: raw OHLCV → structural fingerprint.

This is the main interface between raw market data and the similarity engine.
It combines feature building, normalization, and optional dimensionality
reduction into a single encoding step.
"""

from typing import Optional

import numpy as np
import pandas as pd

from app.config import FeatureConfig
from app.core.feature_builder import FeatureBuilder
from app.core.normalization import FeatureNormalizer, NormMethod, TimeWarpNormalizer


class PatternEncoder:
    """
    Encodes an OHLCV window into a normalized, fixed-length feature vector.

    Pipeline:
    1. (Optional) Time-warp resample to canonical length
    2. Extract structural features via FeatureBuilder
    3. Normalize via FeatureNormalizer
    4. (Optional) Dimensionality reduction via PCA
    """

    def __init__(
        self,
        feature_config: Optional[FeatureConfig] = None,
        norm_method: NormMethod = NormMethod.COMBINED,
        use_time_warp: bool = True,
        canonical_length: int = 100,
    ):
        self.feature_builder = FeatureBuilder(config=feature_config)
        self.normalizer = FeatureNormalizer(method=norm_method)
        self.use_time_warp = use_time_warp
        self.time_warper = TimeWarpNormalizer(canonical_length=canonical_length)
        self._pca = None
        self._fitted = False

    def encode(self, ohlcv: pd.DataFrame) -> np.ndarray:
        """
        Encode a single OHLCV window into a feature vector.

        Parameters
        ----------
        ohlcv : pd.DataFrame
            Raw OHLCV data for the window.

        Returns
        -------
        np.ndarray
            Normalized feature vector.
        """
        if self.use_time_warp:
            ohlcv = self.time_warper.resample_ohlcv(ohlcv)

        raw_features = self.feature_builder.build(ohlcv)

        if self._fitted:
            normalized = self.normalizer.transform(raw_features)
        else:
            normalized = raw_features  # pre-fit not available yet

        if self._pca is not None:
            normalized = self._pca.transform(normalized.reshape(1, -1)).flatten()

        return normalized

    def encode_batch(self, windows: list[pd.DataFrame]) -> np.ndarray:
        """
        Encode a batch of OHLCV windows.

        Parameters
        ----------
        windows : list[pd.DataFrame]
            List of OHLCV DataFrames.

        Returns
        -------
        np.ndarray
            Shape (N, D) matrix of feature vectors.
        """
        raw_vectors = []
        for ohlcv in windows:
            if self.use_time_warp:
                ohlcv = self.time_warper.resample_ohlcv(ohlcv)
            raw_vectors.append(self.feature_builder.build(ohlcv))

        matrix = np.vstack(raw_vectors)
        return matrix

    def fit(self, windows: list[pd.DataFrame]) -> "PatternEncoder":
        """
        Fit the normalizer (and optionally PCA) on a population of windows.
        Should be called once with a representative historical dataset.
        """
        matrix = self.encode_batch(windows)
        self.normalizer.fit(matrix)
        self._fitted = True
        return self

    def fit_encode(self, windows: list[pd.DataFrame]) -> np.ndarray:
        """Fit on the population and return normalized vectors."""
        matrix = self.encode_batch(windows)
        normalized = self.normalizer.fit_transform(matrix)
        self._fitted = True
        return normalized

    def enable_pca(self, n_components: int, training_vectors: np.ndarray):
        """
        Enable PCA dimensionality reduction.
        Useful when feature vectors are very high-dimensional.
        """
        from sklearn.decomposition import PCA

        self._pca = PCA(n_components=n_components)
        self._pca.fit(training_vectors)

    @property
    def feature_dim(self) -> int:
        """Return the dimensionality of raw feature vectors."""
        # Build a dummy to measure
        dummy = pd.DataFrame({
            "open": np.random.randn(100),
            "high": np.random.randn(100),
            "low": np.random.randn(100),
            "close": np.random.randn(100),
            "volume": np.abs(np.random.randn(100)) * 1000,
        })
        return len(self.feature_builder.build(dummy))
