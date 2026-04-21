"""
ECHO FORGE — Normalization Layer
Ensures feature vectors are invariant across price scale, volatility
differences, and timeframe compression/stretching.

Methods:
- Z-score normalization (per-feature, using population stats)
- Min-max scaling to [0, 1]
- Rank normalization (percentile-based)
- Time warp normalization (DTW alignment)
"""

from enum import Enum
from typing import Optional

import numpy as np
from scipy.interpolate import interp1d


class NormMethod(str, Enum):
    ZSCORE = "zscore"
    MINMAX = "minmax"
    RANK = "rank"
    COMBINED = "combined"


class FeatureNormalizer:
    """
    Normalizes feature vectors to ensure cross-asset and cross-timeframe
    comparability.

    Supports fitting to a population of vectors (for stable z-scoring)
    or stateless normalization.
    """

    def __init__(self, method: NormMethod = NormMethod.COMBINED):
        self.method = method
        self._fitted = False
        self._means: Optional[np.ndarray] = None
        self._stds: Optional[np.ndarray] = None
        self._mins: Optional[np.ndarray] = None
        self._maxs: Optional[np.ndarray] = None

    def fit(self, vectors: np.ndarray) -> "FeatureNormalizer":
        """
        Fit normalization parameters to a population of feature vectors.

        Parameters
        ----------
        vectors : np.ndarray
            Shape (N, D) where N is number of patterns and D is feature dim.
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        self._means = np.mean(vectors, axis=0)
        self._stds = np.std(vectors, axis=0) + 1e-10
        self._mins = np.min(vectors, axis=0)
        self._maxs = np.max(vectors, axis=0)
        self._fitted = True
        return self

    def transform(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalize a single feature vector or batch.

        If not fitted, falls back to self-normalization.
        """
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False

        if self.method == NormMethod.ZSCORE:
            result = self._zscore(vector)
        elif self.method == NormMethod.MINMAX:
            result = self._minmax(vector)
        elif self.method == NormMethod.RANK:
            result = self._rank(vector)
        elif self.method == NormMethod.COMBINED:
            result = self._combined(vector)
        else:
            result = vector

        if squeeze:
            return result.squeeze(0)
        return result

    def fit_transform(self, vectors: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(vectors)
        return self.transform(vectors)

    def _zscore(self, vectors: np.ndarray) -> np.ndarray:
        if self._fitted:
            return (vectors - self._means) / self._stds
        # Self-normalize
        means = np.mean(vectors, axis=0)
        stds = np.std(vectors, axis=0) + 1e-10
        return (vectors - means) / stds

    def _minmax(self, vectors: np.ndarray) -> np.ndarray:
        if self._fitted:
            ranges = self._maxs - self._mins + 1e-10
            return (vectors - self._mins) / ranges
        mins = np.min(vectors, axis=0)
        maxs = np.max(vectors, axis=0)
        ranges = maxs - mins + 1e-10
        return (vectors - mins) / ranges

    def _rank(self, vectors: np.ndarray) -> np.ndarray:
        """Percentile-rank normalization: each feature mapped to [0, 1]."""
        from scipy.stats import rankdata
        result = np.zeros_like(vectors)
        for i in range(vectors.shape[0]):
            for j in range(vectors.shape[1]):
                if self._fitted and vectors.shape[0] == 1:
                    # Can't rank a single vector meaningfully without population
                    # Fall back to z-score then sigmoid
                    z = (vectors[i, j] - self._means[j]) / self._stds[j]
                    result[i, j] = 1.0 / (1.0 + np.exp(-z))
                else:
                    result[:, j] = rankdata(vectors[:, j]) / (vectors.shape[0] + 1)
                    break  # rankdata handles the whole column
            if not self._fitted or vectors.shape[0] > 1:
                break

        if vectors.shape[0] > 1:
            for j in range(vectors.shape[1]):
                result[:, j] = rankdata(vectors[:, j]) / (vectors.shape[0] + 1)
        elif self._fitted:
            for j in range(vectors.shape[1]):
                z = (vectors[0, j] - self._means[j]) / self._stds[j]
                result[0, j] = 1.0 / (1.0 + np.exp(-z))

        return result

    def _combined(self, vectors: np.ndarray) -> np.ndarray:
        """
        Combined normalization: z-score followed by sigmoid squashing.
        Produces values in (0, 1) with good separation.
        """
        z = self._zscore(vectors)
        # Sigmoid squash for bounded output
        return 1.0 / (1.0 + np.exp(-z))


class TimeWarpNormalizer:
    """
    Handles time-scale invariance by resampling feature sequences
    to a canonical length before comparison.

    This allows matching patterns that played out over different
    numbers of bars (e.g., a 30-bar compression matching a 100-bar one).
    """

    def __init__(self, canonical_length: int = 100):
        self.canonical_length = canonical_length

    def resample(self, series: np.ndarray) -> np.ndarray:
        """
        Resample a 1D time series to the canonical length using
        linear interpolation.
        """
        if len(series) == self.canonical_length:
            return series

        x_original = np.linspace(0, 1, len(series))
        x_canonical = np.linspace(0, 1, self.canonical_length)

        interpolator = interp1d(x_original, series, kind="linear")
        return interpolator(x_canonical)

    def resample_ohlcv(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """
        Resample an entire OHLCV DataFrame to canonical length.
        Each column is independently resampled.
        """
        import pandas as pd

        resampled = {}
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                resampled[col] = self.resample(df[col].values)

        return pd.DataFrame(resampled)
