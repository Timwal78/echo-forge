"""
ECHO FORGE — Feature Engineering Pipeline
Converts raw OHLCV windows into high-dimensional structural fingerprints.

This is the intellectual core of the system. Every feature captures a
*structural* property of price action — not a visual pattern.

Feature groups:
1. Volatility regime vector
2. Compression / expansion ratios
3. Directional bias slope
4. Momentum curvature (second derivative of returns)
5. Volume acceleration / deceleration
6. Time symmetry metrics
7. Breakout / failure behavior encoding
8. Relative position within range
"""

import math
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from app.config import FeatureConfig


class FeatureBuilder:
    """
    Transforms an OHLCV window DataFrame into a fixed-length feature vector
    representing the structural fingerprint of that window.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()

    def build(self, ohlcv: pd.DataFrame) -> np.ndarray:
        """
        Build the full feature vector for a single window.

        Parameters
        ----------
        ohlcv : pd.DataFrame
            Window data with columns: open, high, low, close, volume.

        Returns
        -------
        np.ndarray
            Fixed-length feature vector.
        """
        features = []

        features.extend(self._volatility_regime(ohlcv))
        features.extend(self._compression_expansion(ohlcv))
        features.extend(self._directional_bias(ohlcv))
        features.extend(self._momentum_curvature(ohlcv))
        features.extend(self._volume_dynamics(ohlcv))
        features.extend(self._time_symmetry(ohlcv))
        features.extend(self._breakout_failure(ohlcv))
        features.extend(self._relative_positioning(ohlcv))
        features.extend(self._cycle_vectors(ohlcv))

        vec = np.array(features, dtype=np.float64)
        # Replace any NaN/Inf with 0
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        return vec

    def _volatility_regime(self, df: pd.DataFrame) -> list[float]:
        """
        Multi-scale volatility profile.
        Captures the *regime* of volatility (contracting, expanding, stable)
        across multiple lookback windows.
        """
        closes = df["close"].values
        log_returns = np.diff(np.log(np.maximum(closes, 1e-10)))

        features = []
        for lb in self.config.volatility_lookbacks:
            if len(log_returns) < lb:
                features.extend([0.0, 0.0, 0.0])
                continue

            # Rolling realized volatility at this scale
            rolling_vol = pd.Series(log_returns).rolling(lb).std().dropna().values
            if len(rolling_vol) < 2:
                features.extend([0.0, 0.0, 0.0])
                continue

            # Current vol level (last value, z-scored against window)
            mean_vol = np.mean(rolling_vol)
            std_vol = np.std(rolling_vol) + 1e-10
            current_z = (rolling_vol[-1] - mean_vol) / std_vol

            # Vol trend: slope of volatility over the window
            x = np.arange(len(rolling_vol))
            slope, _, _, _, _ = sp_stats.linregress(x, rolling_vol)
            vol_slope_normed = slope / (mean_vol + 1e-10)

            # Vol of vol: second-order instability
            vol_of_vol = np.std(np.diff(rolling_vol)) / (mean_vol + 1e-10)

            features.extend([current_z, vol_slope_normed, vol_of_vol])

        return features

    def _compression_expansion(self, df: pd.DataFrame) -> list[float]:
        """
        Measure compression (range contraction) and expansion (range widening).
        Uses ATR-based and Bollinger-band-width-based metrics.
        """
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        true_ranges = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )

        if len(true_ranges) < 10:
            return [0.0] * 6

        # ATR ratio: recent ATR vs full-window ATR
        atr_full = np.mean(true_ranges)
        atr_recent = np.mean(true_ranges[-10:])
        atr_ratio = atr_recent / (atr_full + 1e-10)

        # Compression cycles: count of consecutive contracting ATR bars
        atr_rolling = pd.Series(true_ranges).rolling(5).mean().dropna().values
        if len(atr_rolling) < 2:
            return [atr_ratio, 0.0, 0.0, 0.0, 0.0, 0.0]

        atr_diff = np.diff(atr_rolling)
        compression_bars = int(np.sum(atr_diff < 0))
        expansion_bars = int(np.sum(atr_diff > 0))
        total = compression_bars + expansion_bars + 1e-10
        compression_ratio = compression_bars / total
        expansion_ratio = expansion_bars / total

        # Bollinger bandwidth proxy
        rolling_std = pd.Series(closes).rolling(20).std().dropna().values
        rolling_mean = pd.Series(closes).rolling(20).mean().dropna().values
        if len(rolling_std) > 0 and len(rolling_mean) > 0:
            bw = rolling_std / (np.abs(rolling_mean) + 1e-10)
            bw_current = bw[-1]
            bw_slope = (bw[-1] - bw[0]) / (len(bw) + 1e-10)
        else:
            bw_current = 0.0
            bw_slope = 0.0

        return [atr_ratio, compression_ratio, expansion_ratio,
                float(bw_current), float(bw_slope), float(compression_bars)]

    def _directional_bias(self, df: pd.DataFrame) -> list[float]:
        """
        Slope and curvature of the directional trend within the window.
        Uses linear regression on closes and on a smoothed series.
        """
        closes = df["close"].values
        n = len(closes)
        if n < 5:
            return [0.0] * 4

        # Normalize closes to [0, 1] for scale invariance
        c_min, c_max = np.min(closes), np.max(closes)
        rng = c_max - c_min + 1e-10
        normed = (closes - c_min) / rng

        x = np.arange(n)
        slope, intercept, r_value, _, _ = sp_stats.linregress(x, normed)

        # R² measures how linear the trend is
        r_squared = r_value ** 2

        # Curvature: fit a quadratic and extract the coefficient
        coeffs = np.polyfit(x, normed, 2)
        curvature = coeffs[0]  # x² coefficient

        # Endpoint deviation from trend
        predicted_end = slope * (n - 1) + intercept
        endpoint_dev = normed[-1] - predicted_end

        return [slope, r_squared, curvature, endpoint_dev]

    def _momentum_curvature(self, df: pd.DataFrame) -> list[float]:
        """
        Second derivative of returns — captures *acceleration* of momentum.
        Multi-scale analysis.
        """
        closes = df["close"].values
        log_returns = np.diff(np.log(np.maximum(closes, 1e-10)))

        features = []
        for lb in self.config.momentum_lookbacks:
            if len(log_returns) < lb + 2:
                features.extend([0.0, 0.0, 0.0])
                continue

            # Smoothed momentum at this scale
            momentum = pd.Series(log_returns).rolling(lb).mean().dropna().values
            if len(momentum) < 3:
                features.extend([0.0, 0.0, 0.0])
                continue

            # First derivative of momentum (acceleration)
            accel = np.diff(momentum)
            # Second derivative (jerk / curvature of momentum)
            jerk = np.diff(accel)

            # Summary statistics
            mean_accel = float(np.mean(accel))
            accel_trend = float(accel[-1] - accel[0]) / (len(accel) + 1e-10)
            mean_jerk = float(np.mean(jerk)) if len(jerk) > 0 else 0.0

            features.extend([mean_accel, accel_trend, mean_jerk])

        return features

    def _volume_dynamics(self, df: pd.DataFrame) -> list[float]:
        """
        Volume acceleration/deceleration and its relationship to price action.
        """
        volumes = df["volume"].values.astype(np.float64)
        closes = df["close"].values

        if len(volumes) < 10 or np.sum(volumes) == 0:
            return [0.0] * 6

        # Normalize volume
        mean_vol = np.mean(volumes) + 1e-10
        normed_vol = volumes / mean_vol

        # Volume trend
        x = np.arange(len(normed_vol))
        vol_slope, _, vol_r, _, _ = sp_stats.linregress(x, normed_vol)

        # Volume acceleration
        vol_diff = np.diff(normed_vol)
        vol_accel = np.mean(np.diff(vol_diff)) if len(vol_diff) > 1 else 0.0

        # Price-volume correlation
        log_returns = np.diff(np.log(np.maximum(closes, 1e-10)))
        min_len = min(len(log_returns), len(volumes) - 1)
        if min_len > 2:
            pv_corr = float(np.corrcoef(
                np.abs(log_returns[:min_len]),
                volumes[1:min_len + 1]
            )[0, 1])
        else:
            pv_corr = 0.0

        # Climactic volume detection
        vol_zscore_last = (normed_vol[-1] - 1.0) / (np.std(normed_vol) + 1e-10)

        # Volume concentration: what fraction of total volume is in the last 20%
        last_20pct = max(1, len(volumes) // 5)
        vol_concentration = np.sum(volumes[-last_20pct:]) / (np.sum(volumes) + 1e-10)

        return [
            float(vol_slope),
            float(vol_r ** 2),
            float(vol_accel),
            float(np.nan_to_num(pv_corr)),
            float(vol_zscore_last),
            float(vol_concentration),
        ]

    def _time_symmetry(self, df: pd.DataFrame) -> list[float]:
        """
        Measure temporal symmetry of the pattern.
        A perfectly symmetric window has equal behavior in its first and second half.
        """
        closes = df["close"].values
        n = len(closes)
        if n < 6:
            return [0.0] * 4

        mid = n // 2
        first_half = closes[:mid]
        second_half = closes[mid:2 * mid]  # same length

        # Range symmetry
        range_first = (np.max(first_half) - np.min(first_half))
        range_second = (np.max(second_half) - np.min(second_half))
        range_ratio = range_second / (range_first + 1e-10)

        # Volatility symmetry
        vol_first = np.std(np.diff(first_half))
        vol_second = np.std(np.diff(second_half))
        vol_ratio = vol_second / (vol_first + 1e-10)

        # Directional symmetry: do both halves trend the same way?
        slope_first = (first_half[-1] - first_half[0]) / (len(first_half) + 1e-10)
        slope_second = (second_half[-1] - second_half[0]) / (len(second_half) + 1e-10)
        # +1 = same direction, -1 = opposite
        if abs(slope_first) < 1e-12 or abs(slope_second) < 1e-12:
            dir_symmetry = 0.0
        else:
            dir_symmetry = np.sign(slope_first) * np.sign(slope_second)

        # Correlation between first and second half (after normalizing)
        if len(first_half) == len(second_half) and len(first_half) > 1:
            f_norm = (first_half - np.mean(first_half)) / (np.std(first_half) + 1e-10)
            s_norm = (second_half - np.mean(second_half)) / (np.std(second_half) + 1e-10)
            half_corr = float(np.corrcoef(f_norm, s_norm)[0, 1])
        else:
            half_corr = 0.0

        return [
            float(range_ratio),
            float(vol_ratio),
            float(dir_symmetry),
            float(np.nan_to_num(half_corr)),
        ]

    def _breakout_failure(self, df: pd.DataFrame) -> list[float]:
        """
        Encode breakout and failure behavior within the window.
        Detects false breakouts, failed breakdowns, and range violations.
        """
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        n = len(closes)

        if n < 10:
            return [0.0] * 5

        # Define range from first 50% of window
        range_end = n // 2
        range_high = np.max(highs[:range_end])
        range_low = np.min(lows[:range_end])
        range_size = range_high - range_low + 1e-10

        # Count upside breakouts (close above range_high)
        upside_breaks = np.sum(closes[range_end:] > range_high)
        # Count downside breakouts (close below range_low)
        downside_breaks = np.sum(closes[range_end:] < range_low)

        remaining = n - range_end + 1e-10
        upside_ratio = upside_breaks / remaining
        downside_ratio = downside_breaks / remaining

        # Failed breakout detection: broke out then returned inside range
        second_half_closes = closes[range_end:]
        broke_up = np.any(second_half_closes > range_high)
        broke_down = np.any(second_half_closes < range_low)
        ended_inside = range_low <= closes[-1] <= range_high

        failed_breakout_score = 0.0
        if (broke_up or broke_down) and ended_inside:
            failed_breakout_score = 1.0
        elif broke_up and closes[-1] < range_high:
            failed_breakout_score = 0.5
        elif broke_down and closes[-1] > range_low:
            failed_breakout_score = 0.5

        # Maximum excursion beyond range (normalized)
        max_upside_excursion = (np.max(highs[range_end:]) - range_high) / range_size
        max_downside_excursion = (range_low - np.min(lows[range_end:])) / range_size

        return [
            float(upside_ratio),
            float(downside_ratio),
            float(failed_breakout_score),
            float(max(0, max_upside_excursion)),
            float(max(0, max_downside_excursion)),
        ]

    def _relative_positioning(self, df: pd.DataFrame) -> list[float]:
        """
        Where the current price sits relative to the window's range
        and key statistical levels.
        """
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values

        if len(closes) < 2:
            return [0.0] * 4

        window_high = np.max(highs)
        window_low = np.min(lows)
        window_range = window_high - window_low + 1e-10

        # Position within range [0, 1]
        position = (closes[-1] - window_low) / window_range

        # Distance from VWAP proxy (volume-weighted mean close)
        volumes = df["volume"].values.astype(np.float64)
        total_vol = np.sum(volumes) + 1e-10
        vwap = np.sum(closes * volumes) / total_vol
        vwap_distance = (closes[-1] - vwap) / window_range

        # Percentile rank of last close within all closes in window
        percentile = float(sp_stats.percentileofscore(closes, closes[-1]) / 100.0)

        # Mean reversion potential: distance from window mean (z-scored)
        mean_close = np.mean(closes)
        std_close = np.std(closes) + 1e-10
        mean_rev_z = (closes[-1] - mean_close) / std_close

        return [float(position), float(vwap_distance), percentile, float(mean_rev_z)]

    def _cycle_vectors(self, df: pd.DataFrame) -> list[float]:
        """
        Encodes proximity and phase relative to specific market cycles.
        Default: 147-day cycle (Meme Stock Intelligence).
        
        Uses sinusoidal encoding to capture the cyclical nature without 
        step-function discontinuities.
        """
        if "timestamp" not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            return [0.0, 0.0]

        # Use the latest timestamp in the window
        current_time = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else df["timestamp"].iloc[-1]
        
        # Reference date from config
        ref_date = datetime.strptime(self.config.cycle_reference_date, "%Y-%m-%d").replace(tzinfo=current_time.tzinfo)
        
        # Calculate days since reference
        delta_days = (current_time - ref_date).total_seconds() / (24 * 3600)
        
        # Period from config
        period = self.config.cycle_period_days
        
        # Normalized phase [0, 2*pi]
        phase = (2 * math.pi * delta_days) / period
        
        # Sin/Cos encoding (captures 'where' in the 147-day cycle we are)
        return [math.sin(phase), math.cos(phase)]
