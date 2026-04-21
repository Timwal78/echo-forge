"""
ECHO FORGE — Window Extraction Engine
Extracts rolling analysis windows from historical OHLCV data and computes
forward outcomes for each window.

Design principles:
- Multi-timeframe support via configurable bar semantics
- Overlapping windows with configurable stride
- Forward outcome labeling for every extracted window
- Cross-asset agnostic: operates on normalized DataFrames
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from app.config import WindowConfig


@dataclass
class PatternWindowData:
    """Raw data container for a single extracted window plus its forward outcome."""
    ticker: str
    timeframe: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    ohlcv: pd.DataFrame          # The window's OHLCV bars
    forward_ohlcv: pd.DataFrame  # Bars immediately after the window (outcome horizon)
    forward_return: float
    max_drawdown: float
    max_runup: float
    time_to_resolution: int      # bars until max move completes


class WindowExtractor:
    """
    Extracts overlapping analysis windows from a historical OHLCV DataFrame.

    Each window is a contiguous block of bars. For each window, the extractor
    also captures a forward-looking horizon used for outcome labeling.
    """

    def __init__(self, config: Optional[WindowConfig] = None):
        self.config = config or WindowConfig()

    def extract(
        self,
        df: pd.DataFrame,
        ticker: str,
        timeframe: str,
        window_size: Optional[int] = None,
        overlap_ratio: Optional[float] = None,
        forward_horizon: Optional[int] = None,
    ) -> list[PatternWindowData]:
        """
        Extract all valid windows from the input DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: open, high, low, close, volume
            Index should be DatetimeIndex or contain a 'timestamp' column.
        ticker : str
            Instrument identifier.
        timeframe : str
            Bar timeframe label (e.g., '1h', '4h', '1d').
        window_size : int, optional
            Number of bars per window. Defaults to config.
        overlap_ratio : float, optional
            Fraction of window to overlap. 0.5 = 50% overlap.
        forward_horizon : int, optional
            Bars to capture after each window for outcome labeling.

        Returns
        -------
        list[PatternWindowData]
        """
        window_size = window_size or self.config.default_window_size
        overlap_ratio = overlap_ratio if overlap_ratio is not None else self.config.overlap_ratio
        forward_horizon = forward_horizon or self.config.forward_horizon

        df = self._validate_and_prepare(df)
        stride = max(1, int(window_size * (1 - overlap_ratio)))
        total_bars = len(df)
        min_required = window_size + forward_horizon

        if total_bars < min_required:
            return []

        windows = []
        start = 0
        while start + min_required <= total_bars:
            window_df = df.iloc[start: start + window_size].copy()
            forward_df = df.iloc[start + window_size: start + window_size + forward_horizon].copy()

            outcome = self._compute_forward_outcome(window_df, forward_df)

            windows.append(PatternWindowData(
                ticker=ticker,
                timeframe=timeframe,
                start_time=window_df.index[0],
                end_time=window_df.index[-1],
                ohlcv=window_df,
                forward_ohlcv=forward_df,
                forward_return=outcome["forward_return"],
                max_drawdown=outcome["max_drawdown"],
                max_runup=outcome["max_runup"],
                time_to_resolution=outcome["time_to_resolution"],
            ))
            start += stride

        return windows

    def extract_single(
        self,
        df: pd.DataFrame,
        ticker: str,
        timeframe: str,
        window_size: Optional[int] = None,
    ) -> Optional[PatternWindowData]:
        """
        Extract the most recent window from the data (no forward outcome).
        Used for live scanning where no future data exists.
        """
        window_size = window_size or self.config.default_window_size
        df = self._validate_and_prepare(df)

        if len(df) < window_size:
            return None

        window_df = df.iloc[-window_size:].copy()
        return PatternWindowData(
            ticker=ticker,
            timeframe=timeframe,
            start_time=window_df.index[0],
            end_time=window_df.index[-1],
            ohlcv=window_df,
            forward_ohlcv=pd.DataFrame(),
            forward_return=0.0,
            max_drawdown=0.0,
            max_runup=0.0,
            time_to_resolution=0,
        )

    def _validate_and_prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has required columns and a proper DatetimeIndex."""
        required = {"open", "high", "low", "close", "volume"}
        cols = set(df.columns.str.lower())
        missing = required - cols
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df.copy()
        df.columns = df.columns.str.lower()

        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp")
            else:
                df.index = pd.to_datetime(df.index)

        df = df.sort_index()
        return df

    @staticmethod
    def _compute_forward_outcome(
        window_df: pd.DataFrame, forward_df: pd.DataFrame
    ) -> dict:
        """
        Compute outcome metrics for the forward horizon after a window.

        Returns dict with: forward_return, max_drawdown, max_runup, time_to_resolution
        """
        if forward_df.empty:
            return {
                "forward_return": 0.0,
                "max_drawdown": 0.0,
                "max_runup": 0.0,
                "time_to_resolution": 0,
            }

        entry_price = window_df["close"].iloc[-1]
        if entry_price == 0:
            entry_price = 1e-8

        forward_closes = forward_df["close"].values
        returns = (forward_closes - entry_price) / entry_price

        cumulative_max = np.maximum.accumulate(forward_closes)
        cumulative_min = np.minimum.accumulate(forward_closes)

        drawdowns = (cumulative_min - entry_price) / entry_price
        runups = (cumulative_max - entry_price) / entry_price

        max_dd = float(np.min(drawdowns))
        max_ru = float(np.max(runups))
        final_return = float(returns[-1])

        # Time to resolution: bar index where max absolute move occurs
        abs_moves = np.abs(returns)
        time_to_res = int(np.argmax(abs_moves)) + 1

        return {
            "forward_return": final_return,
            "max_drawdown": max_dd,
            "max_runup": max_ru,
            "time_to_resolution": time_to_res,
        }
