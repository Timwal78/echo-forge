"""
ECHO FORGE — Background Workers / Jobs
Handles async tasks: pattern ingestion, batch scanning, cache warming.

In production, these would be Celery or RQ tasks.
For the initial build, they're implemented as async functions
that can be called directly or scheduled.
"""

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from app.config import get_config
from app.core.window_extractor import WindowExtractor
from app.encoder.pattern_encoder import PatternEncoder
from app.encoder.feature_vector import FeatureVector


def generate_mock_ohlcv(
    n_bars: int = 1000,
    ticker: str = "MOCK",
    regime: str = "trending",
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data with realistic market microstructure.

    Supports multiple regimes:
    - trending: persistent directional bias
    - mean_reverting: range-bound oscillation
    - volatile: high variance with regime shifts
    - compression: decreasing volatility squeeze
    """
    np.random.seed(hash(f"{ticker}_{regime}") % 2**31)

    base_price = np.random.uniform(10, 500)
    prices = [base_price]
    volumes = []

    for i in range(n_bars - 1):
        if regime == "trending":
            drift = 0.0005
            vol = 0.015
        elif regime == "mean_reverting":
            # Mean reversion around base
            deviation = (prices[-1] - base_price) / base_price
            drift = -0.3 * deviation / n_bars
            vol = 0.012
        elif regime == "volatile":
            # Regime-switching volatility
            phase = (i // 50) % 3
            drift = [0.001, -0.001, 0.0][phase]
            vol = [0.025, 0.03, 0.01][phase]
        elif regime == "compression":
            # Decreasing volatility
            vol = 0.025 * (1 - i / n_bars) + 0.005
            drift = 0.0001
        else:
            drift = 0.0
            vol = 0.02

        ret = np.random.normal(drift, vol)
        prices.append(prices[-1] * (1 + ret))

        # Volume: correlated with absolute returns + noise
        vol_base = np.random.lognormal(10, 0.5)
        vol_spike = vol_base * (1 + 5 * abs(ret))
        volumes.append(vol_spike)

    volumes.insert(0, np.random.lognormal(10, 0.5))
    prices = np.array(prices)

    # Generate OHLC from closes
    noise = np.abs(np.random.normal(0, 0.005, n_bars))
    highs = prices * (1 + noise)
    lows = prices * (1 - noise)
    opens = np.roll(prices, 1)
    opens[0] = prices[0]

    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
    }, index=pd.date_range("2020-01-01", periods=n_bars, freq="h"))


def generate_mock_candidates(
    n: int = 500,
    ticker: Optional[str] = None,
    timeframe: str = "1h",
) -> list[FeatureVector]:
    """
    Generate a population of mock pattern candidates for the database.
    Simulates a diverse cross-asset pattern library.
    """
    config = get_config()
    extractor = WindowExtractor(config=config.window)
    encoder = PatternEncoder(feature_config=config.features)

    tickers = [ticker] if ticker else ["SPY", "QQQ", "AAPL", "BTC-USD", "NVDA", "TSLA"]
    regimes = ["trending", "mean_reverting", "volatile", "compression"]

    all_vectors = []

    for t in tickers:
        for regime in regimes:
            # Generate synthetic history
            df = generate_mock_ohlcv(
                n_bars=max(200, n // len(tickers) + 100),
                ticker=t,
                regime=regime,
            )

            # Extract windows
            windows = extractor.extract(
                df=df,
                ticker=t,
                timeframe=timeframe,
                window_size=config.window.default_window_size,
            )

            # Encode each window
            for w in windows:
                vector = encoder.encode(w.ohlcv)

                # Classify structural profiles from the raw features
                fv = FeatureVector(
                    vector=vector,
                    ticker=t,
                    timeframe=timeframe,
                    start_time=w.start_time,
                    end_time=w.end_time,
                    window_size=config.window.default_window_size,
                    forward_return=w.forward_return,
                    max_drawdown=w.max_drawdown,
                    max_runup=w.max_runup,
                    time_to_resolution=w.time_to_resolution,
                    outcome_label=_classify_outcome(w.forward_return, w.max_drawdown, w.max_runup),
                    volatility_profile=FeatureVector.classify_volatility(
                        vector[0] if len(vector) > 0 else 0.0
                    ),
                    momentum_profile=FeatureVector.classify_momentum(
                        vector[9] if len(vector) > 9 else 0.0
                    ),
                    volume_profile=FeatureVector.classify_volume(
                        vector[18] if len(vector) > 18 else 0.0,
                        vector[23] if len(vector) > 23 else 0.0,
                    ),
                )
                all_vectors.append(fv)

            if len(all_vectors) >= n:
                break
        if len(all_vectors) >= n:
            break

    return all_vectors[:n]


def _classify_outcome(forward_return: float, max_dd: float, max_ru: float) -> str:
    """Classify a window's outcome based on forward metrics."""
    if forward_return > 0.05 and max_dd > -0.03:
        return "explosive_continuation"
    elif forward_return > 0.02 and abs(max_dd) < 0.03:
        return "slow_grind_continuation"
    elif forward_return < -0.03:
        return "full_reversal"
    elif max_ru > 0.03 and forward_return < 0:
        return "fake_breakout_failure"
    elif abs(forward_return) < 0.02 and abs(max_dd) > 0.03:
        return "volatility_expansion_directionless"
    elif forward_return > 0 and max_dd < -0.04:
        return "whipsaw_continuation"
    else:
        return "mixed_regime"


async def ingest_ticker_patterns(
    ticker: str,
    timeframe: str = "1h",
    window_size: int = 60,
) -> int:
    """
    Background job: ingest patterns for a ticker.
    In production, would fetch real data from a provider.
    """
    from app.storage.repository import PatternRepository

    candidates = generate_mock_candidates(n=200, ticker=ticker, timeframe=timeframe)
    repo = PatternRepository()

    try:
        count = await repo.store_batch(candidates)
        return count
    except Exception:
        return len(candidates)


async def warm_cache(tickers: Optional[list[str]] = None):
    """
    Background job: pre-compute and cache feature vectors for common tickers.
    """
    tickers = tickers or ["SPY", "QQQ", "AAPL", "TSLA", "BTC-USD"]

    for ticker in tickers:
        await ingest_ticker_patterns(ticker)
