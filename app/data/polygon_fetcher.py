"""
ECHO FORGE — Polygon.io OHLCV Fetcher
Fetches real OHLCV bars from the Polygon.io REST API v2.
STRICT DATA INTEGRITY: No simulated or synthetic fallbacks.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import httpx
import numpy as np
import pandas as pd

from libsml.rate_guard import AsyncPolygonRateGuard

logger = logging.getLogger("echoforge.data.polygon")

# Polygon v2 aggregates endpoint
_POLYGON_BASE = "https://api.polygon.io/v2/aggs/ticker"


# Timeframe string → (multiplier, timespan)
_TIMEFRAME_MAP: dict[str, tuple[int, str]] = {
    "1m":  (1,  "minute"),
    "2m":  (2,  "minute"),
    "5m":  (5,  "minute"),
    "10m": (10, "minute"),
    "15m": (15, "minute"),
    "30m": (30, "minute"),
    "1h":  (1,  "hour"),
    "2h":  (2,  "hour"),
    "4h":  (4,  "hour"),
    "1d":  (1,  "day"),
    "1w":  (1,  "week"),
}


def _polygon_ticker(ticker: str) -> str:
    """
    Polygon uses different formats for crypto vs equities.
    BTC-USD → X:BTCUSD
    ETH-USD → X:ETHUSD
    Equities are unchanged.
    """
    if "-USD" in ticker.upper():
        base = ticker.upper().replace("-USD", "")
        return f"X:{base}USD"
    return ticker.upper()


async def fetch_ohlcv(
    ticker: str,
    timeframe: str = "1h",
    polygon_key: Optional[str] = None,
    n_bars: int = 120,
) -> pd.DataFrame:
    """
    Fetch current market data from Polygon.io.
    STRICT DATA INTEGRITY: No synthetic fallback. If data is unavailable,
    the system identifies a structural gap and halts the intelligence cycle.
    """
    if not polygon_key:
        logger.error("Data Integrity Error: Polygon key absent for %s", ticker)
        raise ValueError(f"CRITICAL: Missing POLYGON_KEY. Pattern memory requires live data.")

    tf_key = timeframe.lower().replace(" ", "")
    if tf_key not in _TIMEFRAME_MAP:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    multiplier, timespan = _TIMEFRAME_MAP[tf_key]
    poly_ticker = _polygon_ticker(ticker)

    to_dt = datetime.utcnow()
    lookahead_days = max(14, n_bars * 2) if timespan in ("minute", "hour") else n_bars * 2
    from_dt = to_dt - timedelta(days=lookahead_days)

    url = (
        f"{_POLYGON_BASE}/{poly_ticker}/range/{multiplier}/{timespan}"
        f"/{from_dt.strftime('%Y-%m-%d')}/{to_dt.strftime('%Y-%m-%d')}"
    )
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": min(n_bars + 50, 50000),
        "apiKey": polygon_key,
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                await AsyncPolygonRateGuard.wait()  # Enforce rate limit before every call
                resp = await client.get(url, params=params)
                
                # Law 1.1: Standardize institutional throughput feedback
                if resp.status_code == 429:
                    await AsyncPolygonRateGuard.emergency_backoff()
                    continue
                
                if resp.status_code == 403:
                    logger.error("Authentication Failure: Polygon API key rejected (403).")
                    raise PermissionError("Polygon API: Invalid key or unauthorized.")
                    
                if resp.status_code != 200:
                    logger.error("Data integrity violation: Polygon returned %d", resp.status_code)
                    resp.raise_for_status()

                data = resp.json()
                break # Success
                
            except httpx.RequestError as e:
                logger.error("Connection Error: Failed to reach Polygon API (Attempt %d): %s", attempt + 1, e)
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Institutional Data Feed Offline: {e}")

    results = data.get("results", [])
    if not results:
        # NOTE: Not a 404, but a data gap for the requested range
        logger.warning("Data Gap: No results for %s in requested window.", ticker)
        return pd.DataFrame() 

    df = pd.DataFrame(results)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = df[["open", "high", "low", "close", "volume"]].sort_index()

    return df.tail(n_bars).copy()
