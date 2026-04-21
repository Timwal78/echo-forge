"""
ECHO FORGE — Request Schemas
Pydantic models for API request validation.
"""

from typing import Optional

from pydantic import BaseModel, Field


class EchoScanRequest(BaseModel):
    """Request body for POST /echo_scan."""
    ticker: str = Field(
        ...,
        description="Instrument identifier (e.g., 'AAPL', 'TSLA', 'BTC-USD')",
        examples=["AAPL", "TSLA", "SPY"],
    )
    timeframe: str = Field(
        default="1h",
        description="Bar timeframe",
        examples=["1m", "5m", "15m", "1h", "4h", "1d"],
    )
    window_size: int = Field(
        default=60,
        ge=20,
        le=200,
        description="Number of bars in the analysis window",
    )
    top_n: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of echo matches to return",
    )
    cross_asset: bool = Field(
        default=True,
        description="Enable cross-asset matching",
    )
    include_failure_analysis: bool = Field(
        default=True,
        description="Include failure mode analysis in response",
    )
    include_projections: bool = Field(
        default=True,
        description="Include forward projections in response",
    )
    # BYOK — callers supply their own Polygon key for live market data
    # If not provided, ECHO FORGE uses synthetic data (dev mode)
    polygon_key: Optional[str] = Field(
        default=None,
        description="BYOK: Polygon.io API key for live OHLCV data",
    )


class BatchScanRequest(BaseModel):
    """Request body for batch scanning multiple instruments."""
    tickers: list[str] = Field(
        ...,
        description="List of instrument identifiers",
        min_length=1,
        max_length=50,
    )
    timeframe: str = Field(default="1h")
    window_size: int = Field(default=60, ge=20, le=200)
    top_n: int = Field(default=10, ge=1, le=50)


class IngestRequest(BaseModel):
    """Request body for ingesting historical data into the pattern database."""
    ticker: str = Field(..., description="Instrument identifier")
    timeframe: str = Field(default="1h")
    window_size: int = Field(default=60, ge=20, le=200)
    overlap_ratio: float = Field(default=0.5, ge=0.0, le=0.9)
    days_back: int = Field(default=30, ge=1, le=1000)
    polygon_key: Optional[str] = Field(default=None, description="BYOK: Polygon.io API key")
