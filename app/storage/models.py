"""
ECHO FORGE — Storage Models
SQLAlchemy models for pattern storage and async database setup.
"""

from datetime import datetime
from typing import Optional

import numpy as np
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, LargeBinary,
    Index, text,
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import EchoForgeConfig

Base = declarative_base()

# Module-level engine and session factory
_engine = None
_async_session_factory = None


class PatternWindowModel(Base):
    """Persisted pattern window with feature vector and outcome data."""
    __tablename__ = "pattern_windows"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(32), nullable=False, index=True)
    timeframe = Column(String(8), nullable=False, index=True)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    window_size = Column(Integer, nullable=False)

    # Serialized feature vector (numpy bytes)
    feature_vector = Column(LargeBinary, nullable=False)
    vector_dim = Column(Integer, nullable=False)

    # Structural profiles
    volatility_profile = Column(String(32), default="")
    momentum_profile = Column(String(32), default="")
    volume_profile = Column(String(32), default="")
    compression_cycles = Column(Integer, default=0)
    expansion_cycles = Column(Integer, default=0)

    # Forward outcome data
    outcome_label = Column(String(64), default="")
    forward_return = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    max_runup = Column(Float, default=0.0)
    time_to_resolution = Column(Integer, default=0)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_ticker_timeframe", "ticker", "timeframe"),
        Index("ix_ticker_start", "ticker", "start_time"),
    )

    def get_vector(self) -> np.ndarray:
        """Deserialize feature vector from bytes."""
        return np.frombuffer(self.feature_vector, dtype=np.float64)

    @staticmethod
    def from_vector(vector: np.ndarray) -> bytes:
        """Serialize feature vector to bytes."""
        return vector.astype(np.float64).tobytes()


class ScanResultModel(Base):
    """Cached scan results for replay and audit."""
    __tablename__ = "scan_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(32), nullable=False, index=True)
    timeframe = Column(String(8), nullable=False)
    window_size = Column(Integer, nullable=False)
    echo_type = Column(String(64), nullable=False)
    similarity_score = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    n_matches = Column(Integer, nullable=False)
    narrative = Column(String(2000), default="")
    result_json = Column(LargeBinary, nullable=True)  # full JSON blob
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_scan_ticker_time", "ticker", "created_at"),
    )


async def init_db(config: EchoForgeConfig):
    """Initialize database engine and create tables."""
    global _engine, _async_session_factory

    url = config.async_database_url
    is_sqlite = url.startswith("sqlite")

    engine_kwargs: dict = {"echo": config.debug}
    if not is_sqlite:
        # pool_size / max_overflow are Postgres-only; SQLite uses StaticPool
        engine_kwargs["pool_size"] = 10
        engine_kwargs["max_overflow"] = 20

    _engine = create_async_engine(url, **engine_kwargs)

    _async_session_factory = sessionmaker(
        _engine, class_=AsyncSession, expire_on_commit=False
    )

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def get_session() -> AsyncSession:
    """Get an async database session."""
    if _async_session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _async_session_factory()
