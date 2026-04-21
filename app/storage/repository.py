"""
ECHO FORGE — Pattern Repository
Data access layer for pattern windows and scan results.
"""

from datetime import datetime
from typing import Optional

import numpy as np
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.encoder.feature_vector import FeatureVector
from app.storage.models import PatternWindowModel, ScanResultModel, get_session


class PatternRepository:
    """
    Async repository for pattern window CRUD operations.
    Handles serialization/deserialization of feature vectors.
    """

    async def store_pattern(
        self,
        feature_vector: FeatureVector,
        session: Optional[AsyncSession] = None,
    ) -> int:
        """Store a single pattern window. Returns the new record ID."""
        own_session = session is None
        if own_session:
            session = get_session()

        try:
            model = PatternWindowModel(
                ticker=feature_vector.ticker,
                timeframe=feature_vector.timeframe,
                start_time=feature_vector.start_time,
                end_time=feature_vector.end_time,
                window_size=feature_vector.window_size,
                feature_vector=PatternWindowModel.from_vector(feature_vector.vector),
                vector_dim=len(feature_vector.vector),
                volatility_profile=feature_vector.volatility_profile,
                momentum_profile=feature_vector.momentum_profile,
                volume_profile=feature_vector.volume_profile,
                compression_cycles=feature_vector.compression_cycles,
                expansion_cycles=feature_vector.expansion_cycles,
                outcome_label=feature_vector.outcome_label,
                forward_return=feature_vector.forward_return,
                max_drawdown=feature_vector.max_drawdown,
                max_runup=feature_vector.max_runup,
                time_to_resolution=feature_vector.time_to_resolution,
            )
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return model.id
        finally:
            if own_session:
                await session.close()

    async def store_batch(
        self,
        feature_vectors: list[FeatureVector],
        session: Optional[AsyncSession] = None,
    ) -> int:
        """Store a batch of pattern windows. Returns count stored."""
        own_session = session is None
        if own_session:
            session = get_session()

        try:
            models = []
            for fv in feature_vectors:
                models.append(PatternWindowModel(
                    ticker=fv.ticker,
                    timeframe=fv.timeframe,
                    start_time=fv.start_time,
                    end_time=fv.end_time,
                    window_size=fv.window_size,
                    feature_vector=PatternWindowModel.from_vector(fv.vector),
                    vector_dim=len(fv.vector),
                    volatility_profile=fv.volatility_profile,
                    momentum_profile=fv.momentum_profile,
                    volume_profile=fv.volume_profile,
                    compression_cycles=fv.compression_cycles,
                    expansion_cycles=fv.expansion_cycles,
                    outcome_label=fv.outcome_label,
                    forward_return=fv.forward_return,
                    max_drawdown=fv.max_drawdown,
                    max_runup=fv.max_runup,
                    time_to_resolution=fv.time_to_resolution,
                ))
            session.add_all(models)
            await session.commit()
            return len(models)
        finally:
            if own_session:
                await session.close()

    async def load_candidates(
        self,
        timeframe: Optional[str] = None,
        ticker: Optional[str] = None,
        limit: int = 10000,
        session: Optional[AsyncSession] = None,
    ) -> list[FeatureVector]:
        """
        Load candidate feature vectors for matching.
        Optionally filter by timeframe and/or ticker.
        """
        own_session = session is None
        if own_session:
            session = get_session()

        try:
            query = select(PatternWindowModel)
            if timeframe:
                query = query.where(PatternWindowModel.timeframe == timeframe)
            if ticker:
                query = query.where(PatternWindowModel.ticker == ticker)
            query = query.limit(limit)

            result = await session.execute(query)
            rows = result.scalars().all()

            vectors = []
            for row in rows:
                vectors.append(FeatureVector(
                    vector=row.get_vector(),
                    ticker=row.ticker,
                    timeframe=row.timeframe,
                    start_time=row.start_time,
                    end_time=row.end_time,
                    window_size=row.window_size,
                    forward_return=row.forward_return,
                    max_drawdown=row.max_drawdown,
                    max_runup=row.max_runup,
                    time_to_resolution=row.time_to_resolution,
                    outcome_label=row.outcome_label,
                    volatility_profile=row.volatility_profile,
                    momentum_profile=row.momentum_profile,
                    volume_profile=row.volume_profile,
                    compression_cycles=row.compression_cycles,
                    expansion_cycles=row.expansion_cycles,
                    id=row.id,
                ))

            return vectors
        finally:
            if own_session:
                await session.close()

    async def load_candidate_matrix(
        self,
        timeframe: Optional[str] = None,
        limit: int = 10000,
        session: Optional[AsyncSession] = None,
    ) -> tuple[np.ndarray, list[FeatureVector]]:
        """
        Load candidates as a numpy matrix for vectorized matching.
        Returns (matrix, metadata_list).
        """
        candidates = await self.load_candidates(
            timeframe=timeframe, limit=limit, session=session
        )
        if not candidates:
            return np.array([]), []

        matrix = np.vstack([c.vector for c in candidates])
        return matrix, candidates

    async def count_patterns(
        self,
        ticker: Optional[str] = None,
        session: Optional[AsyncSession] = None,
    ) -> int:
        """Count total patterns in the database."""
        own_session = session is None
        if own_session:
            session = get_session()

        try:
            query = select(func.count(PatternWindowModel.id))
            if ticker:
                query = query.where(PatternWindowModel.ticker == ticker)
            result = await session.execute(query)
            return result.scalar() or 0
        finally:
            if own_session:
                await session.close()

    async def get_tickers(
        self, session: Optional[AsyncSession] = None
    ) -> list[str]:
        """Get list of all tickers in the database."""
        own_session = session is None
        if own_session:
            session = get_session()

        try:
            query = select(PatternWindowModel.ticker).distinct()
            result = await session.execute(query)
            return [row[0] for row in result.all()]
        finally:
            if own_session:
                await session.close()
