"""
ECHO FORGE — Replay API Route
Returns historical pattern evolution and matched echoes for a ticker.
"""

from fastapi import APIRouter, HTTPException

from app.schemas.echo_response import ReplayResponse
from app.storage.repository import PatternRepository

router = APIRouter()


@router.get("/{ticker}", response_model=ReplayResponse)
async def replay_ticker(ticker: str, timeframe: str = "1h", limit: int = 100):
    """
    Replay historical echo evolution for a ticker.

    Returns the sequence of pattern windows and their echo matches
    over time, revealing how the instrument's structural regime
    has evolved.
    """
    repo = PatternRepository()

    try:
        candidates = await repo.load_candidates(
            ticker=ticker, timeframe=timeframe, limit=limit
        )
    except Exception:
        candidates = []

    if not candidates:
        # Generate mock replay data
        candidates = _mock_replay_data(ticker, timeframe, limit)

    echo_evolution = []
    regime_transitions = []
    prev_label = None

    for i, fv in enumerate(candidates):
        entry = {
            "index": i,
            "start_time": fv.start_time.isoformat(),
            "end_time": fv.end_time.isoformat(),
            "forward_return": round(fv.forward_return, 4),
            "max_drawdown": round(fv.max_drawdown, 4),
            "outcome_label": fv.outcome_label,
            "volatility_profile": fv.volatility_profile,
            "momentum_profile": fv.momentum_profile,
        }
        echo_evolution.append(entry)

        # Detect regime transitions
        if fv.outcome_label and fv.outcome_label != prev_label:
            if prev_label is not None:
                regime_transitions.append({
                    "index": i,
                    "time": fv.start_time.isoformat(),
                    "from_regime": prev_label,
                    "to_regime": fv.outcome_label,
                })
            prev_label = fv.outcome_label

    return ReplayResponse(
        ticker=ticker,
        timeframe=timeframe,
        total_windows=len(candidates),
        echo_evolution=echo_evolution,
        regime_transitions=regime_transitions,
    )


def _mock_replay_data(ticker: str, timeframe: str, n: int):
    """Generate mock replay data for demonstration."""
    import numpy as np
    from datetime import datetime, timedelta
    from app.encoder.feature_vector import FeatureVector

    np.random.seed(hash(ticker) % 2**31)

    labels = [
        "explosive_continuation", "slow_grind_continuation",
        "full_reversal", "fake_breakout_failure",
        "volatility_expansion_directionless",
    ]
    vol_profiles = ["contracting", "expanding", "stable"]
    mom_profiles = ["accelerating", "decelerating", "neutral"]

    vectors = []
    base_time = datetime(2023, 1, 1)

    for i in range(n):
        # Regime shifts every ~20 windows
        regime_idx = (i // 20) % len(labels)
        label = labels[regime_idx]

        ret = np.random.normal(
            0.03 if "continuation" in label else -0.02,
            0.05
        )
        dd = -abs(np.random.normal(0.02, 0.03))

        vectors.append(FeatureVector(
            vector=np.random.randn(42),
            ticker=ticker,
            timeframe=timeframe,
            start_time=base_time + timedelta(hours=i),
            end_time=base_time + timedelta(hours=i + 1),
            window_size=60,
            forward_return=float(ret),
            max_drawdown=float(dd),
            max_runup=float(abs(ret) + 0.01),
            time_to_resolution=np.random.randint(3, 15),
            outcome_label=label,
            volatility_profile=np.random.choice(vol_profiles),
            momentum_profile=np.random.choice(mom_profiles),
        ))

    return vectors
