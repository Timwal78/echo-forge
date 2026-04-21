"""
ECHO FORGE — Feature Vector Container
Typed container for encoded pattern fingerprints with metadata.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np


@dataclass
class FeatureVector:
    """
    Immutable container for a pattern's encoded fingerprint
    plus metadata for retrieval and matching.
    """
    vector: np.ndarray
    ticker: str
    timeframe: str
    start_time: datetime
    end_time: datetime
    window_size: int

    # Outcome data (populated for historical patterns)
    forward_return: float = 0.0
    max_drawdown: float = 0.0
    max_runup: float = 0.0
    time_to_resolution: int = 0
    outcome_label: str = ""

    # Structural metadata
    volatility_profile: str = ""   # e.g., "contracting", "expanding", "stable"
    momentum_profile: str = ""     # e.g., "accelerating", "decelerating", "neutral"
    volume_profile: str = ""       # e.g., "climactic", "dry", "accumulating"
    compression_cycles: int = 0
    expansion_cycles: int = 0

    # Database ID (set after persistence)
    id: Optional[int] = None

    def similarity_to(self, other: "FeatureVector") -> float:
        """Quick cosine similarity to another feature vector."""
        a = self.vector
        b = other.vector
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def to_dict(self) -> dict:
        """Serialize to dictionary for storage/transport."""
        return {
            "ticker": self.ticker,
            "timeframe": self.timeframe,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "window_size": self.window_size,
            "forward_return": self.forward_return,
            "max_drawdown": self.max_drawdown,
            "max_runup": self.max_runup,
            "time_to_resolution": self.time_to_resolution,
            "outcome_label": self.outcome_label,
            "volatility_profile": self.volatility_profile,
            "momentum_profile": self.momentum_profile,
            "volume_profile": self.volume_profile,
            "compression_cycles": self.compression_cycles,
            "expansion_cycles": self.expansion_cycles,
            "vector_dim": len(self.vector),
        }

    @staticmethod
    def classify_volatility(vol_z: float) -> str:
        if vol_z < -0.5:
            return "contracting"
        elif vol_z > 0.5:
            return "expanding"
        return "stable"

    @staticmethod
    def classify_momentum(accel: float) -> str:
        if accel > 0.01:
            return "accelerating"
        elif accel < -0.01:
            return "decelerating"
        return "neutral"

    @staticmethod
    def classify_volume(vol_slope: float, concentration: float) -> str:
        if concentration > 0.35 and vol_slope > 0.1:
            return "climactic"
        elif vol_slope < -0.05:
            return "dry"
        elif vol_slope > 0.05:
            return "accumulating"
        return "neutral"
