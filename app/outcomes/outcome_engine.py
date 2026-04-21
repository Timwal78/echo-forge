"""
ECHO FORGE — Outcome Engine
Computes aggregate outcome statistics from matched echo patterns.

This is the primary statistical engine that turns matched patterns
into actionable probability assessments.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats as sp_stats

from app.similarity.matcher import EchoMatch


@dataclass
class OutcomeStatistics:
    """Comprehensive outcome statistics from echo matches."""
    n_matches: int
    continuation_probability: float
    reversal_probability: float
    failure_probability: float
    avg_return: float
    median_return: float
    weighted_avg_return: float  # similarity-weighted
    max_adverse_excursion: float
    max_favorable_excursion: float
    avg_time_to_resolution: float
    median_time_to_resolution: float
    return_std: float
    sharpe_proxy: float  # avg_return / std (simplified)
    win_rate: float  # fraction with positive returns
    profit_factor: float  # sum of wins / abs(sum of losses)

    def to_dict(self) -> dict:
        return {
            "n_matches": self.n_matches,
            "continuation_probability": round(self.continuation_probability, 3),
            "reversal_probability": round(self.reversal_probability, 3),
            "failure_probability": round(self.failure_probability, 3),
            "avg_return": round(self.avg_return, 4),
            "median_return": round(self.median_return, 4),
            "weighted_avg_return": round(self.weighted_avg_return, 4),
            "max_adverse_excursion": round(self.max_adverse_excursion, 4),
            "max_favorable_excursion": round(self.max_favorable_excursion, 4),
            "avg_time_to_resolution": round(self.avg_time_to_resolution, 1),
            "median_time_to_resolution": round(self.median_time_to_resolution, 1),
            "return_std": round(self.return_std, 4),
            "sharpe_proxy": round(self.sharpe_proxy, 3),
            "win_rate": round(self.win_rate, 3),
            "profit_factor": round(self.profit_factor, 3),
        }


class OutcomeEngine:
    """Computes outcome statistics from a set of echo matches."""

    def compute(self, matches: list[EchoMatch]) -> OutcomeStatistics:
        """
        Compute full outcome statistics.

        Parameters
        ----------
        matches : list[EchoMatch]
            Ranked echo matches with outcome data.

        Returns
        -------
        OutcomeStatistics
        """
        if not matches:
            return self._empty_stats()

        returns = np.array([m.feature_vector.forward_return for m in matches])
        drawdowns = np.array([m.feature_vector.max_drawdown for m in matches])
        runups = np.array([m.feature_vector.max_runup for m in matches])
        times = np.array([m.feature_vector.time_to_resolution for m in matches])
        similarities = np.array([m.composite_score for m in matches])

        # Similarity-weighted return
        sim_weights = similarities / (np.sum(similarities) + 1e-10)
        weighted_return = float(np.dot(sim_weights, returns))

        # Directional probabilities (Institutional Hardening)
        config = get_config()
        reversal_thresh = config.projection.reversal_ret_threshold
        failure_thresh = config.projection.failure_drawdown_threshold
        
        continuation_prob = float(np.mean(returns > 0))
        reversal_prob = float(np.mean(returns < reversal_thresh))
        failure_prob = float(np.mean(
            (drawdowns < failure_thresh) & (returns < 0)
        ))

        # Win rate and profit factor
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        win_rate = len(wins) / len(returns) if len(returns) > 0 else 0.0
        sum_wins = np.sum(wins) if len(wins) > 0 else 0.0
        sum_losses = abs(np.sum(losses)) if len(losses) > 0 else 1e-10
        profit_factor = sum_wins / sum_losses

        # Sharpe proxy
        ret_std = float(np.std(returns))
        sharpe = float(np.mean(returns)) / (ret_std + 1e-10)

        return OutcomeStatistics(
            n_matches=len(matches),
            continuation_probability=continuation_prob,
            reversal_probability=reversal_prob,
            failure_probability=failure_prob,
            avg_return=float(np.mean(returns)),
            median_return=float(np.median(returns)),
            weighted_avg_return=weighted_return,
            max_adverse_excursion=float(np.min(drawdowns)),
            max_favorable_excursion=float(np.max(runups)),
            avg_time_to_resolution=float(np.mean(times)),
            median_time_to_resolution=float(np.median(times)),
            return_std=ret_std,
            sharpe_proxy=sharpe,
            win_rate=win_rate,
            profit_factor=float(profit_factor),
        )

    def _empty_stats(self) -> OutcomeStatistics:
        return OutcomeStatistics(
            n_matches=0,
            continuation_probability=0.0,
            reversal_probability=0.0,
            failure_probability=0.0,
            avg_return=0.0,
            median_return=0.0,
            weighted_avg_return=0.0,
            max_adverse_excursion=0.0,
            max_favorable_excursion=0.0,
            avg_time_to_resolution=0.0,
            median_time_to_resolution=0.0,
            return_std=0.0,
            sharpe_proxy=0.0,
            win_rate=0.0,
            profit_factor=0.0,
        )
