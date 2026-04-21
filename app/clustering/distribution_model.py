"""
ECHO FORGE — Distribution Model
Builds probability distributions from clustered outcomes.

Produces the statistical backbone for forward projections.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats as sp_stats

from app.clustering.outcome_cluster import OutcomeCluster


@dataclass
class OutcomeDistribution:
    """Full probability distribution over outcomes."""
    continuation_prob: float
    reversal_prob: float
    failure_prob: float
    neutral_prob: float

    mean_return: float
    median_return: float
    std_return: float
    skew: float
    kurtosis: float

    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float

    def to_dict(self) -> dict:
        return {
            "continuation": round(self.continuation_prob, 3),
            "reversal": round(self.reversal_prob, 3),
            "failure": round(self.failure_prob, 3),
            "neutral": round(self.neutral_prob, 3),
            "mean_return": round(self.mean_return, 4),
            "median_return": round(self.median_return, 4),
            "std_return": round(self.std_return, 4),
            "skew": round(self.skew, 3),
            "kurtosis": round(self.kurtosis, 3),
            "percentile_5": round(self.percentile_5, 4),
            "percentile_25": round(self.percentile_25, 4),
            "percentile_75": round(self.percentile_75, 4),
            "percentile_95": round(self.percentile_95, 4),
        }


class DistributionModeler:
    """Builds outcome distributions from cluster data."""

    # Map cluster labels to directional categories
    CONTINUATION_LABELS = {
        "explosive_continuation",
        "slow_grind_continuation",
        "whipsaw_continuation",
    }
    REVERSAL_LABELS = {"full_reversal"}
    FAILURE_LABELS = {"fake_breakout_failure"}

    def build_distribution(
        self, clusters: list[OutcomeCluster]
    ) -> OutcomeDistribution:
        """
        Aggregate cluster-level outcomes into a single probability distribution.
        """
        if not clusters:
            return self._empty_distribution()

        # Collect all returns from all clusters
        all_returns = []
        continuation_weight = 0.0
        reversal_weight = 0.0
        failure_weight = 0.0
        neutral_weight = 0.0

        for cluster in clusters:
            for member in cluster.members:
                all_returns.append(member.feature_vector.forward_return)

            if cluster.label in self.CONTINUATION_LABELS:
                continuation_weight += cluster.probability
            elif cluster.label in self.REVERSAL_LABELS:
                reversal_weight += cluster.probability
            elif cluster.label in self.FAILURE_LABELS:
                failure_weight += cluster.probability
            else:
                neutral_weight += cluster.probability

        returns = np.array(all_returns)

        if len(returns) < 2:
            return self._empty_distribution()

        return OutcomeDistribution(
            continuation_prob=continuation_weight,
            reversal_prob=reversal_weight,
            failure_prob=failure_weight,
            neutral_prob=neutral_weight,
            mean_return=float(np.mean(returns)),
            median_return=float(np.median(returns)),
            std_return=float(np.std(returns)),
            skew=float(sp_stats.skew(returns)),
            kurtosis=float(sp_stats.kurtosis(returns)),
            percentile_5=float(np.percentile(returns, 5)),
            percentile_25=float(np.percentile(returns, 25)),
            percentile_75=float(np.percentile(returns, 75)),
            percentile_95=float(np.percentile(returns, 95)),
        )

    def _empty_distribution(self) -> OutcomeDistribution:
        return OutcomeDistribution(
            continuation_prob=0.0,
            reversal_prob=0.0,
            failure_prob=0.0,
            neutral_prob=1.0,
            mean_return=0.0,
            median_return=0.0,
            std_return=0.0,
            skew=0.0,
            kurtosis=0.0,
            percentile_5=0.0,
            percentile_25=0.0,
            percentile_75=0.0,
            percentile_95=0.0,
        )
