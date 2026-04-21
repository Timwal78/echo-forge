"""
ECHO FORGE — Projection Engine
Produces forward-looking probabilistic scenarios from echo analysis.

Generates:
- Most likely path
- Alternative paths (from cluster centroids)
- Confidence scores
- Time window estimates
- Institutional-grade narrative
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.config import ProjectionConfig
from app.clustering.outcome_cluster import OutcomeCluster
from app.clustering.distribution_model import OutcomeDistribution
from app.outcomes.outcome_engine import OutcomeStatistics
from app.outcomes.failure_analysis import FailureAnalysis
from app.similarity.matcher import EchoMatch
from app.similarity.ranker import MatchRanker


@dataclass
class Scenario:
    """A single projected forward scenario."""
    label: str
    probability: float
    expected_return: float
    return_range: tuple[float, float]  # (low, high)
    time_to_resolution: str
    confidence: float
    description: str

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "probability": round(self.probability, 3),
            "expected_return": round(self.expected_return, 4),
            "return_range": [
                round(self.return_range[0], 4),
                round(self.return_range[1], 4),
            ],
            "time_to_resolution": self.time_to_resolution,
            "confidence": round(self.confidence, 3),
            "description": self.description,
        }


@dataclass
class ProjectionResult:
    """Complete projection output."""
    primary_scenario: Scenario
    alternative_scenarios: list[Scenario]
    overall_confidence: float
    time_horizon: str
    narrative: str

    def to_dict(self) -> dict:
        return {
            "primary_scenario": self.primary_scenario.to_dict(),
            "alternative_scenarios": [
                s.to_dict() for s in self.alternative_scenarios
            ],
            "overall_confidence": round(self.overall_confidence, 3),
            "time_horizon": self.time_horizon,
            "narrative": self.narrative,
        }


class ProjectionEngine:
    """
    Synthesizes all analysis components into forward-looking projections.
    """

    def __init__(self, config: Optional[ProjectionConfig] = None):
        self.config = config or ProjectionConfig()

    def project(
        self,
        matches: list[EchoMatch],
        clusters: list[OutcomeCluster],
        distribution: OutcomeDistribution,
        outcome_stats: OutcomeStatistics,
        failure_analysis: FailureAnalysis,
        ticker: str,
        timeframe: str,
    ) -> ProjectionResult:
        """
        Generate complete forward projection.
        """
        if not matches or not clusters:
            return self._empty_projection(ticker, timeframe)

        # Build scenarios from clusters
        scenarios = self._build_scenarios(clusters, outcome_stats)

        if not scenarios:
            return self._empty_projection(ticker, timeframe)

        # Primary scenario = highest probability
        primary = scenarios[0]
        alternatives = scenarios[1:]

        # Overall confidence
        ranker = MatchRanker()
        match_confidence = ranker.compute_confidence(matches)
        cluster_clarity = 1 - failure_analysis.failure_risk_score
        overall_confidence = 0.6 * match_confidence + 0.4 * cluster_clarity

        # Time horizon
        avg_resolution = outcome_stats.avg_time_to_resolution
        time_horizon = self._format_time_horizon(avg_resolution, timeframe)

        # Narrative
        narrative = self._build_narrative(
            ticker=ticker,
            timeframe=timeframe,
            primary=primary,
            alternatives=alternatives,
            distribution=distribution,
            outcome_stats=outcome_stats,
            failure_analysis=failure_analysis,
            overall_confidence=overall_confidence,
            time_horizon=time_horizon,
            echo_type=ranker.classify_echo_type(matches),
        )

        return ProjectionResult(
            primary_scenario=primary,
            alternative_scenarios=alternatives,
            overall_confidence=overall_confidence,
            time_horizon=time_horizon,
            narrative=narrative,
        )

    def _build_scenarios(
        self,
        clusters: list[OutcomeCluster],
        stats: OutcomeStatistics,
    ) -> list[Scenario]:
        """Build one scenario per cluster, sorted by probability."""
        scenarios = []

        for cluster in clusters:
            if not cluster.members:
                continue

            returns = [
                m.feature_vector.forward_return for m in cluster.members
            ]
            times = [
                m.feature_vector.time_to_resolution for m in cluster.members
            ]

            ret_arr = np.array(returns)
            avg_time = np.mean(times)

            scenario = Scenario(
                label=cluster.label,
                probability=cluster.probability,
                expected_return=float(np.mean(ret_arr)),
                return_range=(
                    float(np.percentile(ret_arr, 10)),
                    float(np.percentile(ret_arr, 90)),
                ),
                time_to_resolution=f"{avg_time:.0f} bars",
                confidence=min(1.0, cluster.probability * len(cluster.members) / 5),
                description=self._describe_scenario(cluster),
            )
            scenarios.append(scenario)

        scenarios.sort(key=lambda s: s.probability, reverse=True)
        return scenarios

    def _describe_scenario(self, cluster: OutcomeCluster) -> str:
        """Generate institutional-tone description for a scenario."""
        ret = cluster.centroid["avg_return"]
        dd = cluster.centroid["avg_drawdown"]
        n = len(cluster.members)

        descriptions = {
            "explosive_continuation": (
                f"Historically resolves with aggressive directional follow-through. "
                f"Mean return {ret:+.1%} across {n} precedents with contained "
                f"intermediate drawdown ({dd:.1%})."
            ),
            "slow_grind_continuation": (
                f"Gradual directional resolution with low volatility. "
                f"Mean return {ret:+.1%} across {n} precedents. "
                f"Characteristic of absorption-driven continuation."
            ),
            "full_reversal": (
                f"Complete directional failure. Mean return {ret:+.1%} "
                f"with drawdown of {dd:.1%} across {n} precedents. "
                f"Structure breaks down rather than resolving directionally."
            ),
            "fake_breakout_failure": (
                f"Initial expansion followed by reversal. Mean return {ret:+.1%} "
                f"despite positive intermediate excursion. {n} precedents "
                f"suggest initial move is a trap."
            ),
            "volatility_expansion_directionless": (
                f"Range expansion without directional conviction. "
                f"Mean return {ret:+.1%} with high intermediate volatility. "
                f"{n} precedents show both sides getting tested."
            ),
            "whipsaw_continuation": (
                f"Continuation through volatility. Mean return {ret:+.1%} "
                f"but with significant drawdown ({dd:.1%}) before resolution. "
                f"{n} precedents suggest conviction is required."
            ),
        }

        return descriptions.get(
            cluster.label,
            f"Mixed outcome cluster with {ret:+.1%} mean return "
            f"across {n} precedents."
        )

    def _format_time_horizon(self, avg_bars: float, timeframe: str) -> str:
        """Convert bar count to human-readable time estimate."""
        tf_multipliers = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "4h": 240, "1d": 1440, "1w": 10080,
        }
        minutes_per_bar = tf_multipliers.get(timeframe, 60)
        total_minutes = avg_bars * minutes_per_bar

        if total_minutes < 60:
            return f"{total_minutes:.0f} minutes"
        elif total_minutes < 1440:
            return f"{total_minutes / 60:.0f} hours"
        elif total_minutes < 10080:
            days = total_minutes / 1440
            return f"{days:.0f}-{days + 1:.0f} sessions"
        else:
            weeks = total_minutes / 10080
            return f"{weeks:.0f}-{weeks + 1:.0f} weeks"

    def _build_narrative(
        self,
        ticker: str,
        timeframe: str,
        primary: Scenario,
        alternatives: list[Scenario],
        distribution: OutcomeDistribution,
        outcome_stats: OutcomeStatistics,
        failure_analysis: FailureAnalysis,
        overall_confidence: float,
        time_horizon: str,
        echo_type: str,
    ) -> str:
        """
        Build institutional-grade analytical narrative.
        No hype. No retail language. Pure analytical assessment.
        """
        parts = []

        # Opening: structural classification
        parts.append(
            f"Current {ticker} structure on {timeframe} matches "
            f"{echo_type.replace('_', ' ')} patterns with "
            f"{overall_confidence:.0%} confidence across "
            f"{outcome_stats.n_matches} historical precedents."
        )

        # Primary path
        parts.append(
            f"Primary resolution path ({primary.probability:.0%} probability): "
            f"{primary.label.replace('_', ' ')}. "
            f"Expected return {primary.expected_return:+.1%} within {time_horizon}."
        )

        # Alternative paths
        if alternatives:
            alt_desc = "; ".join(
                f"{s.label.replace('_', ' ')} ({s.probability:.0%})"
                for s in alternatives[:2]
            )
            parts.append(f"Alternative paths: {alt_desc}.")

        # Risk assessment
        if failure_analysis.failure_risk_score > 0.3:
            parts.append(
                f"Failure risk is elevated at {failure_analysis.failure_risk_score:.0%}. "
                f"Historical failure rate: {failure_analysis.failure_rate:.0%}."
            )

        if failure_analysis.divergence_signals:
            parts.append(failure_analysis.divergence_signals[0])

        # Statistical edge
        if outcome_stats.sharpe_proxy > 0.5:
            parts.append(
                f"Historical edge is positive (Sharpe proxy: "
                f"{outcome_stats.sharpe_proxy:.2f}, win rate: "
                f"{outcome_stats.win_rate:.0%})."
            )
        elif outcome_stats.sharpe_proxy < -0.3:
            parts.append(
                f"Historical edge is negative (Sharpe proxy: "
                f"{outcome_stats.sharpe_proxy:.2f}). "
                f"Structural precedent does not favor continuation."
            )

        return " ".join(parts)

    def _empty_projection(self, ticker: str, timeframe: str) -> ProjectionResult:
        return ProjectionResult(
            primary_scenario=Scenario(
                label="insufficient_data",
                probability=0.0,
                expected_return=0.0,
                return_range=(0.0, 0.0),
                time_to_resolution="N/A",
                confidence=0.0,
                description="Insufficient historical precedents for projection.",
            ),
            alternative_scenarios=[],
            overall_confidence=0.0,
            time_horizon="N/A",
            narrative=(
                f"Insufficient structural matches for {ticker} on {timeframe}. "
                f"Pattern database requires additional historical data."
            ),
        )
