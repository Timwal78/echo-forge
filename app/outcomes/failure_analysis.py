"""
ECHO FORGE — Failure Analysis Engine
Identifies conditions where structurally similar patterns failed,
diverged from expected paths, and early warning signals.

This module is critical for institutional quality output.
Retail tools only show success bias. This engine explicitly
models the failure distribution.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats as sp_stats

from app.similarity.matcher import EchoMatch
from app.clustering.outcome_cluster import OutcomeCluster


@dataclass
class FailureCase:
    """A specific historical failure case."""
    match: EchoMatch
    failure_type: str
    severity: float  # 0-1, how bad the failure was
    divergence_bar: int  # bar where expected path diverged
    description: str


@dataclass
class FailureAnalysis:
    """Comprehensive failure analysis for an echo scan."""
    failure_rate: float
    avg_failure_severity: float
    max_failure_severity: float
    failure_cases: list[FailureCase]
    divergence_signals: list[str]
    risk_factors: list[str]
    failure_risk_score: float  # 0-1 composite risk

    def to_dict(self) -> dict:
        return {
            "failure_rate": round(self.failure_rate, 3),
            "avg_failure_severity": round(self.avg_failure_severity, 3),
            "max_failure_severity": round(self.max_failure_severity, 3),
            "failure_risk_score": round(self.failure_risk_score, 3),
            "n_failure_cases": len(self.failure_cases),
            "divergence_signals": self.divergence_signals,
            "risk_factors": self.risk_factors,
            "failure_cases": [
                {
                    "ticker": fc.match.ticker,
                    "failure_type": fc.failure_type,
                    "severity": round(fc.severity, 3),
                    "divergence_bar": fc.divergence_bar,
                    "description": fc.description,
                }
                for fc in self.failure_cases[:5]  # top 5 most severe
            ],
        }


class FailureAnalyzer:
    """
    Analyzes failure modes in echo matches.
    Identifies what went wrong when structurally similar patterns
    did NOT produce the expected outcome.
    """

    # Failure thresholds
    FAILURE_RETURN_THRESHOLD = -0.03
    SEVERE_DRAWDOWN_THRESHOLD = -0.08
    FALSE_BREAKOUT_THRESHOLD = 0.03

    def analyze(
        self,
        matches: list[EchoMatch],
        clusters: Optional[list[OutcomeCluster]] = None,
    ) -> FailureAnalysis:
        """
        Perform comprehensive failure analysis.

        Parameters
        ----------
        matches : list[EchoMatch]
            All echo matches (including failures).
        clusters : list[OutcomeCluster], optional
            Outcome clusters for cluster-aware analysis.

        Returns
        -------
        FailureAnalysis
        """
        if not matches:
            return self._empty_analysis()

        failure_cases = self._identify_failures(matches)
        divergence_signals = self._detect_divergence_signals(matches, clusters)
        risk_factors = self._identify_risk_factors(matches)
        failure_risk_score = self._compute_risk_score(matches, failure_cases)

        severities = [fc.severity for fc in failure_cases]

        return FailureAnalysis(
            failure_rate=len(failure_cases) / len(matches),
            avg_failure_severity=float(np.mean(severities)) if severities else 0.0,
            max_failure_severity=float(np.max(severities)) if severities else 0.0,
            failure_cases=sorted(failure_cases, key=lambda x: x.severity, reverse=True),
            divergence_signals=divergence_signals,
            risk_factors=risk_factors,
            failure_risk_score=failure_risk_score,
        )

    def _identify_failures(self, matches: list[EchoMatch]) -> list[FailureCase]:
        """Classify each matched pattern that resulted in a negative outcome."""
        failures = []

        for match in matches:
            fv = match.feature_vector
            ret = fv.forward_return
            dd = fv.max_drawdown
            ru = fv.max_runup

            failure_type = None
            severity = 0.0
            description = ""
            divergence_bar = fv.time_to_resolution

            # False breakout: ran up then collapsed
            if ru > self.FALSE_BREAKOUT_THRESHOLD and ret < 0:
                failure_type = "false_breakout"
                severity = min(1.0, abs(ret - ru) / 0.1)
                description = (
                    f"Pattern showed initial expansion of {ru:.1%} "
                    f"before reversing to {ret:.1%}. "
                    f"Classic false breakout / bull trap behavior."
                )

            # Deep drawdown failure
            elif dd < self.SEVERE_DRAWDOWN_THRESHOLD:
                failure_type = "severe_drawdown"
                severity = min(1.0, abs(dd) / 0.15)
                description = (
                    f"Pattern experienced severe drawdown of {dd:.1%} "
                    f"with final return of {ret:.1%}. "
                    f"Structural breakdown exceeded historical norms."
                )

            # Slow bleed
            elif ret < self.FAILURE_RETURN_THRESHOLD and abs(dd) < 0.05:
                failure_type = "slow_bleed"
                severity = min(1.0, abs(ret) / 0.08)
                description = (
                    f"Gradual deterioration with {ret:.1%} return. "
                    f"No acute breakdown but persistent negative drift."
                )

            # Whipsaw
            elif abs(ru) > 0.03 and abs(dd) > 0.03 and abs(ret) < 0.02:
                failure_type = "whipsaw"
                severity = min(1.0, (abs(ru) + abs(dd)) / 0.15)
                description = (
                    f"Volatile resolution with {ru:.1%} upside and {dd:.1%} "
                    f"downside excursion but minimal net move ({ret:.1%}). "
                    f"Directional conviction was unrewarded."
                )

            if failure_type:
                failures.append(FailureCase(
                    match=match,
                    failure_type=failure_type,
                    severity=severity,
                    divergence_bar=divergence_bar,
                    description=description,
                ))

        return failures

    def _detect_divergence_signals(
        self,
        matches: list[EchoMatch],
        clusters: Optional[list[OutcomeCluster]],
    ) -> list[str]:
        """Identify early warning signals that the pattern may diverge."""
        signals = []

        returns = [m.feature_vector.forward_return for m in matches]
        drawdowns = [m.feature_vector.max_drawdown for m in matches]

        # Bimodal return distribution (split outcomes)
        if len(returns) >= 10:
            stat, p_value = sp_stats.normaltest(returns)
            if p_value < 0.05:
                signals.append(
                    "Return distribution is non-normal (p={:.3f}), "
                    "suggesting bimodal or fat-tailed outcomes. "
                    "Pattern resolution is structurally ambiguous.".format(p_value)
                )

        # High variance in outcomes
        ret_std = np.std(returns)
        if ret_std > 0.08:
            signals.append(
                f"Outcome variance is elevated (std={ret_std:.1%}). "
                "Historical precedents show widely divergent resolutions."
            )

        # Cluster imbalance
        if clusters and len(clusters) >= 2:
            probs = [c.probability for c in clusters]
            max_prob = max(probs)
            if max_prob < 0.4:
                signals.append(
                    "No dominant outcome cluster (max probability "
                    f"{max_prob:.0%}). Pattern resolves through multiple "
                    "distinct paths with roughly equal likelihood."
                )

        # Drawdown precedes positive outcomes frequently
        dd_array = np.array(drawdowns)
        ret_array = np.array(returns)
        whipsaw_rate = np.mean((dd_array < -0.03) & (ret_array > 0))
        if whipsaw_rate > 0.3:
            signals.append(
                f"High whipsaw rate ({whipsaw_rate:.0%}): patterns that "
                "ultimately continued positive experienced significant "
                "intermediate drawdowns. Expect volatility before resolution."
            )

        return signals

    def _identify_risk_factors(self, matches: list[EchoMatch]) -> list[str]:
        """Identify structural risk factors from the match set."""
        risks = []

        returns = [m.feature_vector.forward_return for m in matches]

        # Negative skew
        if len(returns) >= 5:
            skew = sp_stats.skew(returns)
            if skew < -0.5:
                risks.append(
                    f"Negative return skew ({skew:.2f}): tail risk "
                    "is concentrated on the downside."
                )

        # High failure concentration in high-similarity matches
        top_5 = matches[:5]
        top_5_failures = sum(
            1 for m in top_5
            if m.feature_vector.forward_return < self.FAILURE_RETURN_THRESHOLD
        )
        if top_5_failures >= 3:
            risks.append(
                "Majority of highest-similarity matches resulted in "
                "negative outcomes. Structural precedent is bearish."
            )

        # Cross-asset divergence
        tickers = set(m.ticker for m in matches)
        if len(tickers) > 1:
            by_ticker = {}
            for m in matches:
                by_ticker.setdefault(m.ticker, []).append(
                    m.feature_vector.forward_return
                )
            ticker_means = {t: np.mean(r) for t, r in by_ticker.items()}
            if max(ticker_means.values()) > 0.03 and min(ticker_means.values()) < -0.03:
                risks.append(
                    "Cross-asset echo matches show conflicting outcomes. "
                    "Same structure resolved bullish in some instruments "
                    "and bearish in others."
                )

        return risks

    def _compute_risk_score(
        self, matches: list[EchoMatch], failure_cases: list[FailureCase]
    ) -> float:
        """Compute a composite failure risk score in [0, 1]."""
        if not matches:
            return 0.0

        failure_rate = len(failure_cases) / len(matches)
        avg_severity = (
            np.mean([fc.severity for fc in failure_cases])
            if failure_cases else 0.0
        )

        returns = [m.feature_vector.forward_return for m in matches]
        ret_std = np.std(returns) if len(returns) > 1 else 0.0

        score = (
            0.4 * failure_rate
            + 0.3 * avg_severity
            + 0.3 * min(1.0, ret_std / 0.1)
        )

        return float(np.clip(score, 0.0, 1.0))

    def _empty_analysis(self) -> FailureAnalysis:
        return FailureAnalysis(
            failure_rate=0.0,
            avg_failure_severity=0.0,
            max_failure_severity=0.0,
            failure_cases=[],
            divergence_signals=["Insufficient data for failure analysis."],
            risk_factors=[],
            failure_risk_score=0.0,
        )
