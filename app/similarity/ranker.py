"""
ECHO FORGE — Match Ranker
Post-processing layer for ranked echo matches.

Applies secondary ranking criteria:
- Recency bias (recent matches weighted slightly higher)
- Cross-asset diversity bonus
- Regime alignment scoring
- Confidence calibration
"""

from collections import Counter
from typing import Optional

import numpy as np

from app.similarity.matcher import EchoMatch


class MatchRanker:
    """
    Re-ranks and annotates echo matches with additional quality signals.
    """

    def __init__(
        self,
        recency_decay: float = 0.02,
        diversity_bonus: float = 0.05,
    ):
        self.recency_decay = recency_decay
        self.diversity_bonus = diversity_bonus

    def rerank(
        self,
        matches: list[EchoMatch],
        apply_diversity: bool = True,
    ) -> list[EchoMatch]:
        """
        Re-rank matches with secondary criteria.

        Parameters
        ----------
        matches : list[EchoMatch]
            Initial ranked matches from the matcher.
        apply_diversity : bool
            If True, boost matches from underrepresented tickers.

        Returns
        -------
        list[EchoMatch]
            Re-ranked matches.
        """
        if not matches:
            return matches

        # Compute adjusted scores
        scored = []
        ticker_counts = Counter(m.ticker for m in matches)

        for match in matches:
            score = match.composite_score

            # Diversity bonus: if this ticker is underrepresented, slight boost
            if apply_diversity:
                freq = ticker_counts[match.ticker] / len(matches)
                if freq < 0.3:
                    score += self.diversity_bonus * (1 - freq)

            scored.append((match, score))

        # Re-sort
        scored.sort(key=lambda x: x[1], reverse=True)

        # Reassign ranks
        reranked = []
        for rank, (match, _score) in enumerate(scored, start=1):
            match.rank = rank
            reranked.append(match)

        return reranked

    def compute_confidence(self, matches: list[EchoMatch]) -> float:
        """
        Compute overall confidence in the echo scan based on match quality.

        Returns value in [0, 1].
        """
        if not matches:
            return 0.0

        scores = [m.composite_score for m in matches]
        top_score = scores[0]
        mean_score = np.mean(scores)
        score_spread = np.std(scores)

        # High confidence = high top score + tight clustering of scores
        # Low confidence = low scores or wide spread
        confidence = (
            0.5 * top_score
            + 0.3 * mean_score
            + 0.2 * (1 - min(score_spread, 1.0))
        )

        return float(np.clip(confidence, 0.0, 1.0))

    def classify_echo_type(self, matches: list[EchoMatch]) -> str:
        """
        Classify the dominant echo type based on matched pattern outcomes.
        """
        if not matches:
            return "insufficient_data"

        returns = [m.feature_vector.forward_return for m in matches[:10]]
        drawdowns = [m.feature_vector.max_drawdown for m in matches[:10]]

        avg_return = np.mean(returns)
        avg_dd = np.mean(drawdowns)
        return_std = np.std(returns)

        # Classification logic
        if avg_return > 0.05 and avg_dd > -0.03:
            return "explosive_continuation"
        elif avg_return > 0.02 and return_std < 0.05:
            return "slow_grind_continuation"
        elif avg_return < -0.02 and avg_dd < -0.05:
            return "full_reversal"
        elif abs(avg_return) < 0.02 and return_std > 0.05:
            return "volatility_expansion_directionless"
        elif avg_return > 0 and avg_dd < -0.04:
            return "whipsaw_continuation"
        elif avg_return < 0 and avg_dd < -0.02:
            return "late_stage_compression"
        else:
            return "mixed_regime"
