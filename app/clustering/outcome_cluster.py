"""
ECHO FORGE — Outcome Clustering Engine
Clusters matched patterns into distinct outcome groups based on
their forward behavior.

This is critical for producing nuanced intelligence — the same
structural pattern can resolve in multiple distinct ways. Clustering
reveals those divergent paths.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from app.config import ClusteringConfig
from app.similarity.matcher import EchoMatch


@dataclass
class OutcomeCluster:
    """A single cluster of similar outcomes."""
    cluster_id: int
    label: str
    members: list[EchoMatch]
    centroid: dict
    probability: float  # fraction of total matches in this cluster

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "label": self.label,
            "n_members": len(self.members),
            "probability": round(self.probability, 3),
            "avg_return": round(self.centroid.get("avg_return", 0.0), 4),
            "avg_drawdown": round(self.centroid.get("avg_drawdown", 0.0), 4),
            "avg_runup": round(self.centroid.get("avg_runup", 0.0), 4),
            "avg_time_to_resolution": round(
                self.centroid.get("avg_time_to_resolution", 0.0), 1
            ),
            "members": [m.to_dict() for m in self.members],
        }


class OutcomeClusterer:
    """
    Clusters echo matches by their forward outcomes (return, drawdown, time).
    """

    # Canonical cluster labels based on outcome characteristics
    LABEL_MAP = {
        "high_return_low_dd": "explosive_continuation",
        "moderate_return_low_dd": "slow_grind_continuation",
        "negative_return_high_dd": "full_reversal",
        "low_return_high_vol": "volatility_expansion_directionless",
        "positive_then_reversal": "fake_breakout_failure",
    }

    def __init__(self, config: Optional[ClusteringConfig] = None):
        self.config = config or ClusteringConfig()

    def cluster(self, matches: list[EchoMatch]) -> list[OutcomeCluster]:
        """
        Cluster matched patterns by their forward outcomes.

        Parameters
        ----------
        matches : list[EchoMatch]
            Ranked echo matches with outcome data.

        Returns
        -------
        list[OutcomeCluster]
            Outcome clusters sorted by probability (descending).
        """
        if len(matches) < self.config.min_cluster_size:
            # Not enough data to cluster — return single cluster
            return self._single_cluster(matches)

        # Build outcome feature matrix
        outcome_matrix = self._build_outcome_matrix(matches)

        # Determine effective cluster count
        n_clusters = min(self.config.n_clusters, len(matches) // self.config.min_cluster_size)
        n_clusters = max(2, n_clusters)

        # Scale features
        scaler = StandardScaler()
        scaled = scaler.fit_transform(outcome_matrix)

        # Cluster
        if self.config.method == "hierarchical":
            labels = AgglomerativeClustering(
                n_clusters=n_clusters
            ).fit_predict(scaled)
        else:
            labels = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
            ).fit_predict(scaled)

        # Build cluster objects
        clusters = self._build_clusters(matches, labels, outcome_matrix)

        # Sort by probability descending
        clusters.sort(key=lambda c: c.probability, reverse=True)

        return clusters

    def _build_outcome_matrix(self, matches: list[EchoMatch]) -> np.ndarray:
        """Extract outcome features from matches for clustering."""
        rows = []
        for m in matches:
            fv = m.feature_vector
            rows.append([
                fv.forward_return,
                fv.max_drawdown,
                fv.max_runup,
                fv.time_to_resolution,
            ])
        return np.array(rows, dtype=np.float64)

    def _build_clusters(
        self,
        matches: list[EchoMatch],
        labels: np.ndarray,
        outcome_matrix: np.ndarray,
    ) -> list[OutcomeCluster]:
        """Construct OutcomeCluster objects from clustering results."""
        unique_labels = set(labels)
        total = len(matches)
        clusters = []

        for cid in sorted(unique_labels):
            mask = labels == cid
            members = [m for m, belongs in zip(matches, mask) if belongs]
            cluster_outcomes = outcome_matrix[mask]

            centroid = {
                "avg_return": float(np.mean(cluster_outcomes[:, 0])),
                "avg_drawdown": float(np.mean(cluster_outcomes[:, 1])),
                "avg_runup": float(np.mean(cluster_outcomes[:, 2])),
                "avg_time_to_resolution": float(np.mean(cluster_outcomes[:, 3])),
            }

            label = self._label_cluster(centroid)

            clusters.append(OutcomeCluster(
                cluster_id=int(cid),
                label=label,
                members=members,
                centroid=centroid,
                probability=len(members) / total,
            ))

        return clusters

    def _label_cluster(self, centroid: dict) -> str:
        """Assign a human-readable label based on centroid characteristics."""
        ret = centroid["avg_return"]
        dd = centroid["avg_drawdown"]
        ru = centroid["avg_runup"]

        if ret > 0.05 and dd > -0.03:
            return "explosive_continuation"
        elif ret > 0.02 and dd > -0.02:
            return "slow_grind_continuation"
        elif ret < -0.03:
            return "full_reversal"
        elif abs(ret) < 0.02 and ru > 0.03 and dd < -0.03:
            return "volatility_expansion_directionless"
        elif ru > 0.03 and ret < 0:
            return "fake_breakout_failure"
        elif ret > 0 and dd < -0.04:
            return "whipsaw_continuation"
        else:
            return "mixed_regime"

    def _single_cluster(self, matches: list[EchoMatch]) -> list[OutcomeCluster]:
        """Fallback when insufficient data for multi-cluster analysis."""
        if not matches:
            return []

        outcome_matrix = self._build_outcome_matrix(matches)
        centroid = {
            "avg_return": float(np.mean(outcome_matrix[:, 0])),
            "avg_drawdown": float(np.mean(outcome_matrix[:, 1])),
            "avg_runup": float(np.mean(outcome_matrix[:, 2])),
            "avg_time_to_resolution": float(np.mean(outcome_matrix[:, 3])),
        }

        return [OutcomeCluster(
            cluster_id=0,
            label=self._label_cluster(centroid),
            members=matches,
            centroid=centroid,
            probability=1.0,
        )]
