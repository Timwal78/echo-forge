"""
ECHO FORGE — Response Schemas
Pydantic models for API response serialization.
"""

from typing import Optional

from pydantic import BaseModel, Field


class MatchResponse(BaseModel):
    ticker: str
    timeframe: str
    start_time: str
    end_time: str
    similarity: float
    cosine: float
    euclidean: float
    forward_return: float
    max_drawdown: float
    time_to_resolution: int
    outcome_label: str
    rank: int
    # Price Data (Law 1: 100% Fetch Policy)
    c1_open: Optional[float] = Field(default=0.0)
    c2_close: Optional[float] = Field(default=0.0)
    h_max: Optional[float] = Field(default=0.0)
    l_min: Optional[float] = Field(default=0.0)


class OutcomeDistributionResponse(BaseModel):
    continuation: float
    reversal: float
    failure: float
    neutral: float
    mean_return: float
    median_return: float
    std_return: float
    skew: float
    kurtosis: float
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float


class ClusterResponse(BaseModel):
    cluster_id: int
    label: str
    n_members: int
    probability: float
    avg_return: float
    avg_drawdown: float
    avg_runup: float
    avg_time_to_resolution: float


class FailureCaseResponse(BaseModel):
    ticker: str
    failure_type: str
    severity: float
    divergence_bar: int
    description: str


class FailureAnalysisResponse(BaseModel):
    failure_rate: float
    avg_failure_severity: float
    max_failure_severity: float
    failure_risk_score: float
    n_failure_cases: int
    divergence_signals: list[str]
    risk_factors: list[str]
    failure_cases: list[FailureCaseResponse]


class ScenarioResponse(BaseModel):
    label: str
    probability: float
    expected_return: float
    return_range: list[float]
    time_to_resolution: str
    confidence: float
    description: str


class ProjectionResponse(BaseModel):
    primary_scenario: ScenarioResponse
    alternative_scenarios: list[ScenarioResponse]
    overall_confidence: float
    time_horizon: str
    narrative: str


class EchoScanResponse(BaseModel):
    """Full response for POST /echo_scan."""
    ticker: str
    timeframe: str
    window_size: int
    similarity_score: float = Field(
        description="Top match similarity score"
    )
    echo_type: str = Field(
        description="Classified echo pattern type"
    )
    confidence: float = Field(
        description="Overall scan confidence"
    )
    n_matches: int
    top_matches: list[MatchResponse]
    outcome_distribution: OutcomeDistributionResponse
    outcome_clusters: list[ClusterResponse]
    failure_analysis: Optional[FailureAnalysisResponse] = None
    projection: Optional[ProjectionResponse] = None
    narrative: str


class ReplayResponse(BaseModel):
    """Response for GET /replay/{ticker}."""
    ticker: str
    timeframe: str
    total_windows: int
    echo_evolution: list[dict]
    regime_transitions: list[dict]


class HealthResponse(BaseModel):
    status: str
    system: str
    version: str
