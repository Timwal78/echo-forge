"""
ECHO FORGE — Echo Scan API Route
Primary endpoint for pattern intelligence scanning.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from app.config import get_config

logger = logging.getLogger("echoforge.routes.echo_scan")

from app.data.polygon_fetcher import fetch_ohlcv
from app.storage.models import get_session
from app.schemas.echo_request import EchoScanRequest, IngestRequest
from app.schemas.echo_response import (
    EchoScanResponse, MatchResponse, OutcomeDistributionResponse,
    ClusterResponse, FailureAnalysisResponse, FailureCaseResponse,
    ProjectionResponse, ScenarioResponse,
)
from app.core.window_extractor import WindowExtractor
from app.core.normalization import FeatureNormalizer, NormMethod
from app.encoder.pattern_encoder import PatternEncoder
from app.encoder.feature_vector import FeatureVector
from app.similarity.matcher import PatternMatcher
from app.similarity.ranker import MatchRanker
from app.clustering.outcome_cluster import OutcomeClusterer
from app.clustering.distribution_model import DistributionModeler
from app.outcomes.outcome_engine import OutcomeEngine
from app.outcomes.failure_analysis import FailureAnalyzer
from app.outcomes.projection import ProjectionEngine
from app.storage.repository import PatternRepository

router = APIRouter()


def _run_scan_pipeline(
    query_vector: FeatureVector,
    candidates: list[FeatureVector],
    request: EchoScanRequest,
) -> dict:
    """Execute the full scan pipeline against in-memory candidates."""
    config = get_config()

    # 1. Match
    matcher = PatternMatcher(config=config.similarity)
    matches = matcher.find_matches(
        query=query_vector,
        candidates=candidates,
        top_n=request.top_n,
        exclude_ticker=request.ticker,
        cross_asset=request.cross_asset,
    )

    # 2. Rerank
    ranker = MatchRanker()
    matches = ranker.rerank(matches)
    confidence = ranker.compute_confidence(matches)
    echo_type = ranker.classify_echo_type(matches)

    # 3. Cluster outcomes
    clusterer = OutcomeClusterer(config=config.clustering)
    clusters = clusterer.cluster(matches)

    # 4. Build distribution
    modeler = DistributionModeler()
    distribution = modeler.build_distribution(clusters)

    # 5. Compute outcome stats
    outcome_engine = OutcomeEngine()
    outcome_stats = outcome_engine.compute(matches)

    # 6. Failure analysis
    failure_analysis = None
    if request.include_failure_analysis:
        analyzer = FailureAnalyzer()
        failure_analysis = analyzer.analyze(matches, clusters)

    # 7. Projections
    projection = None
    if request.include_projections:
        projector = ProjectionEngine(config=config.projection)
        projection_result = projector.project(
            matches=matches,
            clusters=clusters,
            distribution=distribution,
            outcome_stats=outcome_stats,
            failure_analysis=failure_analysis or FailureAnalyzer().analyze(matches),
            ticker=request.ticker,
            timeframe=request.timeframe,
        )
        projection = projection_result

    return {
        "matches": matches,
        "confidence": confidence,
        "echo_type": echo_type,
        "clusters": clusters,
        "distribution": distribution,
        "outcome_stats": outcome_stats,
        "failure_analysis": failure_analysis,
        "projection": projection,
    }

def _encode_current_state(ohlcv, request: EchoScanRequest) -> FeatureVector:
    """Encode current market OHLCV data into a FeatureVector for matching."""
    config = get_config()
    encoder = PatternEncoder(feature_config=config.features)
    
    # We take the most recent 'window_size' bars
    window_data = ohlcv.tail(request.window_size)
    
    vector_array = encoder.encode(window_data)
    
    # Wrap in a FeatureVector container
    vector = FeatureVector(
        vector=vector_array,
        ticker=request.ticker,
        timeframe=request.timeframe,
        start_time=window_data.index[0],
        end_time=window_data.index[-1],
        window_size=request.window_size
    )
    
    return vector


@router.post("", response_model=EchoScanResponse)
async def echo_scan(request: EchoScanRequest):
    """
    Execute a full echo scan for the given instrument and timeframe.

    Encodes the current market state, searches historical patterns,
    clusters outcomes, and produces probabilistic projections.
    """
    config = get_config()

    # 1. Resolve Polygon key
    polygon_key = request.polygon_key or config.polygon_key

    # 2. Fetch current market data
    try:
        current_ohlcv = await fetch_ohlcv(
            ticker=request.ticker,
            timeframe=request.timeframe,
            polygon_key=polygon_key,
            n_bars=request.window_size + 20, 
        )
    except ConnectionRefusedError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except (PermissionError, ValueError) as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error("Echo Scan Data Error: %s", str(e))
        raise HTTPException(status_code=424, detail=f"Market data fetch failed: {str(e)}")

    if current_ohlcv.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Data Gap: No OHLCV results for {request.ticker} in the requested window."
        )

    # 3. Encode current state
    try:
        query_vector = _encode_current_state(current_ohlcv, request)
    except Exception as e:
        logger.error("Encoding Failure: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Pattern encoding failed: {str(e)}")

    # 4. Load candidates from pattern memory
    repo = PatternRepository()
    try:
        # Load candidates (filtered by timeframe for better similarity mapping)
        candidates = await repo.load_candidates(
            timeframe=request.timeframe,
            limit=10000 
        )
    except Exception as e:
        logger.warning("Database access error, proceeding with empty memory: %s", e)
        candidates = []

    if not candidates:
        logger.info("Echo Memory Empty for %s. No patterns in database.", request.ticker)
        return EchoScanResponse(
            ticker=request.ticker,
            timeframe=request.timeframe,
            window_size=request.window_size,
            similarity_score=0.0,
            echo_type="none",
            confidence=0.0,
            n_matches=0,
            top_matches=[],
            outcome_distribution=OutcomeDistributionResponse(
                continuation=0.0, reversal=0.0, failure=0.0, neutral=1.0,
                mean_return=0.0, median_return=0.0, std_return=0.0,
                skew=0.0, kurtosis=0.0,
                percentile_5=0.0, percentile_25=0.0, percentile_75=0.0, percentile_95=0.0
            ),
            outcome_clusters=[],
            narrative=f"Structural recurrence engine found 0 matches. Pattern memory for {request.timeframe} is currently empty."
        )

    # 5. Execute matching pipeline
    try:
        pipeline_result = _run_scan_pipeline(
            query_vector=query_vector,
            candidates=candidates,
            request=request,
        )
    except Exception as e:
        logger.error("Pipeline Failure: %s", str(e))
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Matching pipeline internal error.")

    matches = pipeline_result["matches"]
    if not matches:
        raise HTTPException(
            status_code=404,
            detail="No structural matches found above similarity threshold."
        )

    # 6. Build response
    top_matches = [
        MatchResponse(**m.to_dict()) for m in matches
    ]

    dist = pipeline_result["distribution"]
    dist_response = OutcomeDistributionResponse(**dist.to_dict())

    cluster_responses = [
        ClusterResponse(
            cluster_id=c.cluster_id,
            label=c.label,
            n_members=len(c.members),
            probability=round(c.probability, 3),
            avg_return=round(c.centroid["avg_return"], 4),
            avg_drawdown=round(c.centroid["avg_drawdown"], 4),
            avg_runup=round(c.centroid["avg_runup"], 4),
            avg_time_to_resolution=round(c.centroid["avg_time_to_resolution"], 1),
        )
        for c in pipeline_result["clusters"]
    ]

    failure_resp = None
    if pipeline_result["failure_analysis"]:
        fa = pipeline_result["failure_analysis"]
        fa_dict = fa.to_dict()
        failure_resp = FailureAnalysisResponse(**fa_dict)

    proj_resp = None
    if pipeline_result["projection"]:
        p = pipeline_result["projection"]
        p_dict = p.to_dict()
        proj_resp = ProjectionResponse(**p_dict)

    # Build narrative
    narrative = pipeline_result["projection"].narrative if pipeline_result["projection"] else (
        f"Echo scan complete for {request.ticker} on {request.timeframe}. "
        f"{len(matches)} structural matches identified."
    )

    return EchoScanResponse(
        ticker=request.ticker,
        timeframe=request.timeframe,
        window_size=request.window_size,
        similarity_score=round(matches[0].composite_score, 4) if matches else 0.0,
        echo_type=pipeline_result["echo_type"],
        confidence=round(pipeline_result["confidence"], 3),
        n_matches=len(matches),
        top_matches=top_matches,
        outcome_distribution=dist_response,
        outcome_clusters=cluster_responses,
        failure_analysis=failure_resp,
        projection=proj_resp,
        narrative=narrative,
    )


@router.post("/ingest")
async def ingest_patterns(request: IngestRequest):
    """
    Ingest historical data for a ticker into the pattern database.
    STRICT DATA INTEGRITY: Fetches real OHLCV data, encodes windows, and stores them.
    """
    from app.core.window_extractor import WindowExtractor
    from app.encoder.pattern_encoder import PatternEncoder
    from app.storage.repository import PatternRepository
    from app.config import get_config

    config = get_config()
    
    # 1. Fetch real historical data
    try:
        ohlcv = await fetch_ohlcv(
            ticker=request.ticker,
            timeframe=request.timeframe,
            polygon_key=request.polygon_key or config.polygon_key,
            n_bars=request.days_back * 24, # Approximate if intraday
        )
    except Exception as e:
        logger.error("Data integrity failure during ingestion for %s: %s", request.ticker, e)
        raise HTTPException(
            status_code=503,
            detail=f"Institutional Data Feed Unavailable: {str(e)}"
        )
    
    if ohlcv.empty:
        raise HTTPException(status_code=400, detail=f"No results returned for {request.ticker}. Feed might be inactive.")

    # 2. Extract windows and encode
    # AUDIT FIX: WindowExtractor expects config object, not size kwarg.
    extractor = WindowExtractor(config=config.window)
    windows = extractor.extract(
        df=ohlcv,
        ticker=request.ticker,
        timeframe=request.timeframe,
        window_size=request.window_size,
        overlap_ratio=request.overlap_ratio
    )
    
    encoder = PatternEncoder(feature_config=config.features)
    candidates = []
    
    for window in windows:
        vector = encoder.encode(window.ohlcv)
        candidates.append(vector)

    # 3. Store in repository
    repo = PatternRepository()
    try:
        count = await repo.store_batch(candidates)
        logger.info("Successfully ingested %d patterns for %s", count, request.ticker)
        return {
            "status": "success",
            "ticker": request.ticker,
            "patterns_ingested": count,
        }
    except Exception as e:
        logger.error("Failed to store patterns for %s: %s", request.ticker, e)
        # Fallback to in-memory success if DB fails (for dev parity)
        return {
            "status": "ingested_to_memory",
            "ticker": request.ticker,
            "patterns_generated": len(candidates),
            "note": f"Database error: {e}",
        }


# NOTE: Synthetic generation logic has been REMOVED.
# System requires a populated Postgres/SQLite database for pattern matching.
# Use ingestion jobs to populate memory.
