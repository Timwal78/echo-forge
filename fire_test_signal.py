import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv

# Set PYTHONPATH to include the app directory
import sys
sys.path.append(os.getcwd())

from app.services.notifier import DiscordNotifier
from app.services.signal_generator import SignalGenerator
from app.schemas.echo_response import (
    EchoScanResponse, MatchResponse, ProjectionResponse, ScenarioResponse,
    OutcomeDistributionResponse
)

# Load the env we just updated
load_dotenv()

async def fire_test():
    webhook = os.getenv("DISCORD_WEBHOOK_ECHO")
    print(f"Testing Webhook: {webhook[:40]}...")
    
    notifier = DiscordNotifier(webhook)
    generator = SignalGenerator()
    
    # Mock a high-grade GME signal
    mock_scan = EchoScanResponse(
        ticker="GME",
        timeframe="1h",
        window_size=60,
        similarity_score=0.92,
        echo_type="explosive_continuation",
        confidence=0.85,
        n_matches=12,
        top_matches=[
            MatchResponse(
                ticker="GME", timeframe="1h", start_time="2021-01-10", end_time="2021-01-20",
                similarity=0.92, cosine=0.93, euclidean=0.08, forward_return=1.50,
                max_drawdown=0.15, time_to_resolution=15, outcome_label="continuation", rank=1,
                c1_open=15.00, c2_close=22.40
            )
        ],
        outcome_distribution=OutcomeDistributionResponse(
            continuation=0.9, reversal=0.05, failure=0.05, neutral=0.0,
            mean_return=0.85, median_return=0.75, std_return=0.30,
            skew=1.2, kurtosis=2.0, percentile_5=-0.10, percentile_25=0.40,
            percentile_75=1.2, percentile_95=2.5
        ),
        outcome_clusters=[],
        projection=ProjectionResponse(
            primary_scenario=ScenarioResponse(
                label="Gamma Expansion Pattern",
                probability=0.85,
                expected_return=1.50,
                return_range=[0.80, 2.50],
                time_to_resolution="15 bars",
                confidence=0.9,
                description="Structural match with Jan 2021 pre-surge fractal."
            ),
            alternative_scenarios=[],
            overall_confidence=0.85,
            time_horizon="1-2 sessions",
            narrative="Current GME structure shows extreme cohesion with historical pre-squeeze volatility expansion."
        ),
        narrative="Institutional Signal: Structural match confirmed with 92% similarity."
    )
    
    embed = generator.generate_trade_card(mock_scan)
    embed["title"] = f"🚀 MOASS CANDIDATE 🚀 — {embed['title']}"
    embed["color"] = 0xFF00FF # Purple highlight
    
    success = await notifier.send_embed(embed)
    if success:
        print("SUCCESS: Test signal delivered to Echo Forge Brain.")
    else:
        print("FAILURE: Could not deliver test signal.")

if __name__ == "__main__":
    asyncio.run(fire_test())
