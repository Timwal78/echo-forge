"""
ECHO FORGE — Signal Generation Service
Converts raw pattern matches and projections into high-density trade signals.
"""

import math
from datetime import datetime, timezone, timedelta
from typing import Optional

from app.schemas.echo_response import EchoScanResponse
from app.config import get_config

class SignalGenerator:
    """
    Synthesizes fractal, sinusoidal, and projection data into 
    actionable institutional-grade trade alerts.
    """

    def __init__(self):
        self.config = get_config()

    def generate_trade_card(self, scan: EchoScanResponse) -> dict:
        """
        Generate a rich Discord embed for a high-confidence echo scan.
        """
        # Institutional Rule 1.1: If data is missing or confidence is zero, force F grade
        if not scan.top_matches or scan.confidence <= 0:
            return self._generate_failure_card(scan)

        top_match = scan.top_matches[0]
        proj = scan.projection
        
        # 1. Scoring & Grading (Law 2 Compliance)
        score_data = self._calculate_institutional_score(scan)
        total_score = score_data["total"]
        grade = self._get_letter_grade(total_score)
        color = self._get_direction_color(proj.primary_scenario.expected_return)
            
        # 2. Setup Naming
        setup_name = self._get_setup_name(scan.echo_type)
        
        # 3. Sinusoidal Alignment (147-day Cycle)
        cycle_phase_desc = self._calculate_cycle_phase()
        
        # 4. Decision Support (Directives)
        directives = self._calculate_directives(scan)
        
        # 5. Ranking & Formatting Embed
        is_advertising = scan.ticker in self.config.mega_caps or top_match.c2_close > self.config.alert_max_price
        
        # SML Institutional Ranking
        rank_label = "RANK: BETA ⭐"
        rank_color = 0x00FFFF # Cyan
        
        if self.config.price_tier_small[0] <= top_match.c2_close <= self.config.price_tier_small[1]:
            rank_label = "RANK: ALPHA ⭐⭐"
            rank_color = 0xFFD700 # Gold
        elif is_advertising:
            rank_label = "RANK: BENCHMARK 📺"
            rank_color = 0xAAAAAA # Grey

        title_prefix = "🚨 ECHO SIGNAL"
        if is_advertising:
            title_prefix = "📺 ADVERTISING BENCHMARK"
            
        fields = [
            {"name": "🧠 INTEL BREADCRUMB", "value": f"**Rank**: `{rank_label}` | **Setup**: `{setup_name}`", "inline": False},
            {"name": "📊 PRIMARY PROJECTION", "value": f"**Expected Move**: `{proj.primary_scenario.expected_return:+.1%}`\n**Probability**: `{proj.primary_scenario.probability:.0%}`", "inline": True},
            {"name": "⏳ TIME HORIZON", "value": f"**Resolution**: `{scan.projection.time_horizon}`\n**Grade**: **{grade}** (`{total_score}/100`)", "inline": True},
            {"name": "🌀 CYCLE ALIGNMENT (147d)", "value": f"`{cycle_phase_desc}`", "inline": False},
        ]
        
        if directives:
            fields.append({
                "name": "⚡ TRADE DIRECTIVES (ACTIONABLE)", 
                "value": f"**Entry Zone**: `{directives['entry']}`\n**Profit Target**: `{directives['target']}`\n**Stop Loss**: `{directives['stop']}`", 
                "inline": False
            })

        fields.append({"name": "📜 INSTITUTIONAL NARRATIVE", "value": f"*{scan.narrative}*", "inline": False})

        embed = {
            "title": f"{title_prefix}: {scan.ticker} ({scan.timeframe})",
            "description": f"Targeting historical structural recurrence with {scan.confidence:.0%} confidence.",
            "color": color,
            "fields": fields,
            "footer": {"text": f"SML Echo Forge v0.1 | {datetime.now(timezone.utc).strftime('%H:%M UTC')}"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return embed

    def _calculate_cycle_phase(self) -> str:
        """Calculate the current phase of the 147-day cycle."""
        ref_date = datetime.strptime(self.config.features.cycle_reference_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta_days = (now - ref_date).total_seconds() / (24 * 3600)
        period = self.config.features.cycle_period_days
        
        # Proximity to peak (0, 147, 294...)
        days_in_cycle = delta_days % period
        proximity = min(days_in_cycle, period - days_in_cycle)
        
        if proximity < 5:
            return "💥 CRITICAL CONVERGENCE (Peak Phase)"
        elif proximity < 15:
            return "⚠️ ELEVATED COHESION (Arrival Phase)"
        elif days_in_cycle < period / 2:
            return "📈 EXPANDING GAP (Early Cycle)"
        else:
            return "📉 CONTRACTING GAP (Late Cycle)"

    def _calculate_directives(self, scan: EchoScanResponse) -> Optional[dict]:
        """Generate high-density trade directives with real price levels."""
        if not scan.projection or not scan.top_matches:
            return None
        
        # Use the most recent match's closing price as proxy for current price
        last_price = scan.top_matches[0].c2_close
        if last_price <= 0:
            return None
            
        expected_ret = scan.projection.primary_scenario.expected_return
        
        # Risks units based on average historical drawdown of matches
        avg_drawdown = sum(m.max_drawdown for m in scan.top_matches) / len(scan.top_matches)
        
        # Institutional R:R buffer
        risk_buffer = max(self.config.risk_buffer_min, avg_drawdown * self.config.rr_buffer_mult) 
        
        if expected_ret > 0:
            target_price = last_price * (1 + expected_ret)
            stop_price = last_price * (1 - risk_buffer)
        else:
            target_price = last_price * (1 + expected_ret)  # expected_ret is negative
            stop_price = last_price * (1 + risk_buffer)

        rr = abs(expected_ret / risk_buffer) if risk_buffer > 0 else 0

        return {
            "entry": f"${last_price:,.2f} (Structural Pivot)",
            "target": f"${target_price:,.2f} ({expected_ret:+.1%} | Prob: {scan.projection.primary_scenario.probability:.0%})",
            "stop": f"${stop_price:,.2f} ({-risk_buffer:+.1%} | Avg DD: {avg_drawdown:.1%})",
            "rr": f"{rr:.1f}:1"
        }

    def _calculate_institutional_score(self, scan: EchoScanResponse) -> dict:
        """
        Weighted 6-factor institutional scoring engine.
        Law 2 Compliance: Parameterized weights from config.
        """
        # Similarity Factor (0-100)
        f_similarity = scan.similarity_score * 100
        
        # Edge Factor (0-100)
        # Standardized Sharpe Proxy: 0.5+ is strong, scaled to 100
        edge_proxy = (scan.outcome_distribution.mean_return / max(0.01, scan.outcome_distribution.std_return))
        f_edge = min(100, max(0, edge_proxy * 50))
        
        # Probability Factor (0-100)
        f_prob = scan.projection.primary_scenario.probability * 100
        
        # Cycle Factor (0-100)
        # Days from peak, closer to 147d multiples = higher score
        ref_date = datetime.strptime(self.config.features.cycle_reference_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        delta_days = (datetime.now(timezone.utc) - ref_date).total_seconds() / (24 * 3600)
        period = self.config.features.cycle_period_days
        proximity = min(delta_days % period, period - (delta_days % period))
        f_cycle = max(0, 100 - (proximity * (100 / 30))) # 30 day window for alignment
        
        # Confidence Factor (0-100)
        f_conf = scan.confidence * 100

        total = (
            f_similarity * self.config.weight_similarity +
            f_edge * self.config.weight_edge +
            f_prob * self.config.weight_probability +
            f_cycle * self.config.weight_cycle +
            f_conf * self.config.weight_confidence
        )

        return {"total": round(total, 1), "factors": {}}

    def _get_letter_grade(self, score: float) -> str:
        """Returns SML standard letter grade."""
        for grade, threshold in self.config.grade_thresholds.items():
            if score >= threshold:
                return grade
        return "F"

    def _get_direction_color(self, expected_return: float) -> int:
        """Returns SML standard directional color."""
        if expected_return > 0:
            return 0x00FF88 # Institutional Bullish Green
        elif expected_return < 0:
            return 0xFF4444 # Institutional Bearish Red
        return 0x888888 # Neutral Grey

    def _get_setup_name(self, echo_type: str) -> str:
        """Maps raw echo types to standardized setup names."""
        mapping = {
            "explosive_continuation": "Structural Surge",
            "slow_grind_continuation": "Absorption Grind",
            "full_reversal": "Regime Rejection",
            "fake_breakout_failure": "Expansion Trap",
            "volatility_expansion_directionless": "Volatility Squeeze",
            "whipsaw_continuation": "Shakeout Resolution"
        }
        return mapping.get(echo_type, "Structural Recurrence")

    def _generate_failure_card(self, scan: EchoScanResponse) -> dict:
        """Returns a standardized 'Failure' alert for missing/invalid data."""
        return {
            "title": f"❌ DATA INTEGRITY FAILURE: {scan.ticker}",
            "description": "Scan failed to meet Institutional Rule 1.1 (Zero-Fake Policy).",
            "color": 0xFF4444,
            "fields": [
                {"name": "Status", "value": "`Awaiting Institutional Data`", "inline": True},
                {"name": "Grade", "value": "**F**", "inline": True},
            ],
            "footer": {"text": "SML Echo Forge | Integrity Audit Active"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
