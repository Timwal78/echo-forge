import asyncio
import logging
import random
from datetime import datetime, timezone, timedelta

import httpx

from app.config import get_config
from app.routes.echo_scan import echo_scan
from app.schemas.echo_request import EchoScanRequest
from app.services.notifier import DiscordNotifier
from app.services.signal_generator import SignalGenerator

logger = logging.getLogger("echoforge.services.driver")


class IntelligenceDriver:
    """
    The background 'Brain' of Echo Forge.
    Periodically scans the watchlist and sends detailed alerts when 
    recurrent structural patterns are detected with high confidence.
    """

    def __init__(self):
        self.config = get_config()
        self.notifier = DiscordNotifier(self.config.discord_webhook_echo)
        self.generator = SignalGenerator()
        self.is_running = False
        
        # Discovery State
        self.discovery_queue = []
        self.alert_history = {} # symbol -> last_alert_time
        self.mega_cap_alerts_in_cycle = 0
        self.last_tier_runs = {
            "beta": datetime.now(timezone.utc) - timedelta(minutes=15),
            "benchmark": datetime.now(timezone.utc) - timedelta(hours=1)
        }
        
        # Resilience State (API Guardian)
        self.conservation_mode_until = datetime.now(timezone.utc)
        self.pulse_sent = False
        self.consecutive_429s = 0

    async def start_loop(self):
        """Main monitoring loop."""
        self.is_running = True
        logger.info("Intelligence Driver Active. Discovery Mode: %s", self.config.enable_dynamic_watchlist)
        
        # Initial wait to let DB/Network stabilize
        await asyncio.sleep(5)

        while self.is_running:
            try:
                # 1. Update scanning universe
                scan_universe = await self._build_scan_universe()
                
                now = datetime.now(timezone.utc)
                if now < self.conservation_mode_until:
                    logger.warning("🛡️ API GUARDIAN: Conservation Mode Active. Focus: Tier 1 ALPHA Only.")
                
                logger.info("Executing global scan cycle for %d symbols...", len(scan_universe))
                
                self.mega_cap_alerts_in_cycle = 0 # Reset throttle
                
                for ticker in scan_universe:
                    if not self.is_running: break
                    
                    try:
                        await self._scan_asset(ticker)
                        self.consecutive_429s = 0 # Successful scan
                    except Exception as e:
                        if "429" in str(e):
                            await self._handle_429_event()
                            break # Terminate current cycle to allow API to breathe
                        else:
                            logger.error("Asset Scan Error (%s): %s", ticker, e)
                    
                    # Institutional Jitter
                    await asyncio.sleep(random.uniform(10, 15))

            except Exception as e:
                logger.error("🚨 GLOBAL DRIVER ERROR: %s", e)
                await asyncio.sleep(60) # Safety wait
            
            await asyncio.sleep(self.config.scan_interval_sec)

    async def _build_scan_universe(self) -> list[str]:
        """
        Dynamically builds the scan list with Tiered Intelligence Priority.
        """
        now = datetime.now(timezone.utc)
        alpha_targets = set() # Rule 1: Zero hardcoded tickers. All from discovery.
        beta_targets = set()
        benchmark_targets = set()
        
        # A. SqueezeOS Sync (Primary Candidates)
        if self.config.enable_dynamic_watchlist:
            hot_symbols = await self._fetch_squeeze_candidates()
            # We don't know their price yet, so we'll classify them during scan
            # but for build logic, assume they are high priority
            alpha_targets.update(hot_symbols)
        
        # B. Triage based on Rank Cadence
        in_conservation = now < self.conservation_mode_until
        
        run_beta = (now - self.last_tier_runs["beta"]) >= timedelta(minutes=15) and not in_conservation
        run_benchmark = (now - self.last_tier_runs["benchmark"]) >= timedelta(hours=1) and not in_conservation

        # ── Discovery Rotation: rotate through SqueezeOS candidates across cycles ──
        # If queue is empty, re-seed from SqueezeOS. Only fall back to regime indices
        # if SqueezeOS itself is unreachable.
        if not self.discovery_queue:
            if self.config.enable_dynamic_watchlist:
                self.discovery_queue = await self._fetch_squeeze_candidates()
            if not self.discovery_queue:
                self.discovery_queue = ["SPY", "QQQ", "IWM"]  # Absolute fallback

        # Pop next 5 from the front of the queue — advance the rotation
        batch_size = 5
        next_batch = self.discovery_queue[:batch_size]
        self.discovery_queue = self.discovery_queue[batch_size:]

        # Priority 1: All ALPHA (Small Cap) and Sync candidates
        final_list = list(alpha_targets)

        # Priority 2: Rotate BETA (Mid Cap) every 15 min
        if run_beta:
            final_list.extend(next_batch)
            self.last_tier_runs["beta"] = now

        # Priority 3: Rotate BENCHMARK (Large Cap) every 60 min
        if run_benchmark:
            final_list.append("SPY")  # Benchmark tracking
            self.last_tier_runs["benchmark"] = now

        # ── API GUARDIAN: Cap per-cycle universe to 5 tickers ──
        # Polygon Free tier = 5 req/min. With 13s enforced gap between calls,
        # 5 tickers costs ~65s — safe for one scan interval.
        MAX_TICKERS_PER_CYCLE = 5
        final_list = list(set(final_list))[:MAX_TICKERS_PER_CYCLE]
        
        return final_list

    async def _fetch_squeeze_candidates(self) -> list[str]:
        """Fetch candidates from the SqueezeOS API."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.config.squeeze_os_url}/api/beast/signals")
                if resp.status_code == 200:
                    data = resp.json()
                    symbols = [s['symbol'] for s in data.get('data', [])]
                    logger.info("Synced %d candidates from SqueezeOS", len(symbols))
                    return symbols
        except Exception as e:
            logger.warning("Could not sync with SqueezeOS: %s", e)
        return []

    async def _handle_429_event(self):
        """Action taken when Polygon rate limits are reached."""
        now = datetime.now(timezone.utc)
        self.consecutive_429s += 1
        self.conservation_mode_until = now + timedelta(minutes=15)
        
        # 1. Hibernate (Protocol 429)
        logger.warning("📉 PROTOCOL 429: Rate Limit Reached. Entering 60s Global Hibernation.")
        
        # 2. Status Pulse Alert (Once per event)
        if not self.pulse_sent:
            try:
                msg = {
                    "title": "🛡️ API GUARDIAN: CONSERVATION ACTIVE",
                    "description": "Shared API limit reached (429). The Brain is entering **15-minute Tier 1 Lockdown** to protect your $1-$15 Small Cap signals.",
                    "color": 0xFF9900,
                    "fields": [
                        {"name": "STATUS", "value": "Entering 60s Hibernation...", "inline": True},
                        {"name": "PROTOCOL", "value": "Wait & Resume (ALPHA-ONLY)", "inline": True}
                    ],
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                await self.notifier.send_embed(msg)
                self.pulse_sent = True
            except: pass
            
        await asyncio.sleep(60)
        
        # Reset pulse after hibernation so we can notify again if it happens hours later
        self.pulse_sent = False

    async def _scan_asset(self, ticker: str):
        """Performs a scan and fires an alert if conditions are met."""
        
        # Alert Suppression (Law 3): 8-hour cooldown
        now = datetime.now(timezone.utc)
        if ticker in self.alert_history:
            last_alert = self.alert_history[ticker]
            if now - last_alert < timedelta(hours=self.config.alert_cooldown_hours):
                return
        
        request = EchoScanRequest(
            ticker=ticker,
            timeframe="1h", # Default institutional timeframe
            window_size=60,
            top_n=20,
            include_failure_analysis=True,
            include_projections=True,
        )

        try:
            # We call the echo_scan logic directly
            # Note: This performs real Polygon fetches and pattern matching
            response = await echo_scan(request)
            
            # Institutional Rule 3 & Price Range Refinement
            last_price = response.top_matches[0].c2_close if response.top_matches else 0
            is_mega = ticker in self.config.mega_caps
            is_overpriced = last_price > self.config.alert_max_price
            
            # Decision Logic: Should we alert?
            # Law 3 Compliance: Cadence and noise suppression
            if response.confidence >= self.config.alert_min_confidence:
                # Calculate institutional score to check grade threshold
                score_data = self.generator._calculate_institutional_score(response)
                total_score = score_data["total"]
                grade = self.generator._get_letter_grade(total_score)
                
                # SML Rule: Only fire alerts for Grade C+ and above to ensure intelligence quality
                if total_score >= self.config.grade_thresholds["C+"]:
                    
                    # --- RULE 3 & PRICE OVERRIDE ---
                    # If overpriced, only fire if A+
                    if is_overpriced and grade != "A+":
                        logger.info("Signal suppressed for %s: Price ($%.2f) exceeds ceiling and grade is not A+.", ticker, last_price)
                        return
                    
                    # If mega-cap, respect throttle (Manifesto Rule 3)
                    if is_mega:
                        if self.mega_cap_alerts_in_cycle >= 1:
                            logger.info("Mega Cap signal suppressed for %s: Throttled to 1 per cycle.", ticker)
                            return
                        self.mega_cap_alerts_in_cycle += 1

                    logger.info("High Grade Signal Detected for %s (%s). Generating signal...", 
                                ticker, grade)
                    
                    # MOASS Detection (Institutional Standard)
                    is_moass = total_score >= 90  # Rule 1: ANY ticker can earn MOASS status
                    
                    embed = self.generator.generate_trade_card(response)
                    
                    if is_moass:
                        embed["title"] = f"🚀 MOASS CANDIDATE 🚀 — {embed['title']}"
                        embed["color"] = 0xFF00FF # Purple highlight for MOASS
                        
                    sent = await self.notifier.send_embed(embed)
                    if sent:
                        self.alert_history[ticker] = now
                else:
                    logger.info("Scan complete for %s. Grade below C+ (%s). Suppression active.", 
                                ticker, self.generator._get_letter_grade(total_score))
            else:
                logger.info("Scan complete for %s. Confidence (%d%%) below threshold.", 
                            ticker, int(response.confidence * 100))

        except Exception as e:
            logger.error("Intelligence scan failed for %s: %s", ticker, e)

    def stop(self):
        self.is_running = False
        logger.info("Intelligence Driver Shutdown Requested.")
