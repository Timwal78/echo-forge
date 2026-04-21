"""
ECHO FORGE — Discord Notifier Service
Robust delivery of intelligence signals to Discord.
"""

import logging
import time
from datetime import datetime
from typing import Optional

import httpx

logger = logging.getLogger("echoforge.services.notifier")


class DiscordNotifier:
    """
    Handles secure delivery of rich embeds to Discord webhooks.
    Includes rate-limit handling and institutional logging.
    """

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.rate_limit_until = 0.0

    async def send_embed(self, embed: dict) -> bool:
        """
        Send a single rich embed to the configured webhook.
        """
        if not self.webhook_url or self.webhook_url == "REPLACE_ME":
            logger.warning("Discord Alert Skipped: No webhook URL configured.")
            return False

        now = time.time()
        if now < self.rate_limit_until:
            logger.warning("Discord Alert Skipped: Rate limited for another %.2fs", self.rate_limit_until - now)
            return False

        payload = {
            "embeds": [embed],
            "username": "ECHO FORGE — Intelligence",
            "avatar_url": "https://i.imgur.com/8dd294b.png", # Placeholder high-tech icon
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                resp = await client.post(self.webhook_url, json=payload)
                
                if resp.status_code == 204:
                    logger.info("Discord Alert Sent Successfully: %s", embed.get("title"))
                    return True
                
                if resp.status_code == 429:
                    retry_after = resp.json().get("retry_after", 5)
                    self.rate_limit_until = time.time() + retry_after
                    logger.warning("Discord Rate Limit Hit. Pausing for %ds", retry_after)
                    return False
                
                logger.error("Discord API Error (%d): %s", resp.status_code, resp.text)
                return False

            except Exception as e:
                logger.error("Failed to deliver Discord alert: %s", e)
                return False
