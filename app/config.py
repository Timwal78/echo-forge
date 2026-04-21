"""
ECHO FORGE — Configuration Module
Central configuration for all system components.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()


@dataclass
class DatabaseConfig:
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    user: str = os.getenv("POSTGRES_USER", "echoforge")
    password: str = os.getenv("POSTGRES_PASSWORD", "echoforge_dev")
    database: str = os.getenv("POSTGRES_DB", "echoforge")

    @property
    def dsn(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def sync_dsn(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class RedisConfig:
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    db: int = int(os.getenv("REDIS_DB", "0"))

    @property
    def url(self) -> str:
        return f"redis://{self.host}:{self.port}/{self.db}"


@dataclass
class WindowConfig:
    """Configuration for pattern window extraction."""
    default_window_size: int = int(os.getenv("ECHO_DEFAULT_WINDOW_SIZE", "60"))
    min_window_size: int = int(os.getenv("ECHO_MIN_WINDOW_SIZE", "20"))
    max_window_size: int = int(os.getenv("ECHO_MAX_WINDOW_SIZE", "200"))
    overlap_ratio: float = float(os.getenv("ECHO_OVERLAP_RATIO", "0.5"))
    forward_horizon: int = int(os.getenv("ECHO_FORWARD_HORIZON", "20"))  # bars to look ahead for outcome labeling


@dataclass
class FeatureConfig:
    """Configuration for feature engineering pipeline."""
    volatility_lookbacks: list[int] = field(default_factory=lambda: [5, 10, 20])
    momentum_lookbacks: list[int] = field(default_factory=lambda: [5, 10, 20])
    volume_lookbacks: list[int] = field(default_factory=lambda: [5, 10, 20])
    compression_threshold: float = float(os.getenv("ECHO_COMPRESSION_THRESH", "0.3"))
    expansion_threshold: float = float(os.getenv("ECHO_EXPANSION_THRESH", "0.7"))
    feature_vector_dim: int = 128
    # Meme Stock Cycle Analysis (Zero-Fake Data Policy)
    cycle_reference_date: str = os.getenv("CYCLE_REFERENCE_DATE", "2021-01-27")
    cycle_period_days: float = float(os.getenv("CYCLE_PERIOD_DAYS", "147.0"))


@dataclass
class SimilarityConfig:
    """Configuration for similarity matching."""
    top_n_matches: int = int(os.getenv("ECHO_TOP_N_MATCHES", "20"))
    min_similarity_threshold: float = float(os.getenv("ECHO_MIN_SIM_THRESH", "0.5"))
    cosine_weight: float = float(os.getenv("ECHO_COSINE_WEIGHT", "0.5"))
    euclidean_weight: float = float(os.getenv("ECHO_EUCLIDEAN_WEIGHT", "0.3"))
    dtw_weight: float = float(os.getenv("ECHO_DTW_WEIGHT", "0.2"))
    cross_asset_enabled: bool = os.getenv("ECHO_CROSS_ASSET", "True").lower() == "true"
    time_scale_invariance: bool = os.getenv("ECHO_TIME_INVARIANCE", "True").lower() == "true"


@dataclass
class ClusteringConfig:
    """Configuration for outcome clustering."""
    n_clusters: int = int(os.getenv("ECHO_N_CLUSTERS", "5"))
    method: str = os.getenv("ECHO_CLUSTERING_METHOD", "kmeans")  # kmeans | hierarchical
    min_cluster_size: int = int(os.getenv("ECHO_MIN_CLUSTER_SIZE", "3"))


@dataclass
class ProjectionConfig:
    """Configuration for forward projections."""
    confidence_threshold: float = float(os.getenv("ECHO_PROJ_CONF_THRESH", "0.6"))
    min_historical_matches: int = int(os.getenv("ECHO_PROJ_MIN_MATCHES", "5"))
    max_projection_bars: int = int(os.getenv("ECHO_PROJ_MAX_BARS", "50"))
    # Probability Thresholds (Institutional Hardening)
    reversal_ret_threshold: float = float(os.getenv("ECHO_REVERSAL_THRESH", "-0.02"))
    failure_drawdown_threshold: float = float(os.getenv("ECHO_FAILURE_DD_THRESH", "-0.05"))


@dataclass
class EchoForgeConfig:
    """Master configuration container."""
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    debug: bool = os.getenv("ECHO_FORGE_DEBUG", "false").lower() == "true"
    api_version: str = "0.1.0"
    api_title: str = "ECHO FORGE"
    api_description: str = (
        "Cross-asset pattern memory and echo-matching engine. "
        "Detects structurally similar market conditions across time, "
        "regimes, and instruments."
    )

    # ── Top-level env-driven settings ─────────────────────────────────────────
    # DATABASE_URL overrides the individual DatabaseConfig fields.
    # Defaults to SQLite for local development — no Postgres required.
    # Production: set DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/echoforge
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            "sqlite+aiosqlite:///./echoforge.db",
        )
    )

    # Polygon.io API key — used by the live OHLCV fetcher.
    # STRICT DATA INTEGRITY: Synthetic fallback removed.
    polygon_key: str = field(
        default_factory=lambda: os.getenv("POLYGON_KEY", "")
    )

    # ── Institutional Rule 3: Mega Cap Benchmarks ──────────────────────────
    mega_caps: set[str] = field(
        default_factory=lambda: {
            'AAPL', 'MSFT', 'GOOG', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'LLY', 'V', 'MA', 
            'AVGO', 'HD', 'COST', 'JPM', 'UNH', 'WMT', 'BAC', 'XOM', 'CVX', 'PG', 'ORCL', 
            'ABBV', 'CRM', 'ADBE', 'NFLX', 'AMD', 'INTC', 'DIS', 'PFE', 'KO', 'PEP', 'CSCO', 
            'TMO', 'AZN', 'NKE', 'ABT', 'LIN', 'DHR', 'WFC', 'MRK', 'VZ', 'T', 'NVR', 'BKNG'
        }
    )
    alert_max_price: float = field(
        default_factory=lambda: float(os.getenv("ALERT_MAX_PRICE", "350.0"))
    )
    price_tier_small: tuple[float, float] = (1.0, 15.0)
    price_tier_mid: tuple[float, float] = (15.0, 150.0)

    # ── Market Discovery & Integration ────────────────────────────────────
    squeeze_os_url: str = field(
        default_factory=lambda: os.getenv("SQUEEZE_OS_URL", "http://localhost:8182")
    )
    enable_dynamic_watchlist: bool = field(
        default_factory=lambda: os.getenv("ENABLE_DYNAMIC_WATCHLIST", "true").lower() == "true"
    )
    discovery_max_tickers: int = field(
        default_factory=lambda: int(os.getenv("ECHO_DISCOVERY_LIMIT", "5"))
    )
    alert_cooldown_hours: float = field(
        default_factory=lambda: float(os.getenv("ALERT_COOLDOWN_HOURS", "8.0"))
    )

    # ── Alerting & Notifications ──────────────────────────────────────────
    discord_webhook_echo: str = field(
        default_factory=lambda: os.getenv("DISCORD_WEBHOOK_ECHO", "")
    )
    watchlist: list[str] = field(
        default_factory=lambda: os.getenv("ECHO_WATCHLIST", "SPY,QQQ,BTC-USD,ETH-USD").split(",")
    )
    scan_interval_sec: int = field(
        default_factory=lambda: int(os.getenv("ECHO_SCAN_INTERVAL", "300"))
    )
    alert_min_confidence: float = field(
        default_factory=lambda: float(os.getenv("ALERT_MIN_CONFIDENCE", "0.60"))
    )

    # ── Institutional Grading & Compliance ────────────────────────────────
    weight_similarity: float = field(
        default_factory=lambda: float(os.getenv("ECHO_WEIGHT_SIMILARITY", "0.30"))
    )
    weight_edge: float = field(
        default_factory=lambda: float(os.getenv("ECHO_WEIGHT_EDGE", "0.25"))
    )
    weight_probability: float = field(
        default_factory=lambda: float(os.getenv("ECHO_WEIGHT_PROBABILITY", "0.20"))
    )
    weight_cycle: float = field(
        default_factory=lambda: float(os.getenv("ECHO_WEIGHT_CYCLE", "0.15"))
    )
    weight_confidence: float = field(
        default_factory=lambda: float(os.getenv("ECHO_WEIGHT_CONFIDENCE", "0.10"))
    )

    # ── Institutional Risk Management ─────────────────────────────────────
    risk_buffer_min: float = field(
        default_factory=lambda: float(os.getenv("ECHO_RISK_BUFFER_MIN", "0.02"))
    )
    rr_buffer_mult: float = field(
        default_factory=lambda: float(os.getenv("ECHO_RR_BUFFER_MULT", "1.5"))
    )

    grade_thresholds: dict[str, int] = field(
        default_factory=lambda: {
            "A+": int(os.getenv("GRADE_APLUS", "93")),
            "A": int(os.getenv("GRADE_A", "87")),
            "B+": int(os.getenv("GRADE_BPLUS", "83")),
            "B": int(os.getenv("GRADE_B", "75")),
            "C+": int(os.getenv("GRADE_CPLUS", "67")),
            "C": int(os.getenv("GRADE_C", "55")),
            "D": int(os.getenv("GRADE_D", "40"))
        }
    )

    @property
    def async_database_url(self) -> str:
        url = self.database_url
        if url.startswith("sqlite:///") and "+aiosqlite" not in url:
            return url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
        if url.startswith("postgres://"):
            return url.replace("postgres://", "postgresql+asyncpg://", 1)
        if url.startswith("postgresql://") and "+asyncpg" not in url:
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url


@lru_cache()
def get_config() -> EchoForgeConfig:
    """Return singleton configuration instance."""
    return EchoForgeConfig()
