"""
ECHO FORGE — Application Entry Point
Cross-asset pattern memory and echo-matching engine.
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_config
from app.routes.echo_scan import router as echo_scan_router
from app.routes.replay import router as replay_router
from app.storage.models import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize resources on startup, clean up on shutdown."""
    config = get_config()
    await init_db(config)
    
    # Start the Intelligence Driver (The Brain) in the background
    from app.services.intelligence_driver import IntelligenceDriver
    driver = IntelligenceDriver()
    task = asyncio.create_task(driver.start_loop())
    
    yield
    
    # Cleanup
    driver.stop()
    await task


def create_app() -> FastAPI:
    config = get_config()

    app = FastAPI(
        title=config.api_title,
        description=config.api_description,
        version=config.api_version,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(echo_scan_router, prefix="/echo_scan", tags=["Echo Scan"])
    app.include_router(replay_router, prefix="/replay", tags=["Replay"])

    @app.get("/health")
    async def health():
        return {"status": "operational", "system": "ECHO FORGE", "version": config.api_version}

    return app


app = create_app()
