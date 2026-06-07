"""
GET /health — lightweight liveness / readiness probe.
"""

from __future__ import annotations

import time
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["Health"])

# Record startup time once at module import
_START_TIME = time.time()

# Seconds after startup during which the backend is considered "warming up"
# (FAISS index loading, model client init, etc.)
_WARMUP_SECONDS = 60


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    version: str = "1.0.0"
    ready: bool = True
    warming_up: bool = False


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
)
async def health() -> HealthResponse:
    """Returns 200 OK with basic service metadata.

    - **warming_up** is True for the first 60 s after startup while the FAISS
      index and LLM clients finish initialising.
    - **ready** mirrors warming_up (False while warming, True once ready).
    """
    uptime = round(time.time() - _START_TIME, 2)
    warming = uptime < _WARMUP_SECONDS
    return HealthResponse(
        status="ok",
        uptime_seconds=uptime,
        warming_up=warming,
        ready=not warming,
    )
