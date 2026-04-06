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


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    version: str = "1.0.0"


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
)
async def health() -> HealthResponse:
    """Returns 200 OK with basic service metadata."""
    return HealthResponse(
        status="ok",
        uptime_seconds=round(time.time() - _START_TIME, 2),
    )
