"""
FastAPI application factory — app/main.py
==========================================
Creates the FastAPI app, registers CORS middleware, mounts all routers,
and wires up global exception handlers.

Usage (Uvicorn):
    uvicorn app.main:app --reload --port 8000

Production (Gunicorn + UvicornWorker):
    gunicorn app.main:app -k uvicorn.workers.UvicornWorker -w 2 --bind 0.0.0.0:8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, List

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import chat_router, health_router, admin_router, interview_router, resume_router, negotiation_router, roadmap_router, drills_router

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CORS origins
# ---------------------------------------------------------------------------
# Extend this list with your Vercel deployment URLs.
# Use "*" locally for convenience; tighten for production.

_ALLOWED_ORIGINS: List[str] = [
    # local dev
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    # Vercel preview / production URLs — add yours here
    "https://system-design-mastery.vercel.app",
]

# Optionally inject extra origins via environment variable (comma-separated)
_extra = os.getenv("CORS_ORIGINS", "")
if _extra:
    _ALLOWED_ORIGINS.extend([o.strip() for o in _extra.split(",") if o.strip()])


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown hooks)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Run startup tasks before the server starts accepting requests,
    and teardown tasks after the last request is handled.
    """
    logger.info("=== SD Learning Assistant starting up ===")

    # Warm up the FAISS index in the background so the first request isn't slow.
    # This is a fire-and-forget — the retriever handles missing index gracefully.
    try:
        from app.services.retriever_service import retriever_service
        import asyncio
        asyncio.create_task(_warm_retriever(retriever_service))
    except Exception as exc:
        logger.warning("Could not schedule retriever warm-up: %s", exc)

    yield  # application runs here

    logger.info("=== SD Learning Assistant shutting down ===")


async def _warm_retriever(svc) -> None:
    """Load the FAISS index from disk before the first user query."""
    try:
        await svc.retrieve_context("warm-up")
        logger.info("Retriever warm-up complete.")
    except Exception as exc:
        logger.warning("Retriever warm-up failed (index may not exist yet): %s", exc)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""

    application = FastAPI(
        title="SD Learning Assistant API",
        description=(
            "AI-powered system design learning assistant. "
            "Ask concept questions, get full system designs, or request design reviews."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # -------------------------------------------------------------------
    # CORS — allow Vercel frontend and local dev origins
    # -------------------------------------------------------------------
    application.add_middleware(
        CORSMiddleware,
        allow_origins=_ALLOWED_ORIGINS,
        allow_origin_regex=r"https://.*\.vercel\.app",   # wildcard for all Vercel previews
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Session-ID"],
        max_age=600,  # cache preflight for 10 minutes
    )

    # -------------------------------------------------------------------
    # Routers
    # -------------------------------------------------------------------
    application.include_router(health_router)                          # GET  /health
    application.include_router(chat_router,   prefix="/api/v1")        # POST /api/v1/chat
    application.include_router(admin_router,  prefix="/api/v1")        # GET  /api/v1/admin/...
    application.include_router(interview_router, prefix="/api/v1")     # POST /api/v1/interview/...
    application.include_router(resume_router, prefix="/api/v1/resume") # POST /api/v1/resume/review
    application.include_router(negotiation_router, prefix="/api/v1/negotiation") # POST /api/v1/negotiation/chat
    application.include_router(roadmap_router, prefix="/api/v1/roadmap") # POST /api/v1/roadmap/generate
    application.include_router(drills_router, prefix="/api/v1")          # POST /api/v1/drills/...

    # -------------------------------------------------------------------
    # Global exception handlers
    # -------------------------------------------------------------------

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An unexpected internal error occurred.", "error_code": "INTERNAL_ERROR"},
        )

    return application


# ---------------------------------------------------------------------------
# Singleton app instance — imported by Uvicorn / Gunicorn
# ---------------------------------------------------------------------------

app = create_app()
