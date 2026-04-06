"""
Root entry-point — main.py
===========================
Re-exports the application from app.main so Uvicorn can be started with:

    uvicorn main:app --reload --port 8000

or equivalently:

    uvicorn app.main:app --reload --port 8000
"""

from app.main import app  # noqa: F401 — re-export for Uvicorn discovery

__all__ = ["app"]
