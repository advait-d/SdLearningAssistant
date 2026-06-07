"""
app.api.routes — route blueprint registry.
"""

from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router
from app.api.routes.admin import router as admin_router
from app.api.routes.interview import router as interview_router

__all__ = ["chat_router", "health_router", "admin_router", "interview_router"]

