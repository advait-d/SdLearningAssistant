"""
app.api.routes — route blueprint registry.
"""

from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router
from app.api.routes.admin import router as admin_router
from app.api.routes.interview import router as interview_router
from app.api.routes.resume import router as resume_router
from app.api.routes.negotiation import router as negotiation_router
from app.api.routes.roadmap import router as roadmap_router
from app.api.routes.drills import router as drills_router
from app.api.routes.whiteboard import router as whiteboard_router

__all__ = ["chat_router", "health_router", "admin_router", "interview_router", "resume_router", "negotiation_router", "roadmap_router", "drills_router", "whiteboard_router"]
