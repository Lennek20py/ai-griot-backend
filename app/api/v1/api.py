from fastapi import APIRouter

from app.api.v1.endpoints import auth, users, stories, media, ai_processing

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(stories.router, prefix="/stories", tags=["stories"])
api_router.include_router(media.router, prefix="/media", tags=["media"])
api_router.include_router(ai_processing.router, prefix="/ai", tags=["ai-processing"]) 