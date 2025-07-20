from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import uvicorn
import os

from app.core.config import settings
from app.core.database import engine, Base
from app.api.v1.api import api_router
from app.core.security import get_current_user
from app.models.user import User

# Create database tables
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    # Shutdown
    await engine.dispose()

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    openapi_url="/api/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for media
media_dir = "media"
if not os.path.exists(media_dir):
    os.makedirs(media_dir, exist_ok=True)
    os.makedirs(f"{media_dir}/files", exist_ok=True)
    os.makedirs(f"{media_dir}/files/uploads", exist_ok=True)

app.mount("/media", StaticFiles(directory=media_dir), name="media")

# Include API routes
app.include_router(api_router, prefix="/api/v1")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": settings.VERSION}

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to Digital Griot API",
        "version": "1.0.0",
        "docs": "/docs"
    }

# Media serving endpoint for illustrations
@app.get("/media/illustrations/{filename}")
async def serve_illustration(filename: str):
    """Serve generated illustration images"""
    file_path = f"media/illustrations/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/png")
    else:
        # Return a placeholder image if file doesn't exist
        placeholder_path = "media/placeholder.png"
        if os.path.exists(placeholder_path):
            return FileResponse(placeholder_path, media_type="image/png")
        else:
            # Create a simple placeholder
            return {"error": "Image not found"}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
