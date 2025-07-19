import boto3
import os
import uuid
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import aiofiles
import tempfile

from app.core.database import get_db
from app.core.security import get_current_active_user, get_current_active_user_optional
from app.core.config import settings
from app.models.user import User
from app.models.story import Story, StoryStatus

router = APIRouter()

# Initialize S3 client
s3_client = None
if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
    s3_client = boto3.client(
        's3',
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_REGION
    )

# Local storage directory
LOCAL_MEDIA_DIR = "media/files"
os.makedirs(LOCAL_MEDIA_DIR, exist_ok=True)
os.makedirs(f"{LOCAL_MEDIA_DIR}/uploads", exist_ok=True)

def get_file_extension(filename: str) -> str:
    """Get file extension from filename."""
    return os.path.splitext(filename)[1].lower()

def validate_audio_file(file: UploadFile) -> bool:
    """Validate if the uploaded file is a supported audio format."""
    if not file.filename:
        return False
    
    extension = get_file_extension(file.filename)
    return extension in settings.ALLOWED_AUDIO_FORMATS

async def save_file_locally(content: bytes, filename: str) -> str:
    """Save file locally and return the full URL."""
    file_path = os.path.join(LOCAL_MEDIA_DIR, filename)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save file
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(content)
    
    # Return full URL that frontend can access
    return f"/media/files/{filename}"

async def upload_to_s3(file_path: str, s3_key: str) -> str:
    """Upload file to S3 and return the URL."""
    if not s3_client:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="S3 client not configured"
        )
    
    try:
        s3_client.upload_file(file_path, settings.S3_BUCKET_NAME, s3_key)
        return f"https://{settings.S3_BUCKET_NAME}.s3.{settings.AWS_REGION}.amazonaws.com/{s3_key}"
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload to S3: {str(e)}"
        )

def generate_presigned_url(s3_key: str, expiration: int = 3600) -> str:
    """Generate a presigned URL for secure file access."""
    if not s3_client:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="S3 client not configured"
        )
    
    try:
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': settings.S3_BUCKET_NAME, 'Key': s3_key},
            ExpiresIn=expiration
        )
        return url
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate presigned URL: {str(e)}"
        )

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user)
):
    """Upload a file and return the URL."""
    # Validate file
    if not validate_audio_file(file):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format. Allowed formats: {', '.join(settings.ALLOWED_AUDIO_FORMATS)}"
        )
    
    # Check file size
    if file.size and file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / (1024*1024)}MB"
        )
    
    # Generate unique filename
    file_extension = get_file_extension(file.filename)
    unique_filename = f"uploads/{uuid.uuid4()}{file_extension}"
    
    try:
        # Read file content
        content = await file.read()
        
        # Try S3 upload first, fall back to local storage
        audio_url = None
        if s3_client:
            try:
                # Save file temporarily for S3 upload
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file_path = temp_file.name
                async with aiofiles.open(temp_file_path, 'wb') as f:
                    await f.write(content)
                
                audio_url = await upload_to_s3(temp_file_path, unique_filename)
                
                # Clean up temp file
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"S3 upload failed, falling back to local storage: {e}")
                audio_url = None
        
        # Fall back to local storage if S3 failed or not configured
        if not audio_url:
            audio_url = await save_file_locally(content, unique_filename)
        
        return {
            "file_url": audio_url,
            "file_size": len(content),
            "message": "File uploaded successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}"
        )


@router.post("/upload-audio")
async def upload_audio_file(
    file: UploadFile = File(...),
    story_id: str = Form(...),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Upload an audio file for a story."""
    # Validate file
    if not validate_audio_file(file):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format. Allowed formats: {', '.join(settings.ALLOWED_AUDIO_FORMATS)}"
        )
    
    # Check file size
    if file.size and file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / (1024*1024)}MB"
        )
    
    # Validate story_id
    try:
        story_uuid = uuid.UUID(story_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid story ID format"
        )
    
    # Get story and verify ownership
    result = await db.execute(select(Story).where(Story.id == story_uuid))
    story = result.scalar_one_or_none()
    
    if not story:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Story not found"
        )
    
    if story.contributor_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to upload audio for this story"
        )
    
    # Generate unique filename
    file_extension = get_file_extension(file.filename)
    unique_filename = f"{story_id}/{uuid.uuid4()}{file_extension}"
    
    try:
        # Read file content
        content = await file.read()
        
        # Try S3 upload first, fall back to local storage
        audio_url = None
        if s3_client:
            try:
                # Save file temporarily for S3 upload
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file_path = temp_file.name
                async with aiofiles.open(temp_file_path, 'wb') as f:
                    await f.write(content)
                
                audio_url = await upload_to_s3(temp_file_path, unique_filename)
                
                # Clean up temp file
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"S3 upload failed, falling back to local storage: {e}")
                audio_url = None
        
        # Fall back to local storage if S3 failed or not configured
        if not audio_url:
            audio_url = await save_file_locally(content, unique_filename)
        
        # Update story with audio file information
        story.audio_file_url = audio_url
        story.file_size_bytes = len(content)
        story.status = StoryStatus.PROCESSING
        
        await db.commit()
        await db.refresh(story)
        
        return {
            "message": "Audio file uploaded successfully",
            "audio_url": audio_url,
            "file_size": len(content),
            "story_id": story_id
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload audio file: {str(e)}"
        )

@router.get("/audio/{story_id}")
async def get_audio_url(
    story_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get a URL for accessing an audio file (local or S3)."""
    try:
        story_uuid = uuid.UUID(story_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid story ID format"
        )
    
    # Get story
    result = await db.execute(select(Story).where(Story.id == story_uuid))
    story = result.scalar_one_or_none()
    
    if not story:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Story not found"
        )
    
    if not story.audio_file_url:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Audio file not found for this story"
        )
    
    # Check if file is stored locally or in S3
    is_local_file = (
        story.audio_file_url.startswith('/media/files/') or
        story.audio_file_url.startswith('media/files/') or 
        story.audio_file_url.startswith('http://localhost:8000/media/files/') or
        story.audio_file_url.startswith('https://localhost:8000/media/files/') or
        '/media/files/' in story.audio_file_url
    )
    
    if is_local_file:
        # Local file - ensure we return the proper URL format
        if story.audio_file_url.startswith('/media/files/'):
            # Already in the correct format
            local_url = story.audio_file_url
        elif story.audio_file_url.startswith('media/files/'):
            # Add leading slash
            local_url = f"/{story.audio_file_url}"
        elif 'media/files/' in story.audio_file_url:
            # Extract the media part from full URLs
            file_path = story.audio_file_url.split('media/files/')[1]
            local_url = f"/media/files/{file_path}"
        else:
            # Fallback
            file_path = story.audio_file_url.replace('media/files/', '')
            local_url = f"/media/files/{file_path}"
        
        return {
            "audio_url": local_url,
            "expires_in": None,  # Local files don't expire
            "story_id": story_id
        }
    else:
        # S3 file - use existing S3 logic
        if not s3_client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="S3 service not configured"
            )
        
        # Extract S3 key from URL
        try:
            s3_key = story.audio_file_url.split(f"{settings.S3_BUCKET_NAME}.s3.{settings.AWS_REGION}.amazonaws.com/")[1]
        except IndexError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid S3 URL format"
            )
        
        # Generate presigned URL
        presigned_url = generate_presigned_url(s3_key)
        
        return {
            "audio_url": presigned_url,
            "expires_in": 3600,
            "story_id": story_id
        }

@router.delete("/audio/{story_id}")
async def delete_audio_file(
    story_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete an audio file (only by story owner)."""
    try:
        story_uuid = uuid.UUID(story_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid story ID format"
        )
    
    # Get story and verify ownership
    result = await db.execute(select(Story).where(Story.id == story_uuid))
    story = result.scalar_one_or_none()
    
    if not story:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Story not found"
        )
    
    if story.contributor_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete audio for this story"
        )
    
    if not story.audio_file_url:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Audio file not found for this story"
        )
    
    # Delete from S3
    if s3_client:
        try:
            s3_key = story.audio_file_url.split(f"{settings.S3_BUCKET_NAME}.s3.{settings.AWS_REGION}.amazonaws.com/")[1]
            s3_client.delete_object(Bucket=settings.S3_BUCKET_NAME, Key=s3_key)
        except Exception as e:
            # Log error but don't fail the request
            print(f"Failed to delete from S3: {str(e)}")
    
    # Update story
    story.audio_file_url = None
    story.file_size_bytes = None
    story.status = StoryStatus.DRAFT
    
    await db.commit()
    
    return {"message": "Audio file deleted successfully"}

@router.get("/health")
async def media_health_check():
    """Check if media service is healthy."""
    s3_status = "connected" if s3_client else "not_configured"
    
    return {
        "status": "healthy",
        "s3_status": s3_status,
        "max_file_size": settings.MAX_FILE_SIZE,
        "allowed_formats": settings.ALLOWED_AUDIO_FORMATS
    }

@router.get("/files/{file_path:path}")
async def serve_media_file(
    file_path: str,
    current_user: Optional[User] = Depends(get_current_active_user_optional)
):
    """Serve media files directly."""
    try:
        # Construct the full file path
        full_path = os.path.join(LOCAL_MEDIA_DIR, file_path)
        
        # Security check: ensure the path is within the media directory
        if not os.path.abspath(full_path).startswith(os.path.abspath(LOCAL_MEDIA_DIR)):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Check if file exists
        if not os.path.exists(full_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        # Return the file
        return FileResponse(
            full_path,
            media_type="audio/wav" if full_path.endswith('.wav') else "audio/mpeg"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to serve file: {str(e)}"
        ) 