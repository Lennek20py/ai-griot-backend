from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.core.security import get_current_active_user, get_password_hash
from app.models.user import User, UserUpdate, UserResponse
from app.models.story import Story, StoryResponse, StoryDetailResponse

router = APIRouter()

@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: User = Depends(get_current_active_user)
):
    """Get current user's profile."""
    return current_user

@router.put("/me", response_model=UserResponse)
async def update_current_user_profile(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Update current user's profile."""
    update_data = user_update.dict(exclude_unset=True)
    
    # Hash password if provided
    if "password" in update_data:
        update_data["password_hash"] = get_password_hash(update_data.pop("password"))
    
    # Update user fields
    for field, value in update_data.items():
        setattr(current_user, field, value)
    
    await db.commit()
    await db.refresh(current_user)
    
    return current_user

@router.get("/me/stories", response_model=List[StoryDetailResponse])
async def get_current_user_stories(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    skip: int = 0,
    limit: int = 100
):
    """Get stories contributed by the current user."""
    result = await db.execute(
        select(Story).options(
            selectinload(Story.contributor),
            selectinload(Story.transcript),
            selectinload(Story.translations),
            selectinload(Story.tags),
            selectinload(Story.analytics)
        )
        .where(Story.contributor_id == current_user.id)
        .offset(skip)
        .limit(limit)
        .order_by(Story.created_at.desc())
    )
    stories = result.scalars().all()
    
    # Convert to response models with relationships
    story_responses = []
    for story in stories:
        # Create comprehensive response manually to avoid ORM relationship issues
        response = StoryDetailResponse(
            # Basic story fields
            id=story.id,
            title=story.title,
            description=story.description,
            storyteller_name=story.storyteller_name,
            storyteller_bio=story.storyteller_bio,
            language=story.language,
            origin=story.origin,
            consent_given=story.consent_given,
            contributor_id=story.contributor_id,
            audio_file_url=story.audio_file_url,
            duration_seconds=story.duration_seconds,
            file_size_bytes=story.file_size_bytes,
            status=story.status,
            created_at=story.created_at,
            updated_at=story.updated_at,
            
            # Related data
            contributor={
                "id": str(current_user.id),
                "email": current_user.email,
                "full_name": current_user.full_name,
                "bio": current_user.bio,
                "is_active": current_user.is_active,
                "created_at": current_user.created_at.isoformat(),
                "updated_at": current_user.updated_at.isoformat() if current_user.updated_at else None
            },
            transcript=story.transcript.transcript_json if story.transcript else None,
            translations=[
                {
                    "id": str(t.id),
                    "language": t.language,
                    "text": t.translated_text,
                    "confidence": t.confidence_score,
                    "created_at": t.created_at.isoformat()
                } for t in story.translations
            ] if story.translations else [],
            tags=[
                {
                    "id": str(t.id),
                    "name": t.name,
                    "description": t.description,
                    "created_at": t.created_at.isoformat()
                } for t in story.tags
            ] if story.tags else [],
            analytics={
                "id": str(story.analytics[0].id),
                "views": story.analytics[0].views,
                "listens": story.analytics[0].listens,
                "downloads": getattr(story.analytics[0], 'downloads', 0),
                "shares": getattr(story.analytics[0], 'shares', 0),
                "likes": getattr(story.analytics[0], 'likes', 0),
                "average_rating": story.analytics[0].avg_rating,
                "total_ratings": getattr(story.analytics[0], 'total_ratings', 0),
                "created_at": story.analytics[0].created_at.isoformat(),
                "updated_at": story.analytics[0].updated_at.isoformat() if story.analytics[0].updated_at else None
            } if story.analytics and len(story.analytics) > 0 else None
        )
        
        story_responses.append(response)
    
    return story_responses

@router.get("/{user_id}", response_model=UserResponse)
async def get_user_profile(
    user_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get a user's public profile."""
    try:
        import uuid
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID format"
        )
    
    result = await db.execute(select(User).where(User.id == user_uuid))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return user

@router.get("/{user_id}/stories", response_model=List[StoryDetailResponse])
async def get_user_stories(
    user_id: str,
    db: AsyncSession = Depends(get_db),
    skip: int = 0,
    limit: int = 100
):
    """Get stories contributed by a specific user."""
    try:
        import uuid
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID format"
        )
    
    # Check if user exists
    result = await db.execute(select(User).where(User.id == user_uuid))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Get user's stories with relationships
    result = await db.execute(
        select(Story).options(
            selectinload(Story.contributor),
            selectinload(Story.transcript),
            selectinload(Story.translations),
            selectinload(Story.tags),
            selectinload(Story.analytics)
        )
        .where(Story.contributor_id == user_uuid)
        .where(Story.status == "published")
        .offset(skip)
        .limit(limit)
        .order_by(Story.created_at.desc())
    )
    stories = result.scalars().all()
    
    # Convert to response models with relationships
    story_responses = []
    for story in stories:
        # Create comprehensive response manually to avoid ORM relationship issues
        response = StoryDetailResponse(
            # Basic story fields
            id=story.id,
            title=story.title,
            description=story.description,
            storyteller_name=story.storyteller_name,
            storyteller_bio=story.storyteller_bio,
            language=story.language,
            origin=story.origin,
            consent_given=story.consent_given,
            contributor_id=story.contributor_id,
            audio_file_url=story.audio_file_url,
            duration_seconds=story.duration_seconds,
            file_size_bytes=story.file_size_bytes,
            status=story.status,
            created_at=story.created_at,
            updated_at=story.updated_at,
            
            # Related data
            contributor={
                "id": str(user.id),
                "email": user.email,
                "full_name": user.full_name,
                "bio": user.bio,
                "is_active": user.is_active,
                "created_at": user.created_at.isoformat(),
                "updated_at": user.updated_at.isoformat() if user.updated_at else None
            },
            transcript=story.transcript.transcript_json if story.transcript else None,
            translations=[
                {
                    "id": str(t.id),
                    "language": t.language,
                    "text": t.translated_text,
                    "confidence": t.confidence_score,
                    "created_at": t.created_at.isoformat()
                } for t in story.translations
            ] if story.translations else [],
            tags=[
                {
                    "id": str(t.id),
                    "name": t.name,
                    "description": t.description,
                    "created_at": t.created_at.isoformat()
                } for t in story.tags
            ] if story.tags else [],
            analytics={
                "id": str(story.analytics[0].id),
                "views": story.analytics[0].views,
                "listens": story.analytics[0].listens,
                "downloads": getattr(story.analytics[0], 'downloads', 0),
                "shares": getattr(story.analytics[0], 'shares', 0),
                "likes": getattr(story.analytics[0], 'likes', 0),
                "average_rating": story.analytics[0].avg_rating,
                "total_ratings": getattr(story.analytics[0], 'total_ratings', 0),
                "created_at": story.analytics[0].created_at.isoformat(),
                "updated_at": story.analytics[0].updated_at.isoformat() if story.analytics[0].updated_at else None
            } if story.analytics and len(story.analytics) > 0 else None
        )
        
        story_responses.append(response)
    
    return story_responses 