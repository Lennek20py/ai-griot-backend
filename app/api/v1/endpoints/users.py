from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from app.core.database import get_db
from app.core.security import get_current_active_user, get_password_hash
from app.models.user import User, UserUpdate, UserResponse
from app.models.story import Story, StoryResponse

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

@router.get("/me/stories", response_model=List[StoryResponse])
async def get_current_user_stories(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    skip: int = 0,
    limit: int = 100
):
    """Get stories contributed by the current user."""
    result = await db.execute(
        select(Story)
        .where(Story.contributor_id == current_user.id)
        .offset(skip)
        .limit(limit)
        .order_by(Story.created_at.desc())
    )
    stories = result.scalars().all()
    
    # Convert to response models with relationships
    story_responses = []
    for story in stories:
        # Get related data
        story_response = StoryResponse.from_orm(story)
        story_response.contributor = UserResponse.from_orm(current_user)
        
        # Get transcript if exists
        if story.transcript:
            story_response.transcript = story.transcript
        
        # Get translations
        story_response.translations = story.translations
        
        # Get tags
        story_response.tags = story.tags
        
        # Get analytics
        if story.analytics:
            story_response.analytics = story.analytics
        
        story_responses.append(story_response)
    
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

@router.get("/{user_id}/stories", response_model=List[StoryResponse])
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
    
    # Get user's stories
    result = await db.execute(
        select(Story)
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
        story_response = StoryResponse.from_orm(story)
        story_response.contributor = UserResponse.from_orm(user)
        
        # Get related data
        if story.transcript:
            story_response.transcript = story.transcript
        story_response.translations = story.translations
        story_response.tags = story.tags
        if story.analytics:
            story_response.analytics = story.analytics
        
        story_responses.append(story_response)
    
    return story_responses 