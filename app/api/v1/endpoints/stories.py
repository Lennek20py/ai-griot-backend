from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, and_
from sqlalchemy.orm import selectinload
from typing import List, Optional
import uuid

from app.core.database import get_db
from app.core.security import get_current_active_user
from app.models.user import User, UserResponse
from app.models.story import (
    Story, StoryCreate, StoryUpdate, StoryResponse, StoryStatus,
    Tag, TagCreate, TagResponse, Analytics, AnalyticsResponse
)

router = APIRouter()

@router.post("/", response_model=StoryResponse, status_code=status.HTTP_201_CREATED)
async def create_story(
    story_data: StoryCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new story."""
    # Create story
    db_story = Story(
        **story_data.dict(),
        contributor_id=current_user.id,
        status=StoryStatus.PROCESSING
    )
    
    db.add(db_story)
    await db.commit()
    await db.refresh(db_story)
    
    # Create analytics record
    analytics = Analytics(story_id=db_story.id)
    db.add(analytics)
    await db.commit()
    
    # Return story with relationships
    story_response = StoryResponse.from_orm(db_story)
    story_response.contributor = UserResponse.from_orm(current_user)
    story_response.analytics = AnalyticsResponse.from_orm(analytics)
    
    return story_response

@router.get("/", response_model=List[StoryResponse])
async def get_stories(
    db: AsyncSession = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    language: Optional[str] = None,
    status: Optional[StoryStatus] = None,
    search: Optional[str] = None,
    tag: Optional[str] = None
):
    """Get stories with filtering and search."""
    query = select(Story).options(
        selectinload(Story.contributor),
        selectinload(Story.transcript),
        selectinload(Story.translations),
        selectinload(Story.tags),
        selectinload(Story.analytics)
    )
    
    # Apply filters
    filters = []
    
    if language:
        filters.append(Story.language == language)
    
    if status:
        filters.append(Story.status == status)
    else:
        # Default to published stories for public access
        filters.append(Story.status == StoryStatus.PUBLISHED)
    
    if search:
        search_filter = or_(
            Story.title.ilike(f"%{search}%"),
            Story.description.ilike(f"%{search}%"),
            Story.storyteller_name.ilike(f"%{search}%")
        )
        filters.append(search_filter)
    
    if tag:
        # Filter by tag (this would need a join in a more complex implementation)
        pass
    
    if filters:
        query = query.where(and_(*filters))
    
    # Apply pagination and ordering
    query = query.offset(skip).limit(limit).order_by(Story.created_at.desc())
    
    result = await db.execute(query)
    stories = result.scalars().all()
    
    # Convert to response models
    story_responses = []
    for story in stories:
        story_response = StoryResponse.from_orm(story)
        story_response.contributor = UserResponse.from_orm(story.contributor)
        
        if story.transcript:
            story_response.transcript = story.transcript
        story_response.translations = story.translations
        story_response.tags = story.tags
        if story.analytics:
            story_response.analytics = story.analytics
        
        story_responses.append(story_response)
    
    return story_responses

@router.get("/{story_id}", response_model=StoryResponse)
async def get_story(
    story_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific story by ID."""
    try:
        story_uuid = uuid.UUID(story_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid story ID format"
        )
    
    result = await db.execute(
        select(Story).options(
            selectinload(Story.contributor),
            selectinload(Story.transcript),
            selectinload(Story.translations),
            selectinload(Story.tags),
            selectinload(Story.analytics)
        ).where(Story.id == story_uuid)
    )
    story = result.scalar_one_or_none()
    
    if not story:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Story not found"
        )
    
    # Increment view count
    if story.analytics:
        story.analytics.views += 1
        await db.commit()
    
    # Convert to response model
    story_response = StoryResponse.from_orm(story)
    story_response.contributor = UserResponse.from_orm(story.contributor)
    
    if story.transcript:
        story_response.transcript = story.transcript
    story_response.translations = story.translations
    story_response.tags = story.tags
    if story.analytics:
        story_response.analytics = story.analytics
    
    return story_response

@router.put("/{story_id}", response_model=StoryResponse)
async def update_story(
    story_id: str,
    story_update: StoryUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Update a story (only by owner)."""
    try:
        story_uuid = uuid.UUID(story_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid story ID format"
        )
    
    result = await db.execute(select(Story).where(Story.id == story_uuid))
    story = result.scalar_one_or_none()
    
    if not story:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Story not found"
        )
    
    # Check ownership
    if story.contributor_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this story"
        )
    
    # Update story fields
    update_data = story_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(story, field, value)
    
    await db.commit()
    await db.refresh(story)
    
    # Return updated story
    story_response = StoryResponse.from_orm(story)
    story_response.contributor = UserResponse.from_orm(current_user)
    
    return story_response

@router.delete("/{story_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_story(
    story_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a story (only by owner)."""
    try:
        story_uuid = uuid.UUID(story_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid story ID format"
        )
    
    result = await db.execute(select(Story).where(Story.id == story_uuid))
    story = result.scalar_one_or_none()
    
    if not story:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Story not found"
        )
    
    # Check ownership
    if story.contributor_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this story"
        )
    
    await db.delete(story)
    await db.commit()

# Tag endpoints
@router.get("/tags/", response_model=List[TagResponse])
async def get_tags(
    db: AsyncSession = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
):
    """Get all tags."""
    result = await db.execute(
        select(Tag).offset(skip).limit(limit).order_by(Tag.name)
    )
    tags = result.scalars().all()
    return [TagResponse.from_orm(tag) for tag in tags]

@router.post("/tags/", response_model=TagResponse, status_code=status.HTTP_201_CREATED)
async def create_tag(
    tag_data: TagCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new tag."""
    # Check if tag already exists
    result = await db.execute(select(Tag).where(Tag.name == tag_data.name))
    existing_tag = result.scalar_one_or_none()
    
    if existing_tag:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tag already exists"
        )
    
    db_tag = Tag(**tag_data.dict())
    db.add(db_tag)
    await db.commit()
    await db.refresh(db_tag)
    
    return TagResponse.from_orm(db_tag) 