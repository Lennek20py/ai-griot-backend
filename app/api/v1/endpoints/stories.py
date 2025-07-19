from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, and_
from sqlalchemy.orm import selectinload
from typing import List, Optional
import uuid

from app.core.database import get_db
from app.core.security import get_current_active_user, get_current_active_user_optional
from app.models.user import User, UserResponse
from app.models.story import (
    Story, StoryCreate, StoryUpdate, StoryResponse, StoryDetailResponse, StoryStatus,
    Tag, TagCreate, TagResponse, Analytics, AnalyticsResponse
)

router = APIRouter()

@router.post("/", response_model=StoryDetailResponse, status_code=status.HTTP_201_CREATED)
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
    
    # Manually construct the response to avoid ORM relationship issues
    user_response = UserResponse.from_orm(current_user)
    analytics_response = AnalyticsResponse.from_orm(analytics)
    
    # Create comprehensive response
    response = StoryDetailResponse(
        # Basic story fields
        id=db_story.id,
        title=db_story.title,
        description=db_story.description,
        storyteller_name=db_story.storyteller_name,
        storyteller_bio=db_story.storyteller_bio,
        language=db_story.language,
        origin=db_story.origin,
        geo_location=db_story.geo_location,
        consent_given=db_story.consent_given,
        contributor_id=db_story.contributor_id,
        audio_file_url=db_story.audio_file_url,
        duration_seconds=db_story.duration_seconds,
        file_size_bytes=db_story.file_size_bytes,
        status=db_story.status,
        created_at=db_story.created_at,
        updated_at=db_story.updated_at,
        
        # Related data
        contributor=user_response.dict(),
        transcript=None,  # No transcript initially
        translations=[],  # No translations initially
        tags=[],  # No tags initially
        analytics=analytics_response.dict()
    )
    
    return response

@router.get("/", response_model=List[StoryDetailResponse])
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
        # Extract analytics data to prevent any potential lazy loading issues
        analytics_data = None
        if story.analytics and len(story.analytics) > 0:
            analytics_obj = story.analytics[0]
            analytics_data = {
                "id": str(analytics_obj.id),
                "views": analytics_obj.views,
                "listens": analytics_obj.listens,
                "downloads": getattr(analytics_obj, 'downloads', 0),
                "shares": getattr(analytics_obj, 'shares', 0),
                "likes": getattr(analytics_obj, 'likes', 0),
                "average_rating": analytics_obj.avg_rating,
                "total_ratings": getattr(analytics_obj, 'total_ratings', 0),
                "created_at": analytics_obj.created_at.isoformat(),
                "updated_at": analytics_obj.updated_at.isoformat() if analytics_obj.updated_at else None
            }
        
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
                "id": str(story.contributor.id),
                "email": story.contributor.email,
                "full_name": story.contributor.full_name,
                "bio": story.contributor.bio,
                "is_active": story.contributor.is_active,
                "created_at": story.contributor.created_at.isoformat(),
                "updated_at": story.contributor.updated_at.isoformat() if story.contributor.updated_at else None
            } if story.contributor else None,
            transcript=story.transcript.transcript_json if story.transcript else None,
            translations=[
                {
                    "id": str(t.id),
                    "language": t.language,
                    "text": t.translated_text,
                    "confidence": t.confidence_score,
                    "created_at": t.created_at.isoformat(),
                    "words": t.translation_json.get("words", []) if t.translation_json else []
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
            analytics=analytics_data
        )
        
        story_responses.append(response)
    
    return story_responses

@router.get("/{story_id}", response_model=StoryDetailResponse)
async def get_story(
    story_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_active_user_optional)
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
    
    # Authorization check: 
    # - Published stories can be viewed by anyone (authenticated or not)
    # - Unpublished stories can only be viewed by their contributors
    if story.status != StoryStatus.PUBLISHED:
        if current_user is None or story.contributor_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to view this story"
            )
    
    # Extract analytics data BEFORE incrementing view count to prevent expired object access
    analytics_data = None
    if story.analytics and len(story.analytics) > 0:
        analytics_obj = story.analytics[0]
        analytics_data = {
            "id": str(analytics_obj.id),
            "views": analytics_obj.views,
            "listens": analytics_obj.listens,
            "downloads": getattr(analytics_obj, 'downloads', 0),
            "shares": getattr(analytics_obj, 'shares', 0),
            "likes": getattr(analytics_obj, 'likes', 0),
            "average_rating": analytics_obj.avg_rating,
            "total_ratings": getattr(analytics_obj, 'total_ratings', 0),
            "created_at": analytics_obj.created_at.isoformat(),
            "updated_at": analytics_obj.updated_at.isoformat() if analytics_obj.updated_at else None
        }
    
    # Increment view count only for published stories (AFTER extracting data)
    if story.status == StoryStatus.PUBLISHED and story.analytics and len(story.analytics) > 0:
        story.analytics[0].views += 1
        await db.commit()
        # Update the view count in our extracted data
        if analytics_data:
            analytics_data["views"] += 1
    
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
            "id": str(story.contributor.id),
            "email": story.contributor.email,
            "full_name": story.contributor.full_name,
            "bio": story.contributor.bio,
            "is_active": story.contributor.is_active,
            "created_at": story.contributor.created_at.isoformat(),
            "updated_at": story.contributor.updated_at.isoformat() if story.contributor.updated_at else None
        } if story.contributor else None,
        transcript=story.transcript.transcript_json if story.transcript else None,
        translations=[
            {
                "id": str(t.id),
                "language": t.language,
                "text": t.translated_text,
                "confidence": t.confidence_score,
                "created_at": t.created_at.isoformat(),
                "words": t.translation_json.get("words", []) if t.translation_json else []
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
        analytics=analytics_data
    )
    
    return response

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