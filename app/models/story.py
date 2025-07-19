from sqlalchemy import Column, String, Text, Integer, DateTime, ForeignKey, Enum, Float
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime
import enum

from app.core.database import Base

class StoryStatus(str, enum.Enum):
    PROCESSING = "processing"
    PUBLISHED = "published"
    REJECTED = "rejected"
    DRAFT = "draft"

class Story(Base):
    __tablename__ = "stories"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    contributor_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    storyteller_name = Column(String, nullable=True)
    storyteller_bio = Column(Text, nullable=True)
    language = Column(String, nullable=False, index=True)  # e.g., "en-US", "fr-FR"
    geo_location = Column(JSONB, nullable=True)  # {"lat": 0.0, "lng": 0.0, "country": "Ghana"}
    audio_file_url = Column(String, nullable=False)
    duration_seconds = Column(Integer, nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    status = Column(Enum(StoryStatus), default=StoryStatus.PROCESSING, index=True)
    consent_given = Column(String, default=False)  # Boolean as string for flexibility
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    contributor = relationship("User", back_populates="stories")
    transcript = relationship("Transcript", back_populates="story", uselist=False)
    translations = relationship("Translation", back_populates="story")
    tags = relationship("Tag", secondary="story_tags", back_populates="stories")
    analytics = relationship("Analytics", back_populates="story", uselist=False)

class Transcript(Base):
    __tablename__ = "transcripts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    story_id = Column(UUID(as_uuid=True), ForeignKey("stories.id"), nullable=False)
    transcript_json = Column(JSONB, nullable=False)  # Timestamped transcript
    language = Column(String, nullable=False)  # Source language
    confidence_score = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    story = relationship("Story", back_populates="transcript")

class Translation(Base):
    __tablename__ = "translations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    story_id = Column(UUID(as_uuid=True), ForeignKey("stories.id"), nullable=False)
    translated_text = Column(Text, nullable=False)
    language = Column(String, nullable=False)  # Target language
    confidence_score = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    story = relationship("Story", back_populates="translations")

class Tag(Base):
    __tablename__ = "tags"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    stories = relationship("Story", secondary="story_tags", back_populates="tags")

# Association table for many-to-many relationship
class StoryTag(Base):
    __tablename__ = "story_tags"
    
    story_id = Column(UUID(as_uuid=True), ForeignKey("stories.id"), primary_key=True)
    tag_id = Column(UUID(as_uuid=True), ForeignKey("tags.id"), primary_key=True)

class Analytics(Base):
    __tablename__ = "analytics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    story_id = Column(UUID(as_uuid=True), ForeignKey("stories.id"), nullable=False, unique=True)
    views = Column(Integer, default=0)
    listens = Column(Integer, default=0)
    avg_rating = Column(Float, default=0.0)
    total_ratings = Column(Integer, default=0)
    sentiment_score = Column(Float, nullable=True)
    entities = Column(JSONB, nullable=True)  # Named entities from NER
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    story = relationship("Story", back_populates="analytics")

# Pydantic schemas
class StoryBase(BaseModel):
    title: str
    description: Optional[str] = None
    storyteller_name: Optional[str] = None
    storyteller_bio: Optional[str] = None
    language: str
    geo_location: Optional[Dict[str, Any]] = None
    consent_given: bool = False

class StoryCreate(StoryBase):
    pass

class StoryUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    storyteller_name: Optional[str] = None
    storyteller_bio: Optional[str] = None
    language: Optional[str] = None
    geo_location: Optional[Dict[str, Any]] = None
    status: Optional[StoryStatus] = None

class StoryInDB(StoryBase):
    id: uuid.UUID
    contributor_id: uuid.UUID
    audio_file_url: str
    duration_seconds: Optional[int] = None
    file_size_bytes: Optional[int] = None
    status: StoryStatus
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class StoryResponse(StoryInDB):
    contributor: Optional["UserResponse"] = None
    transcript: Optional["TranscriptResponse"] = None
    translations: List["TranslationResponse"] = []
    tags: List["TagResponse"] = []
    analytics: Optional["AnalyticsResponse"] = None

class TranscriptBase(BaseModel):
    transcript_json: Dict[str, Any]
    language: str
    confidence_score: Optional[float] = None

class TranscriptCreate(TranscriptBase):
    story_id: uuid.UUID

class TranscriptResponse(TranscriptBase):
    id: uuid.UUID
    story_id: uuid.UUID
    created_at: datetime
    
    class Config:
        from_attributes = True

class TranslationBase(BaseModel):
    translated_text: str
    language: str
    confidence_score: Optional[float] = None

class TranslationCreate(TranslationBase):
    story_id: uuid.UUID

class TranslationResponse(TranslationBase):
    id: uuid.UUID
    story_id: uuid.UUID
    created_at: datetime
    
    class Config:
        from_attributes = True

class TagBase(BaseModel):
    name: str
    description: Optional[str] = None

class TagCreate(TagBase):
    pass

class TagResponse(TagBase):
    id: uuid.UUID
    created_at: datetime
    
    class Config:
        from_attributes = True

class AnalyticsBase(BaseModel):
    views: int = 0
    listens: int = 0
    avg_rating: float = 0.0
    total_ratings: int = 0
    sentiment_score: Optional[float] = None
    entities: Optional[Dict[str, Any]] = None

class AnalyticsResponse(AnalyticsBase):
    id: uuid.UUID
    story_id: uuid.UUID
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

# Update forward references
StoryResponse.model_rebuild()
TranscriptResponse.model_rebuild()
TranslationResponse.model_rebuild()
TagResponse.model_rebuild()
AnalyticsResponse.model_rebuild() 