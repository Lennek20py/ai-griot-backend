from sqlalchemy import Column, String, Text, Integer, DateTime, ForeignKey, Enum, Float, Boolean
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
    origin = Column(String, nullable=True)  # Geographic origin as text (e.g., "Nairobi, Kenya")
    geo_location = Column(JSONB, nullable=True)  # {"lat": 0.0, "lng": 0.0, "country": "Ghana"}
    audio_file_url = Column(String, nullable=True)  # Allow null initially, will be populated after upload
    duration_seconds = Column(Integer, nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    status = Column(Enum(StoryStatus), default=StoryStatus.PROCESSING, index=True)
    consent_given = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    contributor = relationship("User", back_populates="stories")
    transcript = relationship("Transcript", back_populates="story", uselist=False)
    translations = relationship("Translation", back_populates="story")
    tags = relationship("Tag", secondary="story_tags", back_populates="stories")
    analytics = relationship("Analytics", back_populates="story")

class Transcript(Base):
    __tablename__ = "transcripts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    story_id = Column(UUID(as_uuid=True), ForeignKey("stories.id"), nullable=False)
    transcript_json = Column(JSONB, nullable=False)  # Timestamped transcript
    language = Column(String, nullable=False)  # Language of the transcript
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
    translation_json = Column(JSONB, nullable=True)  # Word-level timestamps for translation
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    story = relationship("Story", back_populates="translations")

class Tag(Base):
    __tablename__ = "tags"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    stories = relationship("Story", secondary="story_tags", back_populates="tags")

# Association table for many-to-many relationship between stories and tags
from sqlalchemy import Table
story_tags = Table(
    "story_tags",
    Base.metadata,
    Column("story_id", UUID(as_uuid=True), ForeignKey("stories.id"), primary_key=True),
    Column("tag_id", Integer, ForeignKey("tags.id"), primary_key=True)
)

class Analytics(Base):
    __tablename__ = "analytics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    story_id = Column(UUID(as_uuid=True), ForeignKey("stories.id"), nullable=False)
    views = Column(Integer, default=0)
    listens = Column(Integer, default=0)
    avg_rating = Column(Float, default=0.0)
    sentiment_score = Column(Float, nullable=True)
    entities = Column(JSONB, nullable=True)  # Store extracted entities
    keywords = Column(JSONB, nullable=True)  # Store keywords
    cultural_analysis = Column(JSONB, nullable=True)  # Store cultural analysis
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    story = relationship("Story", back_populates="analytics")

# Pydantic models for API requests/responses

class StoryBase(BaseModel):
    title: str
    description: Optional[str] = None
    storyteller_name: Optional[str] = None
    storyteller_bio: Optional[str] = None
    language: str
    origin: Optional[str] = None  # Geographic origin as text (e.g., "Nairobi, Kenya")
    geo_location: Optional[Dict[str, Any]] = None
    consent_given: bool = False

class StoryCreate(StoryBase):
    audio_file_url: Optional[str] = None  # Audio file URL from upload
    file_size_bytes: Optional[int] = None  # File size in bytes
    duration_seconds: Optional[int] = None  # Audio duration in seconds

class StoryUpdate(StoryBase):
    """Pydantic model for story updates"""
    title: Optional[str] = None
    language: Optional[str] = None
    status: Optional[StoryStatus] = None
    audio_file_url: Optional[str] = None
    duration_seconds: Optional[int] = None
    file_size_bytes: Optional[int] = None
    tags: Optional[List[int]] = None  # List of tag IDs

    class Config:
        from_attributes = True

class StoryResponse(StoryBase):
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

class StoryDetailResponse(BaseModel):
    """Comprehensive story response with all related data"""
    # Basic story fields
    id: uuid.UUID
    title: str
    description: Optional[str] = None
    storyteller_name: Optional[str] = None
    storyteller_bio: Optional[str] = None
    language: str
    origin: Optional[str] = None
    geo_location: Optional[Dict[str, Any]] = None
    consent_given: bool = False
    contributor_id: uuid.UUID
    audio_file_url: str
    duration_seconds: Optional[int] = None
    file_size_bytes: Optional[int] = None
    status: StoryStatus
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    # Related data
    contributor: Optional[Dict[str, Any]] = None
    transcript: Optional[Dict[str, Any]] = None
    translations: List[Dict[str, Any]] = []
    tags: List[Dict[str, Any]] = []
    analytics: Optional[Dict[str, Any]] = None

class TranscriptCreate(BaseModel):
    story_id: str
    transcript_json: Dict[str, Any]
    language: str
    confidence_score: Optional[float] = None

class TranscriptResponse(BaseModel):
    id: uuid.UUID
    story_id: uuid.UUID
    transcript_json: Dict[str, Any]
    language: str
    confidence_score: Optional[float] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class TranslationCreate(BaseModel):
    story_id: str
    translated_text: str
    language: str
    confidence_score: Optional[float] = None

class TranslationResponse(BaseModel):
    id: uuid.UUID
    story_id: uuid.UUID
    translated_text: str
    language: str
    confidence_score: Optional[float] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class TagCreate(BaseModel):
    name: str
    description: Optional[str] = None

class TagResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class AnalyticsResponse(BaseModel):
    id: uuid.UUID
    story_id: uuid.UUID
    views: int
    listens: int
    avg_rating: float
    sentiment_score: Optional[float] = None
    entities: Optional[Dict[str, Any]] = None
    keywords: Optional[Dict[str, Any]] = None
    cultural_analysis: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True 