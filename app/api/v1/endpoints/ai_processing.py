import asyncio
import json
import uuid
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import openai
from google.cloud import translate_v2 as translate
from google.cloud import speech
import spacy
from transformers import pipeline

from app.core.database import get_db
from app.core.security import get_current_active_user
from app.core.config import settings
from app.models.user import User
from app.models.story import (
    Story, StoryStatus, Transcript, Translation, Analytics,
    TranscriptCreate, TranslationCreate
)

router = APIRouter()

# Initialize AI clients
openai_client = None
if settings.OPENAI_API_KEY:
    openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

translate_client = None
if settings.GOOGLE_APPLICATION_CREDENTIALS:
    translate_client = translate.Client()

speech_client = None
if settings.GOOGLE_APPLICATION_CREDENTIALS:
    speech_client = speech.SpeechClient()

# Load spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None

# Initialize sentiment analysis pipeline
try:
    sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
except Exception:
    sentiment_analyzer = None

async def transcribe_audio_with_whisper(audio_url: str, language: str) -> Dict[str, Any]:
    """Transcribe audio using OpenAI Whisper."""
    if not openai_client:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OpenAI client not configured"
        )
    
    try:
        # Download audio file (simplified - in production, you'd stream from S3)
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(audio_url)
            response.raise_for_status()
            audio_data = response.content
        
        # Create temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        try:
            # Transcribe with Whisper
            with open(temp_file_path, "rb") as audio_file:
                transcript = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language.split("-")[0] if "-" in language else language,
                    response_format="verbose_json"
                )
            
            return {
                "text": transcript.text,
                "segments": transcript.segments,
                "language": transcript.language,
                "duration": transcript.duration
            }
        finally:
            import os
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}"
        )

async def transcribe_audio_with_google(audio_url: str, language: str) -> Dict[str, Any]:
    """Transcribe audio using Google Speech-to-Text."""
    if not speech_client:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Google Speech client not configured"
        )
    
    try:
        # Download audio file
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(audio_url)
            response.raise_for_status()
            audio_data = response.content
        
        # Configure recognition
        audio = speech.RecognitionAudio(content=audio_data)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.MP3,
            sample_rate_hertz=16000,
            language_code=language,
            enable_word_time_offsets=True,
            enable_automatic_punctuation=True
        )
        
        # Perform transcription
        response = speech_client.recognize(config=config, audio=audio)
        
        # Process results
        segments = []
        full_text = ""
        
        for result in response.results:
            full_text += result.alternatives[0].transcript + " "
            
            for word_info in result.alternatives[0].words:
                segments.append({
                    "start": word_info.start_time.total_seconds(),
                    "end": word_info.end_time.total_seconds(),
                    "text": word_info.word
                })
        
        return {
            "text": full_text.strip(),
            "segments": segments,
            "language": language
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Google transcription failed: {str(e)}"
        )

async def translate_text(text: str, target_language: str, source_language: str = "auto") -> str:
    """Translate text using Google Translate."""
    if not translate_client:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Google Translate client not configured"
        )
    
    try:
        result = translate_client.translate(
            text,
            target_language=target_language,
            source_language=source_language
        )
        return result["translatedText"]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {str(e)}"
        )

def analyze_sentiment(text: str) -> float:
    """Analyze sentiment of text."""
    if not sentiment_analyzer:
        return 0.0
    
    try:
        result = sentiment_analyzer(text)
        # Convert sentiment to score (-1 to 1)
        if result[0]["label"] == "LABEL_0":  # Negative
            return -result[0]["score"]
        elif result[0]["label"] == "LABEL_2":  # Positive
            return result[0]["score"]
        else:  # Neutral
            return 0.0
    except Exception:
        return 0.0

def extract_entities(text: str) -> List[Dict[str, Any]]:
    """Extract named entities from text."""
    if not nlp:
        return []
    
    try:
        doc = nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        
        return entities
    except Exception:
        return []

@router.post("/transcribe/{story_id}")
async def transcribe_story(
    story_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Start transcription for a story."""
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
            detail="Not authorized to transcribe this story"
        )
    
    if not story.audio_file_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No audio file found for this story"
        )
    
    # Check if transcript already exists
    result = await db.execute(select(Transcript).where(Transcript.story_id == story_uuid))
    existing_transcript = result.scalar_one_or_none()
    
    if existing_transcript:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Transcript already exists for this story"
        )
    
    # Start transcription in background
    background_tasks.add_task(
        process_transcription,
        story_id=story_id,
        audio_url=story.audio_file_url,
        language=story.language
    )
    
    return {"message": "Transcription started", "story_id": story_id}

async def process_transcription(story_id: str, audio_url: str, language: str):
    """Background task to process transcription."""
    from app.core.database import AsyncSessionLocal
    
    async with AsyncSessionLocal() as db:
        try:
            # Try OpenAI Whisper first, fallback to Google
            try:
                transcript_data = await transcribe_audio_with_whisper(audio_url, language)
            except Exception:
                transcript_data = await transcribe_audio_with_google(audio_url, language)
            
            # Create transcript record
            transcript = Transcript(
                story_id=uuid.UUID(story_id),
                transcript_json=transcript_data,
                language=language,
                confidence_score=0.9  # Placeholder
            )
            
            db.add(transcript)
            await db.commit()
            
            # Start translation and analysis
            asyncio.create_task(process_translations(story_id, transcript_data["text"], language))
            asyncio.create_task(process_analysis(story_id, transcript_data["text"]))
            
        except Exception as e:
            print(f"Transcription failed for story {story_id}: {str(e)}")

async def process_translations(story_id: str, text: str, source_language: str):
    """Background task to process translations."""
    from app.core.database import AsyncSessionLocal
    
    async with AsyncSessionLocal() as db:
        try:
            for target_lang in settings.TRANSLATION_TARGET_LANGUAGES:
                if target_lang != source_language.split("-")[0]:
                    translated_text = await translate_text(text, target_lang, source_language.split("-")[0])
                    
                    translation = Translation(
                        story_id=uuid.UUID(story_id),
                        translated_text=translated_text,
                        language=target_lang,
                        confidence_score=0.8  # Placeholder
                    )
                    
                    db.add(translation)
            
            await db.commit()
            
        except Exception as e:
            print(f"Translation failed for story {story_id}: {str(e)}")

async def process_analysis(story_id: str, text: str):
    """Background task to process sentiment and entity analysis."""
    from app.core.database import AsyncSessionLocal
    
    async with AsyncSessionLocal() as db:
        try:
            # Analyze sentiment
            sentiment_score = analyze_sentiment(text)
            
            # Extract entities
            entities = extract_entities(text)
            
            # Update analytics
            result = await db.execute(
                select(Analytics).where(Analytics.story_id == uuid.UUID(story_id))
            )
            analytics = result.scalar_one_or_none()
            
            if analytics:
                analytics.sentiment_score = sentiment_score
                analytics.entities = entities
                await db.commit()
            
        except Exception as e:
            print(f"Analysis failed for story {story_id}: {str(e)}")

@router.get("/transcribe/{story_id}/status")
async def get_transcription_status(
    story_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get transcription status for a story."""
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
            detail="Not authorized to view this story"
        )
    
    # Get transcript
    result = await db.execute(select(Transcript).where(Transcript.story_id == story_uuid))
    transcript = result.scalar_one_or_none()
    
    # Get translations
    result = await db.execute(select(Translation).where(Translation.story_id == story_uuid))
    translations = result.scalars().all()
    
    # Get analytics
    result = await db.execute(select(Analytics).where(Analytics.story_id == story_uuid))
    analytics = result.scalar_one_or_none()
    
    return {
        "story_id": story_id,
        "has_transcript": transcript is not None,
        "transcript_language": transcript.language if transcript else None,
        "translation_count": len(translations),
        "translation_languages": [t.language for t in translations],
        "has_analysis": analytics is not None and analytics.sentiment_score is not None,
        "status": story.status.value
    }

@router.get("/health")
async def ai_health_check():
    """Check AI services health."""
    services = {
        "openai_whisper": openai_client is not None,
        "google_translate": translate_client is not None,
        "google_speech": speech_client is not None,
        "spacy_ner": nlp is not None,
        "sentiment_analysis": sentiment_analyzer is not None
    }
    
    return {
        "status": "healthy",
        "services": services,
        "supported_languages": settings.SUPPORTED_LANGUAGES,
        "translation_targets": settings.TRANSLATION_TARGET_LANGUAGES
    } 