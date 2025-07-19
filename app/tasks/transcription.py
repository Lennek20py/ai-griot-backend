import uuid
from typing import Dict, Any
from celery import current_task
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.celery import celery_app
from app.core.database import AsyncSessionLocal
from app.models.story import Story, Transcript, StoryStatus
from app.api.v1.endpoints.ai_processing import (
    transcribe_audio_with_whisper,
    transcribe_audio_with_google
)

@celery_app.task(bind=True, name="transcribe_audio")
def transcribe_audio_task(self, story_id: str, audio_url: str, language: str) -> Dict[str, Any]:
    """Background task to transcribe audio."""
    try:
        # Update task state
        current_task.update_state(
            state="PROGRESS",
            meta={"status": "Starting transcription", "progress": 0}
        )
        
        # Try OpenAI Whisper first
        try:
            current_task.update_state(
                state="PROGRESS",
                meta={"status": "Transcribing with OpenAI Whisper", "progress": 25}
            )
            
            # Note: This would need to be adapted for Celery (async to sync)
            # For now, we'll use a simplified approach
            transcript_data = {
                "text": "Sample transcription text",
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "Sample"}
                ],
                "language": language,
                "duration": 60.0
            }
            
        except Exception as e:
            # Fallback to Google Speech-to-Text
            current_task.update_state(
                state="PROGRESS",
                meta={"status": "Falling back to Google Speech-to-Text", "progress": 50}
            )
            
            transcript_data = {
                "text": "Sample transcription text (Google)",
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "Sample"}
                ],
                "language": language
            }
        
        # Save transcript to database
        current_task.update_state(
            state="PROGRESS",
            meta={"status": "Saving transcript", "progress": 75}
        )
        
        # This would need to be adapted for async database operations in Celery
        # For now, we'll return the transcript data
        
        current_task.update_state(
            state="PROGRESS",
            meta={"status": "Transcription completed", "progress": 100}
        )
        
        return {
            "status": "success",
            "transcript_data": transcript_data,
            "story_id": story_id
        }
        
    except Exception as e:
        current_task.update_state(
            state="FAILURE",
            meta={"error": str(e)}
        )
        raise

@celery_app.task(bind=True, name="process_story_transcription")
def process_story_transcription_task(self, story_id: str) -> Dict[str, Any]:
    """Process transcription for a story and trigger subsequent tasks."""
    try:
        # Get story information
        story_uuid = uuid.UUID(story_id)
        
        # Update task state
        current_task.update_state(
            state="PROGRESS",
            meta={"status": "Processing story transcription", "progress": 0}
        )
        
        # Start transcription
        transcription_result = transcribe_audio_task.delay(
            story_id=story_id,
            audio_url="audio_url_placeholder",  # Would get from story
            language="en"
        )
        
        # Wait for transcription to complete
        transcript_data = transcription_result.get()
        
        current_task.update_state(
            state="PROGRESS",
            meta={"status": "Transcription completed, starting translation", "progress": 50}
        )
        
        # Trigger translation tasks
        from app.tasks.translation import translate_story_task
        translation_task = translate_story_task.delay(story_id, transcript_data["transcript_data"]["text"])
        
        # Trigger analysis tasks
        from app.tasks.analysis import analyze_story_task
        analysis_task = analyze_story_task.delay(story_id, transcript_data["transcript_data"]["text"])
        
        current_task.update_state(
            state="PROGRESS",
            meta={"status": "All tasks initiated", "progress": 100}
        )
        
        return {
            "status": "success",
            "story_id": story_id,
            "transcription_task_id": transcription_result.id,
            "translation_task_id": translation_task.id,
            "analysis_task_id": analysis_task.id
        }
        
    except Exception as e:
        current_task.update_state(
            state="FAILURE",
            meta={"error": str(e)}
        )
        raise 