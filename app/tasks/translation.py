import uuid
from typing import Dict, Any
from celery import current_task

from app.core.celery import celery_app
from app.core.config import settings

@celery_app.task(bind=True, name="translate_story")
def translate_story_task(self, story_id: str, text: str) -> Dict[str, Any]:
    """Background task to translate story text."""
    try:
        story_uuid = uuid.UUID(story_id)
        
        current_task.update_state(
            state="PROGRESS",
            meta={"status": "Starting translation", "progress": 0}
        )
        
        translations = {}
        total_languages = len(settings.TRANSLATION_TARGET_LANGUAGES)
        
        for i, target_lang in enumerate(settings.TRANSLATION_TARGET_LANGUAGES):
            current_task.update_state(
                state="PROGRESS",
                meta={
                    "status": f"Translating to {target_lang}",
                    "progress": int((i / total_languages) * 100)
                }
            )
            
            # Simulate translation (in real implementation, call Google Translate)
            translated_text = f"Translated text in {target_lang}: {text[:50]}..."
            translations[target_lang] = translated_text
        
        current_task.update_state(
            state="PROGRESS",
            meta={"status": "Translation completed", "progress": 100}
        )
        
        return {
            "status": "success",
            "story_id": story_id,
            "translations": translations
        }
        
    except Exception as e:
        current_task.update_state(
            state="FAILURE",
            meta={"error": str(e)}
        )
        raise

@celery_app.task(bind=True, name="translate_single")
def translate_single_task(self, story_id: str, text: str, target_language: str, source_language: str = "auto") -> Dict[str, Any]:
    """Background task to translate text to a single language."""
    try:
        current_task.update_state(
            state="PROGRESS",
            meta={"status": f"Translating to {target_language}", "progress": 0}
        )
        
        # Simulate translation (in real implementation, call Google Translate)
        translated_text = f"Translated text in {target_language}: {text[:50]}..."
        
        current_task.update_state(
            state="PROGRESS",
            meta={"status": "Translation completed", "progress": 100}
        )
        
        return {
            "status": "success",
            "story_id": story_id,
            "target_language": target_language,
            "translated_text": translated_text
        }
        
    except Exception as e:
        current_task.update_state(
            state="FAILURE",
            meta={"error": str(e)}
        )
        raise 