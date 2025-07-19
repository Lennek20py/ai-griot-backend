#!/usr/bin/env python3
"""
Script to fix malformed transcript data in the database.
This script will clean up the words array in transcript_json and translation_json fields.
"""

import asyncio
import json
import re
from typing import List, Dict, Any
from sqlalchemy import text
from app.core.database import get_db
from app.models.story import Transcript, Translation
from sqlalchemy.ext.asyncio import AsyncSession

def clean_words_array(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Clean and validate words array to ensure proper structure.
    
    Args:
        words: List of word dictionaries that may be malformed
        
    Returns:
        Cleaned list of word dictionaries with proper structure
    """
    cleaned_words = []
    
    for word_data in words:
        # Skip if not a dictionary
        if not isinstance(word_data, dict):
            continue
            
        # Extract word text - handle various possible formats
        word_text = None
        if "word" in word_data:
            word_text = word_data["word"]
        elif "text" in word_data:
            word_text = word_data["text"]
        
        # Skip if no valid word text found
        if not word_text or not isinstance(word_text, str):
            continue
            
        # Clean the word text
        word_text = word_text.strip()
        if not word_text:
            continue
            
        # Extract and validate timestamps
        start_time = None
        end_time = None
        
        # Try to get start_time
        if "start_time" in word_data:
            try:
                start_time = float(word_data["start_time"])
            except (ValueError, TypeError):
                pass
        elif "start" in word_data:
            try:
                start_time = float(word_data["start"])
            except (ValueError, TypeError):
                pass
                
        # Try to get end_time
        if "end_time" in word_data:
            try:
                end_time = float(word_data["end_time"])
            except (ValueError, TypeError):
                pass
        elif "end" in word_data:
            try:
                end_time = float(word_data["end"])
            except (ValueError, TypeError):
                pass
        
        # If we have valid timestamps, add to cleaned words
        if start_time is not None and end_time is not None and start_time < end_time:
            cleaned_words.append({
                "word": word_text,
                "start_time": start_time,
                "end_time": end_time
            })
    
    return cleaned_words

def regenerate_words_from_text(text: str, total_duration: float = 300.0) -> List[Dict[str, Any]]:
    """
    Regenerate word-level timestamps from text when original timestamps are corrupted.
    
    Args:
        text: The transcript text
        total_duration: Estimated total duration in seconds
        
    Returns:
        List of word dictionaries with estimated timestamps
    """
    if not text:
        return []
    
    # Split text into words
    words = re.findall(r'\b\w+\b', text)
    
    if not words:
        return []
    
    # Estimate timing - distribute evenly across total duration
    word_duration = total_duration / len(words)
    
    words_array = []
    for i, word in enumerate(words):
        start_time = i * word_duration
        end_time = (i + 1) * word_duration
        words_array.append({
            "word": word,
            "start_time": start_time,
            "end_time": end_time
        })
    
    return words_array

async def fix_transcript_data():
    """Fix malformed transcript data in the database."""
    async for db in get_db():
        try:
            print("🔧 Starting transcript data cleanup...")
            
            # Get all transcripts
            result = await db.execute(text("SELECT id, transcript_json FROM transcripts"))
            transcripts = result.fetchall()
            
            fixed_count = 0
            for transcript_id, transcript_json in transcripts:
                try:
                    if not transcript_json:
                        continue
                    
                    # Clean words array
                    if "words" in transcript_json:
                        original_words = transcript_json["words"]
                        cleaned_words = clean_words_array(original_words)
                        
                        # If cleaning resulted in significant loss, regenerate from text
                        if len(cleaned_words) < len(original_words) * 0.5:
                            print(f"⚠️  Significant data loss for transcript {transcript_id}, regenerating...")
                            if "original_text" in transcript_json:
                                cleaned_words = regenerate_words_from_text(transcript_json["original_text"])
                            elif "text" in transcript_json:
                                cleaned_words = regenerate_words_from_text(transcript_json["text"])
                        
                        # Update transcript_json
                        transcript_json["words"] = cleaned_words
                        
                        # Update database
                        await db.execute(
                            text("UPDATE transcripts SET transcript_json = :transcript_json WHERE id = :id"),
                            {"transcript_json": json.dumps(transcript_json), "id": transcript_id}
                        )
                        
                        fixed_count += 1
                        print(f"✅ Fixed transcript {transcript_id}: {len(original_words)} -> {len(cleaned_words)} words")
                
                except Exception as e:
                    print(f"❌ Error fixing transcript {transcript_id}: {e}")
                    continue
            
            print(f"✅ Fixed {fixed_count} transcripts")
            
            # Fix translation data
            print("🔧 Starting translation data cleanup...")
            
            result = await db.execute(text("SELECT id, translation_json FROM translations WHERE translation_json IS NOT NULL"))
            translations = result.fetchall()
            
            fixed_translations = 0
            for translation_id, translation_json in translations:
                try:
                    if not translation_json:
                        continue
                    
                    # Clean words array in translation
                    if "words" in translation_json:
                        original_words = translation_json["words"]
                        cleaned_words = clean_words_array(original_words)
                        
                        # If cleaning resulted in significant loss, regenerate from text
                        if len(cleaned_words) < len(original_words) * 0.5:
                            print(f"⚠️  Significant data loss for translation {translation_id}, regenerating...")
                            if "text" in translation_json:
                                cleaned_words = regenerate_words_from_text(translation_json["text"])
                        
                        # Update translation_json
                        translation_json["words"] = cleaned_words
                        
                        # Update database
                        await db.execute(
                            text("UPDATE translations SET translation_json = :translation_json WHERE id = :id"),
                            {"translation_json": json.dumps(translation_json), "id": translation_id}
                        )
                        
                        fixed_translations += 1
                        print(f"✅ Fixed translation {translation_id}: {len(original_words)} -> {len(cleaned_words)} words")
                
                except Exception as e:
                    print(f"❌ Error fixing translation {translation_id}: {e}")
                    continue
            
            print(f"✅ Fixed {fixed_translations} translations")
            
            # Commit changes
            await db.commit()
            print("✅ All changes committed to database")
            
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
            await db.rollback()
        finally:
            await db.close()

if __name__ == "__main__":
    asyncio.run(fix_transcript_data()) 