#!/usr/bin/env python3
"""
Migration script to update audio file URLs in existing stories.
This script converts old URL formats to the new static file serving format.
"""

import asyncio
import os
import sys
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.database import AsyncSessionLocal
# Import all models to avoid circular dependency issues
from app.models.user import User
from app.models.story import Story

async def migrate_audio_urls():
    """Migrate existing audio URLs to the new format."""
    async with AsyncSessionLocal() as db:
        try:
            # Get all stories with audio files
            result = await db.execute(
                select(Story).where(Story.audio_file_url.is_not(None))
            )
            stories = result.scalars().all()
            
            updated_count = 0
            
            for story in stories:
                old_url = story.audio_file_url
                new_url = None
                
                # Convert different old formats to new format
                if old_url.startswith('media/files/'):
                    # Convert "media/files/uploads/file.wav" to "/media/files/uploads/file.wav"
                    new_url = f"/{old_url}"
                elif old_url.startswith('http://localhost:8000/media/files/'):
                    # Convert "http://localhost:8000/media/files/uploads/file.wav" to "/media/files/uploads/file.wav"
                    new_url = old_url.replace('http://localhost:8000', '')
                elif old_url.startswith('https://localhost:8000/media/files/'):
                    # Convert "https://localhost:8000/media/files/uploads/file.wav" to "/media/files/uploads/file.wav"
                    new_url = old_url.replace('https://localhost:8000', '')
                elif old_url.startswith('/api/v1/media/files/'):
                    # Convert "/api/v1/media/files/uploads/file.wav" to "/media/files/uploads/file.wav"
                    file_path = old_url.replace('/api/v1/media/files/', '')
                    new_url = f"/media/files/{file_path}"
                
                # Update if we have a new URL and it's different
                if new_url and new_url != old_url:
                    story.audio_file_url = new_url
                    updated_count += 1
                    print(f"Updated story {story.id}: {old_url} -> {new_url}")
            
            # Commit all changes
            await db.commit()
            print(f"\nMigration completed successfully!")
            print(f"Updated {updated_count} stories")
            
        except Exception as e:
            await db.rollback()
            print(f"Migration failed: {e}")
            raise

if __name__ == "__main__":
    print("Starting audio URL migration...")
    asyncio.run(migrate_audio_urls()) 