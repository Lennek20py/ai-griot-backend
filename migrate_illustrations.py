#!/usr/bin/env python3
"""
Migration script to add paragraphs and illustrations tables for AI-generated visual storytelling.
"""

import asyncio
import sys
import os
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.config import settings
from app.models.story import Paragraph, Illustration

async def create_tables():
    """Create the new paragraphs and illustrations tables"""
    
    # Get database URL from settings
    database_url = settings.DATABASE_URL
    
    # Create async engine
    engine = create_async_engine(database_url)
    
    async with engine.begin() as conn:
        # Create paragraphs table
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS paragraphs (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                story_id UUID NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
                paragraph_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                start_time FLOAT,
                end_time FLOAT,
                word_count INTEGER,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """))
        
        # Create illustrations table
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS illustrations (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                paragraph_id UUID NOT NULL REFERENCES paragraphs(id) ON DELETE CASCADE,
                image_url TEXT NOT NULL,
                prompt_used TEXT,
                style_description TEXT,
                generation_metadata JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """))
        
        # Create indexes for better performance
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_paragraphs_story_id ON paragraphs(story_id)
        """))
        
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_paragraphs_index ON paragraphs(story_id, paragraph_index)
        """))
        
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_illustrations_paragraph_id ON illustrations(paragraph_id)
        """))
        
        print("✅ Tables created successfully!")
    
    await engine.dispose()

async def add_paragraphs_relationship():
    """Add paragraphs relationship to stories table if not exists"""
    
    database_url = settings.DATABASE_URL
    engine = create_async_engine(database_url)
    
    async with engine.begin() as conn:
        # Check if the relationship already exists
        result = await conn.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'stories' AND column_name = 'paragraphs'
        """))
        
        if not result.fetchone():
            print("ℹ️  No direct paragraphs column needed - using foreign key relationship")
        else:
            print("ℹ️  Paragraphs relationship already exists")
    
    await engine.dispose()

async def main():
    """Main migration function"""
    print("🚀 Starting illustration tables migration...")
    
    try:
        await create_tables()
        await add_paragraphs_relationship()
        print("🎉 Migration completed successfully!")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 