#!/usr/bin/env python3
import asyncio
import asyncpg
from app.core.config import settings

async def migrate_translations():
    """Add translation_json column to translations table"""
    try:
        # Fix database URL for asyncpg
        db_url = settings.DATABASE_URL.replace('+asyncpg', '')
        
        # Connect to database
        conn = await asyncpg.connect(db_url)
        
        # Check if column exists
        result = await conn.fetch("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'translations' 
            AND column_name = 'translation_json'
        """)
        
        if not result:
            print("Adding translation_json column to translations table...")
            await conn.execute("""
                ALTER TABLE translations 
                ADD COLUMN translation_json JSONB
            """)
            print("✅ Successfully added translation_json column")
        else:
            print("✅ translation_json column already exists")
        
        await conn.close()
        
    except Exception as e:
        print(f"❌ Error during migration: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(migrate_translations()) 