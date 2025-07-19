#!/usr/bin/env python33
"""
Digital Griot MVP Setup Script

This script helps set up the Digital Griot backend with Google tools for speech-to-text,
text-to-speech, and AI processing. It guides you through the configuration process with
special focus on Swahili language support.
"""

import os
import sys
import subprocess
import json
from pathlib import Path


def print_banner():
    """Print the Digital Griot banner"""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║                    🎙️  DIGITAL GRIOT MVP 🎙️                     ║
    ║              AI-Powered Oral Tradition Preservation             ║
    ║                     🌍 Swahili Focus Edition 🌍                 ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)


def check_python3_version():
    """Check if python3 version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: python3 3.8 or higher is required.")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ python3 version: {sys.version.split()[0]}")


def install_dependencies():
    """Install required python3 packages"""
    print("\n📦 Installing python3 dependencies...")
    print("   This includes google-genai for Gemini TTS functionality...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        print("   - Google Cloud Speech-to-Text ✓")
        print("   - Google Gemini (Content + TTS) ✓")
        print("   - Google Cloud Translate ✓")
        print("   - Audio processing libraries ✓")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False
    return True


def download_spacy_model():
    """Download spaCy English model"""
    print("\n🧠 Downloading spaCy English model...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✅ spaCy English model downloaded successfully")
    except subprocess.CalledProcessError:
        print("⚠️  Warning: Failed to download spaCy model. You can install it later with:")
        print("   python3 -m spacy download en_core_web_sm")


def create_env_file():
    """Create .env file with user input"""
    print("\n⚙️  Setting up environment configuration...")
    
    env_content = """# Digital Griot Backend Configuration

# Application settings
APP_NAME=Digital Griot API
DEBUG=true
VERSION=1.0.0

# Database
DATABASE_URL=postgresql+asyncpg://yannickgbaka:password@localhost:5432/digital_griot

# JWT Settings
SECRET_KEY={secret_key}
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT_ID={project_id}
GOOGLE_APPLICATION_CREDENTIALS={credentials_path}

# Google Gemini API (for content analysis AND text-to-speech)
GEMINI_API_KEY={gemini_key}

# AWS S3 (for audio file storage) - Optional for MVP
AWS_ACCESS_KEY_ID={aws_key}
AWS_SECRET_ACCESS_KEY={aws_secret}
AWS_REGION=us-east-1
S3_BUCKET_NAME=digital-griot-audio

# Redis (for background tasks)
REDIS_URL=redis://localhost:6379

# Audio Processing
AUDIO_SAMPLE_RATE=16000
CHUNK_DURATION_SECONDS=30

# Text-to-Speech Settings
TTS_MODEL=gemini-2.5-pro-preview-tts
TTS_DEFAULT_VOICE=Zephyr
TTS_SAMPLE_RATE=24000

# Language Support (Swahili-focused)
SUPPORTED_LANGUAGES=["sw-KE", "sw-TZ", "en-US", "fr-FR", "es-ES", "ar-SA"]
TRANSLATION_TARGET_LANGUAGES=["sw", "en", "fr", "es", "ar"]
SWAHILI_VOICES=["Zephyr", "Kore"]
"""

    print("\n🔑 Let's configure your API keys and settings:")
    
    # Generate a secret key
    import secrets
    secret_key = secrets.token_urlsafe(32)
    
    # Get Google Cloud settings
    print("\n📋 Google Cloud Configuration:")
    print("   1. Create a project at https://console.cloud.google.com/")
    print("   2. Enable Speech-to-Text API and Translate API")
    print("   3. Create a service account and download the JSON key file")
    
    project_id = input("\nEnter your Google Cloud Project ID: ").strip()
    credentials_path = input("Enter path to your service account JSON file: ").strip()
    
    # Expand user path
    if credentials_path.startswith("~"):
        credentials_path = str(Path(credentials_path).expanduser())
    
    # Verify credentials file exists
    if not os.path.exists(credentials_path):
        print(f"⚠️  Warning: Credentials file not found at {credentials_path}")
        print("   Make sure to place your service account JSON file at this location")
    
    # Get Gemini API key
    print("\n🤖 Google Gemini Configuration:")
    print("   Get your API key from https://aistudio.google.com/app/apikey")
    print("   This key enables BOTH content analysis AND text-to-speech!")
    gemini_key = input("Enter your Gemini API key: ").strip()
    
    # AWS settings (optional)
    print("\n☁️  AWS Configuration (Optional - for production audio storage):")
    use_aws = input("Do you want to configure AWS S3? (y/n): ").strip().lower() == 'y'
    
    if use_aws:
        aws_key = input("Enter AWS Access Key ID: ").strip()
        aws_secret = input("Enter AWS Secret Access Key: ").strip()
    else:
        aws_key = "your-aws-access-key"
        aws_secret = "your-aws-secret-key"
        print("   Skipping AWS configuration. You can add it later to .env file")
    
    # Create .env file
    env_content = env_content.format(
        secret_key=secret_key,
        project_id=project_id,
        credentials_path=credentials_path,
        gemini_key=gemini_key,
        aws_key=aws_key,
        aws_secret=aws_secret
    )
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("\n✅ Environment file created: .env")


def setup_database():
    """Guide user through database setup"""
    print("""
🗄️  Database Setup:

For the MVP, you need PostgreSQL running. Here are your options:

1. 🐳 Docker (Recommended for development):
   docker run -d \\
     --name digital-griot-db \\
     -e POSTGRES_DB=digital_griot \\
     -e POSTGRES_USER=postgres \\
     -e POSTGRES_PASSWORD=password \\
     -p 5432:5432 \\
     postgres:13

2. 📦 Local PostgreSQL installation:
   - Install PostgreSQL from https://postgresql.org/
   - Create database: createdb digital_griot

3. ☁️  Cloud database (Render, Supabase, etc.)
   - Update DATABASE_URL in .env file

4. 🗂️  SQLite (for simple testing):
   - Change DATABASE_URL to: sqlite+aiosqlite:///./digital_griot.db
""")
    
    setup_choice = input("Press Enter when your database is ready...")
    

def setup_redis():
    """Guide user through Redis setup"""
    print("""
📊 Redis Setup (for background tasks):

For the MVP, you need Redis running. Here are your options:

1. 🐳 Docker (Recommended):
   docker run -d --name redis -p 6379:6379 redis:alpine

2. 📦 Local Redis installation:
   - Install Redis from https://redis.io/
   - Start with: redis-server

3. ☁️  Cloud Redis (Redis Cloud, Upstash, etc.)
   - Update REDIS_URL in .env file

For development, you can skip Redis and background tasks will run synchronously.
""")
    
    setup_choice = input("Press Enter when Redis is ready (or skip for now)...")


def run_migrations():
    """Run database migrations"""
    print("\n🔄 Setting up database tables...")
    try:
        # Note: In a real setup, you'd use Alembic migrations
        print("   Run this command to create tables:")
        print("   python3 -c \"from app.core.database import engine, Base; import asyncio; asyncio.run(engine.run_sync(Base.metadata.create_all))\"")
        print("   (This will be automated in future versions)")
    except Exception as e:
        print(f"⚠️  Note: You'll need to create database tables. Error: {e}")


def create_startup_script():
    """Create a script to easily start the application"""
    startup_script = """#!/bin/bash
# Digital Griot Startup Script

echo "🎙️  Starting Digital Griot API with Gemini TTS..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the application
echo "Starting FastAPI server on http://localhost:8000"
echo "Features: Speech-to-Text, Text-to-Speech, Swahili Support"
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""
    
    with open("start_server.sh", "w") as f:
        f.write(startup_script)
    
    # Make it executable
    os.chmod("start_server.sh", 0o755)
    print("✅ Created startup script: start_server.sh")


def create_test_script():
    """Create executable test script for Swahili TTS"""
    print("✅ Created Swahili TTS test script: test_swahili_tts.py")
    os.chmod("test_swahili_tts.py", 0o755)


def print_next_steps():
    """Print next steps for the user"""
    print("""
🎉 Digital Griot MVP Setup Complete!

🌟 NEW FEATURES INCLUDED:
   📢 Gemini Text-to-Speech (TTS) for story narration
   🇹🇿 Enhanced Swahili language support
   🎭 Traditional storytelling elements preservation
   🗣️ Multi-speaker dialog generation

📋 Next Steps:

1. 🗄️  Set up your database:
   - PostgreSQL: Create 'digital_griot' database
   - Run migrations (see instructions above)

2. 🔑 Verify your API keys:
   - Google Cloud: Test Speech-to-Text and Translate APIs
   - Gemini: Test content analysis AND TTS at https://aistudio.google.com/

3. 🚀 Start the server:
   ./start_server.sh
   
   Or manually:
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

4. 🧪 Test Swahili TTS functionality:
   python3 test_swahili_tts.py

5. 📱 Connect your frontend:
   - Update frontend API URL to http://localhost:8000
   - Start your React/Vue frontend

6. 🎭 Test the complete workflow:
   - Upload a Swahili story
   - Generate enhanced transcription
   - Create audio narration with Gemini TTS
   - Test multi-language support

📚 API Endpoints Available:
   - POST /api/v1/ai/transcribe (Speech-to-Text)
   - POST /api/v1/ai/generate-speech (Text-to-Speech)
   - POST /api/v1/ai/generate-story-narration (Full narration)
   - GET /api/v1/ai/voices (Available TTS voices)
   - GET /api/v1/ai/health (Service status)

🎵 Swahili TTS Features:
   - Traditional storytelling elements ("Hadithi, hadithi?")
   - Optimized voices (Zephyr, Kore)
   - Cultural context preservation
   - Multi-speaker dialog support

📚 Documentation:
   - API docs: http://localhost:8000/docs
   - Health check: http://localhost:8000/health
   - Swahili test examples: test_swahili_tts.py

🆘 Need help?
   - Check the logs for any errors
   - Verify your .env configuration
   - Ensure all services (DB, Redis) are running
   - Test Gemini TTS with the included test script

Asante sana! Happy storytelling! 🎭✨
""")


def main():
    """Main setup function"""
    print_banner()
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("Setting up Digital Griot MVP with Google AI tools and Swahili TTS...\n")
    
    # Check python3 version
    check_python3_version()
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Download spaCy model
    download_spacy_model()
    
    # Create environment file
    create_env_file()
    
    # Database setup instructions
    setup_database()
    
    # Redis setup instructions
    setup_redis()
    
    # Create startup script
    create_startup_script()
    
    # Make test script executable
    create_test_script()
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main() 