#!/bin/bash
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
