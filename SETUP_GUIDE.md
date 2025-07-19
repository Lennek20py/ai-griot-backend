# 🎙️ Digital Griot MVP - Quick Setup Guide

**AI-Powered Oral Tradition Preservation Platform with Gemini TTS**

## ✨ What's Been Built

### 🔧 Backend MVP Features
- **Google Cloud Speech-to-Text**: High-quality audio transcription with timestamps
- **Google Gemini AI**: Content enhancement and cultural analysis
- **🆕 Google Gemini TTS**: Text-to-speech for story narration and accessibility
- **Google Translate**: Multi-language translation support
- **FastAPI Framework**: Modern, async Python backend
- **PostgreSQL Database**: Robust data storage with JSON support
- **JWT Authentication**: Secure user management
- **Background Processing**: Async AI pipeline for story processing
- **🇹🇿 Swahili Language Focus**: Enhanced support for East African storytelling

### 📱 Frontend Compatibility
The backend is designed to work with your existing React frontend:
- Compatible with the upload flow in `UploadPage.tsx`
- RESTful API endpoints for all frontend features
- CORS configured for local development
- New TTS endpoints for audio generation

## 🚀 30-Second Setup

```bash
# 1. Navigate to backend
cd ai-griot-backend

# 2. Run the automated setup
python setup_mvp.py

# 3. Start the server
./start_server.sh

# 4. Test Swahili TTS (optional)
python test_swahili_tts.py
```

## 🔑 Required API Keys

### Google Cloud (Speech + Translate)
1. Create project at [Google Cloud Console](https://console.cloud.google.com/)
2. Enable **Speech-to-Text API** and **Cloud Translate API**
3. Create service account and download JSON key file

### Google Gemini (AI Analysis + TTS)
1. Get API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. **Single key enables both content analysis AND text-to-speech!**

## 🎯 MVP Workflow

### Story Processing Pipeline
1. **Upload Audio** → Store in cloud/local storage
2. **Speech-to-Text** → Google Cloud transcription with timestamps
3. **AI Enhancement** → Gemini cleans and improves transcript
4. **Cultural Analysis** → Gemini extracts themes, entities, sentiment
5. **Translation** → Google Translate to multiple languages
6. **🆕 Audio Narration** → Gemini TTS generates story narration
7. **Storage** → Save all results to database

### API Endpoints Ready for Frontend

```http
# Authentication
POST /api/v1/auth/register
POST /api/v1/auth/login

# Story Management
POST /api/v1/stories          # Create story
GET  /api/v1/stories          # List stories
GET  /api/v1/stories/{id}     # Get story details

# Media Upload
POST /api/v1/media/upload     # Upload audio files

# AI Processing
POST /api/v1/ai/transcribe                    # Direct transcription
POST /api/v1/ai/process-story                 # Full AI pipeline
GET  /api/v1/ai/story/{story_id}/analysis     # Get results

# 🆕 Text-to-Speech Endpoints
POST /api/v1/ai/generate-speech               # Generate TTS from text
POST /api/v1/ai/generate-story-narration      # Generate story narration
GET  /api/v1/ai/voices                        # Available TTS voices
```

## 📊 What the AI Does

### Speech-to-Text (Google Cloud)
- Converts audio to text with word-level timestamps
- Supports multiple languages and audio formats
- Automatic punctuation and formatting

### Content Enhancement (Gemini)
- Cleans transcription errors while preserving voice
- Identifies cultural themes and story types
- Extracts people, places, objects mentioned
- Analyzes sentiment and emotional tone
- **🇹🇿 Enhanced for Swahili**: Recognizes traditional elements, Arabic influences

### 🆕 Text-to-Speech (Gemini)
- **Natural Voice Generation**: Creates high-quality audio from text
- **Multi-Speaker Support**: Dialog with different voices
- **Swahili Optimized**: Voices tuned for East African languages
- **Traditional Elements**: Preserves "Hadithi, hadithi?" storytelling markers
- **Multiple Voices**: Zephyr, Puck, Charon, Kore (Zephyr & Kore recommended for Swahili)

### Translation (Google Translate)
- Translates to 5+ languages (Swahili first priority)
- Preserves cultural context and meaning
- Professional-grade quality

## 🧪 Testing Your Setup

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Test Swahili TTS
```bash
# Run the included test script
python test_swahili_tts.py

# Or test manually with curl
curl -X POST "http://localhost:8000/api/v1/ai/generate-speech" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "text=Hadithi, hadithi? Hadithi njoo! Palikuwa na mfalme mkuu..." \
  -F "language=sw" \
  -F "voice_name=Zephyr" \
  --output swahili_test.wav
```

### 3. Upload Test Audio
```python
import requests

# Upload and transcribe (default Swahili)
with open("test_story.mp3", "rb") as f:
    files = {"file": f}
    response = requests.post(
        "http://localhost:8000/api/v1/ai/transcribe",
        files=files,
        params={"language": "sw-KE", "enhance": True}
    )
    
print(response.json())
```

### 4. Frontend Integration
Update your frontend's API base URL to:
```javascript
const API_BASE_URL = "http://localhost:8000/api/v1"

// New TTS functionality
const generateAudio = async (text, language = "sw") => {
  const formData = new FormData();
  formData.append('text', text);
  formData.append('language', language);
  formData.append('voice_name', 'Zephyr');
  
  const response = await fetch(`${API_BASE_URL}/ai/generate-speech`, {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${token}` },
    body: formData
  });
  
  return await response.blob(); // Audio file
};
```

## 🔧 Quick Troubleshooting

### Common Issues
1. **"Gemini TTS client not configured"**
   - Verify your GEMINI_API_KEY in .env file
   - Check that google-genai package is installed
   - Test at https://aistudio.google.com/

2. **"Google Cloud clients not initialized"**
   - Check your service account JSON file path
   - Verify Google Cloud APIs are enabled

3. **Audio generation fails**
   - Check text length (very long texts may fail)
   - Try different voices (Zephyr, Kore work well for Swahili)
   - Verify GEMINI_API_KEY has TTS permissions

4. **Swahili transcription issues**
   - Use language codes: "sw-KE" (Kenya) or "sw-TZ" (Tanzania)
   - Enable enhancement for better cultural context

### Development Mode
For quick testing without external services:
```bash
# Use SQLite instead of PostgreSQL
DATABASE_URL=sqlite+aiosqlite:///./digital_griot.db

# Skip Redis for synchronous processing
REDIS_URL=""

# Test TTS without authentication (development only)
curl -X POST "http://localhost:8000/api/v1/ai/generate-speech" \
  -F "text=Habari yako?" \
  -F "language=sw" \
  --output test.wav
```

## 🎭 What's Next

1. **Connect Your Frontend**: Update upload and playback components
2. **Add TTS Integration**: Generate narrations for uploaded stories
3. **Test Swahili Stories**: Upload authentic Swahili oral traditions
4. **Multi-language Workflow**: Test translation + TTS pipeline
5. **Voice Customization**: Experiment with different TTS voices
6. **Cultural Enhancement**: Leverage enhanced Swahili cultural analysis

## 🎵 Swahili TTS Features

### Traditional Storytelling Support
- **"Hadithi, hadithi?" Integration**: Automatic traditional openings
- **Cultural Preservation**: Maintains authentic storytelling elements
- **Arabic Influence Recognition**: Preserves Islamic cultural elements

### Voice Options
- **Zephyr**: Warm, clear voice - excellent for Swahili
- **Kore**: Rich, expressive - good for dramatic stories
- **Multi-speaker**: Different voices for dialog

### Use Cases
- **Story Narration**: Convert written stories to audio
- **Translation Audio**: Hear stories in different languages
- **Accessibility**: Audio versions for visually impaired users
- **Language Learning**: Hear proper Swahili pronunciation

## 📚 Full Documentation

- **Complete README**: [README.md](README.md)
- **API Docs**: http://localhost:8000/docs (when running)
- **Code Structure**: See `/app` directory for implementation details
- **Swahili Examples**: `test_swahili_tts.py` for working examples

## 🌟 New Capabilities Summary

**Before**: Upload audio → Transcribe → Analyze → Translate → Store
**Now**: Upload audio → Transcribe → Analyze → Translate → **Generate Audio Narration** → Store

---

**🎉 You now have a complete AI-powered oral tradition preservation platform with TTS!**

The MVP includes everything needed to:
- Upload and store audio stories
- Transcribe with Google's best-in-class speech recognition
- Enhance and analyze content with Gemini AI
- Translate to multiple languages
- **🆕 Generate audio narrations with Gemini TTS**
- **🇹🇿 Full Swahili language and cultural support**
- Provide a clean API for your frontend

*Asante sana! Ready to preserve cultural heritage with AI! 🌍✨* 