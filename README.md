# 🎙️ Digital Griot - Backend MVP

AI-powered platform for preserving and sharing oral traditions using Google's advanced AI tools, featuring **Gemini Text-to-Speech** and **enhanced Swahili support**.

## 🌟 Features

### Core Functionality
- **🎵 Audio Upload & Processing**: Support for multiple audio formats (MP3, WAV, M4A, FLAC, OGG, WebM)
- **🗣️ Speech-to-Text**: Powered by Google Cloud Speech-to-Text API with timestamp support
- **🤖 AI Enhancement**: Text enhancement and cultural analysis using Google Gemini
- **🆕 Text-to-Speech**: Generate natural audio narrations using Google Gemini TTS
- **🌍 Multi-language Translation**: Automated translation using Google Cloud Translate
- **📊 Content Analysis**: Cultural context, sentiment analysis, and entity extraction
- **👥 User Management**: Secure authentication and contributor dashboards
- **🔍 Advanced Search**: Full-text search across stories and metadata
- **🇹🇿 Swahili Focus**: Enhanced support for East African storytelling traditions

### MVP Architecture
- **FastAPI Backend**: High-performance async Python API
- **PostgreSQL Database**: Robust data storage with JSON support
- **Google AI Integration**: Speech-to-Text, Gemini (Content + TTS), and Translate APIs
- **Background Processing**: Async story processing pipeline
- **RESTful API**: Clean, documented endpoints for frontend integration

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL database
- Google Cloud Project with enabled APIs
- Google Gemini API key (enables both content analysis and TTS)

### 1. Automated Setup (Recommended)

```bash
# Clone and navigate to backend
cd ai-griot-backend

# Run the setup script (now includes TTS configuration)
python setup_mvp.py

# Start the server
./start_server.sh

# Test Swahili TTS functionality
python test_swahili_tts.py
```

The setup script will guide you through:
- Installing dependencies (including google-genai for TTS)
- Configuring API keys for both content and TTS
- Setting up environment variables with Swahili defaults
- Database and Redis setup instructions

### 2. Manual Setup

#### Install Dependencies
```bash
pip install -r requirements.txt  # Now includes google-genai
python -m spacy download en_core_web_sm
```

#### Configure Environment
Create a `.env` file:

```bash
# Application settings
APP_NAME=Digital Griot API
DEBUG=true
VERSION=1.0.0

# Database
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/digital_griot

# JWT Settings
SECRET_KEY=your-super-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT_ID=your-google-cloud-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

# Google Gemini API (Content Analysis + Text-to-Speech)
GEMINI_API_KEY=your-gemini-api-key

# Text-to-Speech Settings
TTS_MODEL=gemini-2.5-pro-preview-tts
TTS_DEFAULT_VOICE=Zephyr
TTS_SAMPLE_RATE=24000

# Swahili Language Focus
SUPPORTED_LANGUAGES=["sw-KE", "sw-TZ", "en-US", "fr-FR", "es-ES", "ar-SA"]
TRANSLATION_TARGET_LANGUAGES=["sw", "en", "fr", "es", "ar"]
SWAHILI_VOICES=["Zephyr", "Kore"]

# Optional: AWS S3 for production storage
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
S3_BUCKET_NAME=digital-griot-audio

# Redis for background tasks
REDIS_URL=redis://localhost:6379
```

#### Setup Services

1. **Database (PostgreSQL)**:
   ```bash
   # Using Docker (recommended)
   docker run -d \
     --name digital-griot-db \
     -e POSTGRES_DB=digital_griot \
     -e POSTGRES_USER=postgres \
     -e POSTGRES_PASSWORD=password \
     -p 5432:5432 \
     postgres:13
   ```

2. **Redis (Background Tasks)**:
   ```bash
   # Using Docker
   docker run -d --name redis -p 6379:6379 redis:alpine
   ```

3. **Google Cloud Setup**:
   - Create project at [Google Cloud Console](https://console.cloud.google.com/)
   - Enable Speech-to-Text API and Cloud Translate API
   - Create service account and download JSON key
   - Get Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

#### Start the Server
```bash
# Using the generated startup script
./start_server.sh

# Or manually
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 📚 API Documentation

### Base URL
```
http://localhost:8000
```

### Authentication
The API uses JWT tokens. Include in headers:
```
Authorization: Bearer <your-jwt-token>
```

### Key Endpoints

#### Authentication
```http
POST /api/v1/auth/register
POST /api/v1/auth/login
GET  /api/v1/users/me
```

#### Stories
```http
POST /api/v1/stories           # Create new story
GET  /api/v1/stories           # List stories (with filters)
GET  /api/v1/stories/{id}      # Get story details
PUT  /api/v1/stories/{id}      # Update story
```

#### Media Upload
```http
POST /api/v1/media/upload      # Upload audio file
```

#### AI Processing
```http
POST /api/v1/ai/transcribe                    # Direct audio transcription
POST /api/v1/ai/process-story                 # Full story processing
GET  /api/v1/ai/story/{story_id}/analysis     # Get analysis results
```

#### 🆕 Text-to-Speech
```http
POST /api/v1/ai/generate-speech               # Generate TTS from text
POST /api/v1/ai/generate-story-narration      # Generate story narration
GET  /api/v1/ai/voices                        # Available TTS voices
GET  /api/v1/ai/health                        # AI services health check
```

### Example Usage

#### 1. Upload and Process a Story

```python
import requests

# 1. Register/Login
auth_response = requests.post("http://localhost:8000/api/v1/auth/login", {
    "email": "storyteller@example.com",
    "password": "password123"
})
token = auth_response.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# 2. Upload audio file
with open("story.mp3", "rb") as f:
    files = {"file": f}
    upload_response = requests.post(
        "http://localhost:8000/api/v1/media/upload",
        files=files,
        headers=headers
    )
audio_url = upload_response.json()["file_url"]

# 3. Create story
story_data = {
    "title": "Hadithi ya Simba na Kobe",  # Swahili story
    "description": "Traditional Swahili folktale about wisdom",
    "storyteller_name": "Bibi Amina",
    "language": "sw-KE",  # Swahili (Kenya)
    "geo_location": {"country": "Kenya", "region": "Coast"},
    "consent_given": True,
    "audio_file_url": audio_url
}
story_response = requests.post(
    "http://localhost:8000/api/v1/stories",
    json=story_data,
    headers=headers
)
story_id = story_response.json()["id"]

# 4. Start AI processing
process_response = requests.post(
    "http://localhost:8000/api/v1/ai/process-story",
    json={"story_id": story_id},
    headers=headers
)

# 5. Check analysis results (after processing completes)
analysis_response = requests.get(
    f"http://localhost:8000/api/v1/ai/story/{story_id}/analysis"
)
analysis = analysis_response.json()
```

#### 2. 🆕 Generate Swahili Audio Narration

```python
# Generate TTS from Swahili text
tts_data = {
    "text": "Hadithi, hadithi? Hadithi njoo! Palikuwa na mfalme mkuu...",
    "language": "sw",
    "voice_name": "Zephyr",  # Recommended for Swahili
    "return_file": True
}

tts_response = requests.post(
    "http://localhost:8000/api/v1/ai/generate-speech",
    data=tts_data,
    headers=headers
)

# Save generated audio
with open("swahili_narration.wav", "wb") as f:
    f.write(tts_response.content)

print("Swahili narration generated successfully!")
```

#### 3. Generate Story Narration from Existing Story

```python
# Generate narration for processed story
narration_data = {
    "story_id": story_id,
    "language": "sw",  # Swahili narration
    "voice_name": "Zephyr",
    "use_translation": False  # Use original language
}

narration_response = requests.post(
    "http://localhost:8000/api/v1/ai/generate-story-narration",
    data=narration_data,
    headers=headers
)

# Save story narration
with open(f"story_narration_{story_id}.wav", "wb") as f:
    f.write(narration_response.content)
```

#### 4. Test Multi-Speaker Dialog

```python
# Multi-speaker Swahili dialog
dialog_text = """
Speaker 1: Hujambo rafiki yangu! Habari za asubuhi?
Speaker 2: Sijambo, asante sana! Nimefurahi kukuona.
Speaker 1: Leo nina hadithi nzuri ya kuwaambia.
Speaker 2: Hebu niambie! Napenda sana hadithi za zamani.
"""

dialog_data = {
    "text": dialog_text,
    "language": "sw",
    "multi_speaker": True,
    "return_file": True
}

dialog_response = requests.post(
    "http://localhost:8000/api/v1/ai/generate-speech",
    data=dialog_data,
    headers=headers
)

with open("swahili_dialog.wav", "wb") as f:
    f.write(dialog_response.content)
```

## 🔧 Development

### Project Structure
```
ai-griot-backend/
├── app/
│   ├── api/v1/
│   │   └── endpoints/
│   │       ├── auth.py
│   │       ├── users.py
│   │       ├── stories.py
│   │       ├── media.py
│   │       └── ai_processing.py        # Enhanced with TTS
│   ├── core/
│   │   ├── config.py                   # TTS + Swahili settings
│   │   ├── database.py
│   │   └── security.py
│   ├── models/
│   │   ├── user.py
│   │   └── story.py
│   └── main.py
├── requirements.txt                    # Includes google-genai
├── setup_mvp.py                       # Enhanced setup
├── test_swahili_tts.py                # TTS testing
└── README.md
```

### AI Processing Pipeline

The backend processes stories through multiple stages:

1. **Audio Upload**: Store audio file in cloud storage
2. **Speech-to-Text**: Convert audio to text with timestamps using Google Cloud Speech-to-Text
3. **Text Enhancement**: Clean and improve transcript using Google Gemini
4. **Content Analysis**: Extract cultural context, sentiment, and entities using Gemini
5. **Translation**: Generate translations in multiple languages using Google Translate
6. **🆕 Audio Narration**: Generate TTS narrations using Gemini TTS
7. **Storage**: Save all results to database

### Running Tests
```bash
pytest tests/ -v

# Test Swahili TTS functionality
python test_swahili_tts.py
```

### Database Migrations
```bash
# Generate migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head
```

## 🌍 AI Features Deep Dive

### Speech-to-Text Processing
- **Engine**: Google Cloud Speech-to-Text API
- **Features**: Word-level timestamps, automatic punctuation, multiple languages
- **Audio Formats**: MP3, WAV, FLAC, M4A, OGG, WebM
- **Chunking**: Automatic splitting for long audio files
- **🇹🇿 Swahili Support**: Optimized for "sw-KE" and "sw-TZ" language codes

### Gemini AI Integration
- **Text Enhancement**: Cleans transcription errors while preserving authentic voice
- **Cultural Analysis**: Identifies themes, cultural elements, story types, moral lessons
- **Entity Extraction**: Finds people, places, objects, and concepts mentioned
- **Sentiment Analysis**: Determines emotional tone and themes
- **🇹🇿 Swahili Specialization**: Recognizes traditional storytelling elements, Arabic influences, cultural honorifics

### 🆕 Text-to-Speech System
- **Engine**: Google Gemini TTS (gemini-2.5-pro-preview-tts)
- **Voice Quality**: Natural, human-like speech generation
- **Multi-Speaker Support**: Different voices for dialog and conversation
- **Swahili Optimization**: Voices tuned for East African languages and pronunciation
- **Traditional Elements**: Preserves "Hadithi, hadithi?" and other storytelling markers
- **Available Voices**: 
  - **Zephyr**: Warm, clear - excellent for Swahili stories
  - **Kore**: Rich, expressive - good for dramatic narratives
  - **Puck**: Playful, energetic - suitable for children's stories
  - **Charon**: Deep, authoritative - good for serious content

### Translation System
- **Engine**: Google Cloud Translate API
- **Languages**: Swahili, English, French, Spanish, Arabic (Swahili prioritized)
- **Context-Aware**: Preserves cultural terms and meaning
- **Quality**: Professional-grade translation suitable for preservation

## 🎯 Frontend Integration

### Enhanced Story Upload Flow
```javascript
// Frontend integration with TTS capabilities
const uploadStoryWithTTS = async (audioFile, metadata) => {
  // 1. Upload audio
  const formData = new FormData();
  formData.append('file', audioFile);
  
  const uploadResponse = await fetch('/api/v1/media/upload', {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${token}` },
    body: formData
  });
  
  const { file_url } = await uploadResponse.json();
  
  // 2. Create story
  const storyResponse = await fetch('/api/v1/stories', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      ...metadata,
      audio_file_url: file_url
    })
  });
  
  const story = await storyResponse.json();
  
  // 3. Start processing
  await fetch('/api/v1/ai/process-story', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ story_id: story.id })
  });
  
  // 4. Generate narration after processing
  setTimeout(async () => {
    const narrationResponse = await fetch('/api/v1/ai/generate-story-narration', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}` },
      body: new FormData([
        ['story_id', story.id],
        ['language', 'sw'],
        ['voice_name', 'Zephyr']
      ])
    });
    
    const audioBlob = await narrationResponse.blob();
    // Play or download the generated narration
  }, 30000); // Wait for processing to complete
  
  return story;
};
```

### TTS Component Example
```javascript
// React component for TTS functionality
const TTSPlayer = ({ text, language = "sw" }) => {
  const [audioUrl, setAudioUrl] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);
  
  const generateSpeech = async () => {
    setIsGenerating(true);
    
    const formData = new FormData();
    formData.append('text', text);
    formData.append('language', language);
    formData.append('voice_name', 'Zephyr');
    
    try {
      const response = await fetch('/api/v1/ai/generate-speech', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
        body: formData
      });
      
      const audioBlob = await response.blob();
      const url = URL.createObjectURL(audioBlob);
      setAudioUrl(url);
    } catch (error) {
      console.error('TTS generation failed:', error);
    } finally {
      setIsGenerating(false);
    }
  };
  
  return (
    <div>
      <button onClick={generateSpeech} disabled={isGenerating}>
        {isGenerating ? 'Generating...' : '🔊 Generate Audio'}
      </button>
      {audioUrl && (
        <audio controls src={audioUrl}>
          Your browser does not support the audio element.
        </audio>
      )}
    </div>
  );
};
```

## 🔒 Security

- **JWT Authentication**: Secure token-based auth
- **Input Validation**: Comprehensive request validation
- **File Upload Security**: Type and size restrictions
- **CORS Configuration**: Proper cross-origin handling
- **Environment Variables**: Secure credential management
- **API Rate Limiting**: Prevents abuse of TTS and other AI services

## 📊 Monitoring & Health

### Health Check
```http
GET /health
```

### AI Services Status
```http
GET /api/v1/ai/health
```

Returns status of:
- Google Cloud Speech-to-Text
- Google Cloud Translate
- Google Gemini (Content Analysis)
- Google Gemini TTS
- spaCy NER
- Database connection

### TTS Voice Information
```http
GET /api/v1/ai/voices
```

Returns:
```json
{
  "available_voices": ["Zephyr", "Puck", "Charon", "Kore"],
  "swahili_recommended": ["Zephyr", "Kore"],
  "default_voice": "Zephyr",
  "supported_languages": ["sw-KE", "sw-TZ", "en-US", "fr-FR", ...]
}
```

## 🚀 Production Deployment

### Environment Variables
Set these in production:
```bash
DEBUG=false
DATABASE_URL=postgresql://user:pass@host:port/db
GOOGLE_APPLICATION_CREDENTIALS=/app/service-account.json
GEMINI_API_KEY=your-production-gemini-key
TTS_MODEL=gemini-2.5-pro-preview-tts
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
SECRET_KEY=your-production-secret-key
```

### Docker Deployment
```dockerfile
# Example Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🧪 Testing TTS Features

### Quick TTS Test
```bash
# Test Swahili TTS
python test_swahili_tts.py

# Manual curl test
curl -X POST "http://localhost:8000/api/v1/ai/generate-speech" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "text=Hadithi, hadithi? Hadithi njoo!" \
  -F "language=sw" \
  -F "voice_name=Zephyr" \
  --output test_swahili.wav
```

### Test Traditional Swahili Story
```python
traditional_story = """
Hadithi, hadithi? Hadithi njoo!

Palikuwa na mfalme mkuu wa Afrika Mashariki. Alikuwa na hekima nyingi.
Siku moja, alikutana na kobe mkongwe aliyekuwa na uzoefu mkuu.

Mfalme akasema: "Shangazi, hekima ni utajiri mkuu kuliko dhahabu."
Kobe akamjibu: "Ndio mfalme, hekima ni mali ya thamani."

Hadithi inaishia hapa. Funzo: Hekima ni mali ya thamani zaidi.
"""

# Generate with traditional elements preserved
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality (including TTS tests)
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: API docs at `/docs` when server is running
- **Issues**: Report bugs via GitHub issues
- **Community**: Join our discussions for help and feature requests
- **TTS Testing**: Use `test_swahili_tts.py` for testing TTS functionality

---

## 🌟 What's New in This Version

### 🆕 Major Features Added:
1. **Gemini Text-to-Speech Integration**: Generate natural audio narrations
2. **Enhanced Swahili Support**: Optimized for East African storytelling
3. **Multi-Speaker Dialog**: Different voices for conversations
4. **Traditional Elements Preservation**: Maintains "Hadithi, hadithi?" and cultural markers
5. **Voice Optimization**: Zephyr and Kore voices tuned for Swahili
6. **Complete Audio Pipeline**: Upload → Transcribe → Analyze → Translate → Generate Audio

### 🎯 Use Cases Enabled:
- **Accessibility**: Generate audio versions of written stories
- **Language Learning**: Hear proper pronunciation of Swahili stories
- **Cultural Preservation**: Maintain traditional storytelling elements
- **Global Reach**: Translate and narrate stories in multiple languages

**Asante sana! Happy storytelling with AI! 🎭✨**

*Preserving oral traditions with the power of Google's AI*
