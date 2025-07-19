from pydantic_settings import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "Digital Griot API"
    DEBUG: bool = False
    VERSION: str = "1.0.0"
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    BASE_URL: str = "http://localhost:8000"  # Base URL for constructing full URLs
    
    # CORS settings
    ALLOWED_HOSTS: List[str] = ["http://localhost:3000", "http://localhost:8080", "http://localhost:5173"]
    
    # Database settings
    DATABASE_URL: str = "postgresql+asyncpg://yannickgbaka:postgres@localhost:5432/digital_griot"
    
    # JWT settings
    SECRET_KEY: str = "your-secret-key-here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # AWS S3 settings
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str = "digital-griot-audio"
    
    # Google Cloud settings
    GOOGLE_CLOUD_PROJECT_ID: Optional[str] = None
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    
    # Redis settings
    REDIS_URL: str = "redis://localhost:6379"
    
    # File upload settings
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_AUDIO_FORMATS: List[str] = [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"]
    
    # AI Processing settings - Enhanced with Swahili focus
    SUPPORTED_LANGUAGES: List[str] = [
        "sw-KE",  # Swahili (Kenya) - Primary focus
        "sw-TZ",  # Swahili (Tanzania)
        "en-US",  # English (US)
        "en-GB",  # English (UK)
        "fr-FR",  # French
        "es-ES",  # Spanish
        "ar-SA",  # Arabic
        "hi-IN",  # Hindi
        "pt-BR",  # Portuguese (Brazil)
        "de-DE",  # German
        "it-IT",  # Italian
        "ja-JP",  # Japanese
        "zh-CN"   # Chinese (Mandarin)
    ]
    TRANSLATION_TARGET_LANGUAGES: List[str] = ["sw", "en", "fr", "es", "ar"]  # Swahili first
    
    # Audio processing settings
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHANNELS: int = 1
    CHUNK_DURATION_SECONDS: int = 30  # Split audio into 30-second chunks for processing
    
    # Text-to-Speech settings
    TTS_MODEL: str = "gemini-2.5-pro-preview-tts"
    TTS_DEFAULT_VOICE: str = "Zephyr"  # Available: Zephyr, Puck, Charon, Kore
    TTS_SAMPLE_RATE: int = 24000
    TTS_BITS_PER_SAMPLE: int = 16
    
    # Swahili-specific settings
    SWAHILI_VOICES: List[str] = ["Zephyr", "Kore"]  # Voices that work well with Swahili
    SWAHILI_CULTURAL_CONTEXT: bool = True  # Enable enhanced cultural analysis for Swahili content
    
    class Config:
        env_file = ".env"

settings = Settings() 