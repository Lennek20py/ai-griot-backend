import asyncio
import json
import uuid
import tempfile
import os
import io
import struct
import mimetypes
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import google.generativeai as genai
from google import genai as genai_tts
from google.genai import types
from google.cloud import speech
from google.cloud import translate_v2 as translate_client
from google.cloud import storage
import spacy
from transformers import pipeline
import librosa
import numpy as np
from pydub import AudioSegment
import base64
import httpx
from pydantic import BaseModel

from app.core.database import get_db
from app.core.security import get_current_active_user
from app.core.config import settings
from app.models.user import User
from app.models.story import (
    Story, StoryStatus, Transcript, Translation, Analytics,
    TranscriptCreate, TranslationCreate
)

router = APIRouter()

# Request models
class ProcessStoryRequest(BaseModel):
    story_id: str

# Initialize Google clients
speech_client = None
translate_instance = None
storage_client = None
genai_client = None
genai_tts_client = None

# Initialize Google Cloud clients
try:
    if settings.GOOGLE_APPLICATION_CREDENTIALS:
        speech_client = speech.SpeechClient()
        translate_instance = translate_client.Client()
        storage_client = storage.Client()
        print("✅ Google Cloud clients initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize Google Cloud clients: {e}")

# Initialize Gemini clients
try:
    if settings.GEMINI_API_KEY:
        # Initialize standard Gemini client for content generation
        genai.configure(api_key=settings.GEMINI_API_KEY)
        genai_client = genai.GenerativeModel('gemini-1.5-flash')
        
        # Initialize Gemini TTS client
        genai_tts_client = genai_tts.Client(api_key=settings.GEMINI_API_KEY)
        print("✅ Gemini clients (content + TTS) initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize Gemini clients: {e}")

# Load spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None
    print("⚠️ spaCy model not found. Install with: python -m spacy download en_core_web_sm")

# Initialize sentiment analysis pipeline
try:
    sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
except Exception:
    sentiment_analyzer = None
    print("⚠️ Sentiment analyzer not available")


class AudioProcessor:
    """Handles audio processing and chunking for optimal transcription"""
    
    @staticmethod
    def convert_to_wav(audio_file_path: str, target_sample_rate: int = 16000) -> str:
        """Convert audio file to WAV format with specified sample rate"""
        try:
            audio = AudioSegment.from_file(audio_file_path)
            audio = audio.set_frame_rate(target_sample_rate)
            audio = audio.set_channels(1)  # Convert to mono
            
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                audio.export(temp_file.name, format="wav")
                return temp_file.name
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Audio conversion failed: {str(e)}"
            )
    
    @staticmethod
    def split_audio_by_duration(audio_file_path: str, chunk_duration: int = 30) -> List[str]:
        """Split audio into chunks of specified duration (in seconds)"""
        try:
            audio = AudioSegment.from_wav(audio_file_path)
            chunk_length_ms = chunk_duration * 1000
            chunks = []
            
            for i in range(0, len(audio), chunk_length_ms):
                chunk = audio[i:i + chunk_length_ms]
                
                # Create temporary file for chunk
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    chunk.export(temp_file.name, format="wav")
                    chunks.append(temp_file.name)
            
            return chunks
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Audio splitting failed: {str(e)}"
            )


class TranslationProcessor:
    """Handles translation using Google Cloud Translate"""
    
    @staticmethod
    async def translate_text(text: str, target_language: str, source_language: str = "auto") -> Dict[str, Any]:
        """Translate text to target language"""
        if not translate_instance:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Google Translate client not configured"
            )
        
        try:
            result = translate_instance.translate(
                text,
                target_language=target_language,
                source_language=source_language
            )
            
            return {
                "translated_text": result["translatedText"],
                "source_language": result.get("detectedSourceLanguage", source_language),
                "target_language": target_language,
                "confidence": 1.0  # Google Translate doesn't provide confidence scores
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Translation failed: {str(e)}"
            )


class SpeechToTextProcessor:
    """Handles speech-to-text conversion using Google Cloud Speech-to-Text"""
    
    @staticmethod
    async def transcribe_audio_chunk(audio_file_path: str, language_code: str = "en-US") -> Dict[str, Any]:
        """Transcribe a single audio chunk using Google Cloud Speech-to-Text"""
        if not speech_client:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Google Cloud Speech client not configured"
            )
        
        try:
            # Read audio file
            with open(audio_file_path, "rb") as audio_file:
                content = audio_file.read()
            
            # Configure recognition request
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=settings.AUDIO_SAMPLE_RATE,
                language_code=language_code,
                enable_word_time_offsets=True,
                enable_automatic_punctuation=True,
                model="latest_long",
                use_enhanced=True
            )
            
            # Perform transcription
            response = speech_client.recognize(config=config, audio=audio)
            
            # Process results
            transcript_data = {
                "text": "",
                "words": [],
                "confidence": 0.0
            }
            
            for result in response.results:
                alternative = result.alternatives[0]
                transcript_data["text"] += alternative.transcript + " "
                transcript_data["confidence"] = max(transcript_data["confidence"], alternative.confidence)
                
                # Extract word-level timestamps
                for word_info in alternative.words:
                    word_data = {
                        "word": word_info.word,
                        "start_time": word_info.start_time.total_seconds(),
                        "end_time": word_info.end_time.total_seconds()
                    }
                    transcript_data["words"].append(word_data)
            
            return transcript_data
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Transcription failed: {str(e)}"
            )
    
    @staticmethod
    async def transcribe_full_audio(audio_file_path: str, language_code: str = "en-US") -> Dict[str, Any]:
        """Transcribe complete audio file by processing chunks"""
        try:
            # Convert audio to optimal format
            wav_file_path = AudioProcessor.convert_to_wav(audio_file_path)
            
            # Split into chunks if needed
            audio_chunks = AudioProcessor.split_audio_by_duration(
                wav_file_path, 
                settings.CHUNK_DURATION_SECONDS
            )
            
            # Transcribe each chunk
            all_transcripts = []
            total_duration = 0
            
            for i, chunk_path in enumerate(audio_chunks):
                chunk_transcript = await SpeechToTextProcessor.transcribe_audio_chunk(
                    chunk_path, language_code
                )
                
                # Adjust timestamps for chunk offset
                for word in chunk_transcript["words"]:
                    word["start_time"] += total_duration
                    word["end_time"] += total_duration
                
                # Get chunk duration for next offset
                chunk_audio = AudioSegment.from_wav(chunk_path)
                chunk_duration = len(chunk_audio) / 1000.0  # Convert to seconds
                total_duration += chunk_duration
                
                all_transcripts.append(chunk_transcript)
                
                # Clean up temporary chunk file
                os.unlink(chunk_path)
            
            # Clean up temporary WAV file
            os.unlink(wav_file_path)
            
            # Combine all transcripts
            combined_transcript = {
                "text": " ".join([t["text"].strip() for t in all_transcripts]),
                "words": [],
                "confidence": sum([t["confidence"] for t in all_transcripts]) / len(all_transcripts),
                "chunks": all_transcripts
            }
            
            # Combine all words
            for transcript in all_transcripts:
                combined_transcript["words"].extend(transcript["words"])
            
            return combined_transcript
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Full audio transcription failed: {str(e)}"
            )


class TextToSpeechProcessor:
    """Handles text-to-speech conversion using Google Gemini TTS"""
    
    @staticmethod
    def parse_audio_mime_type(mime_type: str) -> Dict[str, int]:
        """Parse bits per sample and rate from an audio MIME type string"""
        bits_per_sample = settings.TTS_BITS_PER_SAMPLE
        rate = settings.TTS_SAMPLE_RATE

        # Extract rate from parameters
        parts = mime_type.split(";")
        for param in parts:
            param = param.strip()
            if param.lower().startswith("rate="):
                try:
                    rate_str = param.split("=", 1)[1]
                    rate = int(rate_str)
                except (ValueError, IndexError):
                    pass
            elif param.startswith("audio/L"):
                try:
                    bits_per_sample = int(param.split("L", 1)[1])
                except (ValueError, IndexError):
                    pass

        return {"bits_per_sample": bits_per_sample, "rate": rate}
    
    @staticmethod
    def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
        """Convert audio data to WAV format"""
        parameters = TextToSpeechProcessor.parse_audio_mime_type(mime_type)
        bits_per_sample = parameters["bits_per_sample"]
        sample_rate = parameters["rate"]
        num_channels = 1
        data_size = len(audio_data)
        bytes_per_sample = bits_per_sample // 8
        block_align = num_channels * bytes_per_sample
        byte_rate = sample_rate * block_align
        chunk_size = 36 + data_size

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",          # ChunkID
            chunk_size,       # ChunkSize
            b"WAVE",          # Format
            b"fmt ",          # Subchunk1ID
            16,               # Subchunk1Size (16 for PCM)
            1,                # AudioFormat (1 for PCM)
            num_channels,     # NumChannels
            sample_rate,      # SampleRate
            byte_rate,        # ByteRate
            block_align,      # BlockAlign
            bits_per_sample,  # BitsPerSample
            b"data",          # Subchunk2ID
            data_size         # Subchunk2Size
        )
        return header + audio_data
    
    @staticmethod
    async def generate_speech(
        text: str, 
        language: str = "sw",  # Default to Swahili
        voice_name: str = None,
        multi_speaker: bool = False
    ) -> bytes:
        """Generate speech from text using Gemini TTS"""
        if not genai_tts_client:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Gemini TTS client not configured"
            )
        
        try:
            # Select appropriate voice for language
            if voice_name is None:
                if language.startswith("sw"):  # Swahili
                    voice_name = settings.SWAHILI_VOICES[0]
                else:
                    voice_name = settings.TTS_DEFAULT_VOICE
            
            # Prepare content for TTS
            if multi_speaker and "Speaker" in text:
                # Multi-speaker setup for dialog
                contents = [
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=text)]
                    )
                ]
                
                # Configure multi-speaker voices
                speech_config = types.SpeechConfig(
                    multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                        speaker_voice_configs=[
                            types.SpeakerVoiceConfig(
                                speaker="Speaker 1",
                                voice_config=types.VoiceConfig(
                                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                        voice_name=voice_name
                                    )
                                ),
                            ),
                            types.SpeakerVoiceConfig(
                                speaker="Speaker 2", 
                                voice_config=types.VoiceConfig(
                                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                        voice_name=settings.SWAHILI_VOICES[1] if len(settings.SWAHILI_VOICES) > 1 else voice_name
                                    )
                                ),
                            )
                        ]
                    )
                )
            else:
                # Single speaker setup
                contents = [
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=text)]
                    )
                ]
                
                speech_config = types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name
                        )
                    )
                )
            
            # Configure generation
            generate_content_config = types.GenerateContentConfig(
                temperature=0.8,
                response_modalities=["audio"],
                speech_config=speech_config
            )
            
            # Generate audio
            audio_chunks = []
            for chunk in genai_tts_client.models.generate_content_stream(
                model=settings.TTS_MODEL,
                contents=contents,
                config=generate_content_config,
            ):
                if (
                    chunk.candidates is None
                    or chunk.candidates[0].content is None
                    or chunk.candidates[0].content.parts is None
                ):
                    continue
                
                part = chunk.candidates[0].content.parts[0]
                if part.inline_data and part.inline_data.data:
                    inline_data = part.inline_data
                    data_buffer = inline_data.data
                    
                    # Convert to WAV if needed
                    file_extension = mimetypes.guess_extension(inline_data.mime_type)
                    if file_extension != ".wav":
                        data_buffer = TextToSpeechProcessor.convert_to_wav(
                            inline_data.data, 
                            inline_data.mime_type
                        )
                    
                    audio_chunks.append(data_buffer)
            
            # Combine all audio chunks
            return b''.join(audio_chunks)
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Text-to-speech generation failed: {str(e)}"
            )


class GeminiProcessor:
    """Handles content analysis and enhancement using Google Gemini"""
    
    @staticmethod
    async def analyze_story_content(transcript_text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze story content for cultural context, sentiment, and entities"""
        if not genai_client:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Gemini client not configured"
            )
        
        try:
            # Enhanced prompt for Swahili cultural context
            language = metadata.get('language', 'en')
            is_swahili = language.startswith('sw')
            
            cultural_context_prompt = ""
            if is_swahili and settings.SWAHILI_CULTURAL_CONTEXT:
                cultural_context_prompt = """
            Special focus for Swahili/East African content:
            - Identify traditional Swahili storytelling elements (e.g., hadithi, methali, hekaya)
            - Look for Islamic cultural influences and traditional beliefs
            - Identify references to East African geography, animals, and customs
            - Note any Arabic loanwords or Islamic concepts
            - Identify traditional roles (e.g., mzee, bibi, kiongozi)
            - Look for moral teachings common in Swahili oral tradition
            """
            
            prompt = f"""
            Please analyze the following oral story transcript and provide detailed analysis:

            Title: {metadata.get('title', 'Untitled')}
            Origin: {metadata.get('origin', 'Unknown')}
            Language: {metadata.get('language', 'Unknown')}
            Storyteller: {metadata.get('storyteller_name', 'Anonymous')}
            
            {cultural_context_prompt}

            Transcript:
            {transcript_text}

            Please provide analysis in the following JSON format:
            {{
                "sentiment": {{
                    "overall": "positive/negative/neutral",
                    "confidence": 0.0-1.0,
                    "emotions": ["joy", "sadness", "wisdom", "caution", etc.]
                }},
                "cultural_context": {{
                    "themes": ["theme1", "theme2", ...],
                    "cultural_elements": ["element1", "element2", ...],
                    "story_type": "creation myth/folktale/proverb/legend/hekaya/hadithi/etc.",
                    "moral_lessons": ["lesson1", "lesson2", ...],
                    "traditional_elements": ["element1", "element2", ...]
                }},
                "entities": {{
                    "people": ["person1", "person2", ...],
                    "places": ["place1", "place2", ...],
                    "objects": ["object1", "object2", ...],
                    "concepts": ["concept1", "concept2", ...],
                    "animals": ["animal1", "animal2", ...]
                }},
                "language_analysis": {{
                    "register": "formal/informal/traditional",
                    "dialect_markers": ["marker1", "marker2", ...],
                    "arabic_influences": ["word1", "word2", ...] 
                }},
                "summary": "A brief 2-3 sentence summary of the story",
                "keywords": ["keyword1", "keyword2", ...],
                "cultural_significance": "Brief description of cultural importance"
            }}
            """
            
            response = genai_client.generate_content(prompt)
            
            # Parse JSON response
            try:
                analysis = json.loads(response.text.strip())
                return analysis
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "sentiment": {"overall": "neutral", "confidence": 0.5, "emotions": []},
                    "cultural_context": {"themes": [], "cultural_elements": [], "story_type": "unknown", "moral_lessons": [], "traditional_elements": []},
                    "entities": {"people": [], "places": [], "objects": [], "concepts": [], "animals": []},
                    "language_analysis": {"register": "unknown", "dialect_markers": [], "arabic_influences": []},
                    "summary": "Analysis not available",
                    "keywords": [],
                    "cultural_significance": "Requires further analysis"
                }
        
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Content analysis failed: {str(e)}"
            )
    
    @staticmethod
    async def enhance_transcript(transcript_text: str, language: str) -> str:
        """Clean and enhance transcript using Gemini"""
        if not genai_client:
            return transcript_text
        
        try:
            is_swahili = language.startswith('sw')
            language_specific_instructions = ""
            
            if is_swahili:
                language_specific_instructions = """
            Special considerations for Swahili:
            - Preserve Arabic loanwords and Islamic terminology
            - Maintain proper Swahili grammar and verb conjugations
            - Keep traditional expressions and idioms intact
            - Preserve cultural honorifics (Mzee, Bibi, Bwana, etc.)
            - Maintain storytelling markers (Hapo zamani, Palikuwa, etc.)
            """
            
            prompt = f"""
            Please clean and enhance this oral story transcript while preserving its authentic voice and cultural elements:

            Original transcript:
            {transcript_text}
            
            {language_specific_instructions}

            Please:
            1. Fix obvious transcription errors
            2. Add appropriate punctuation
            3. Maintain the storyteller's voice and style
            4. Keep cultural terms and expressions intact
            5. Ensure proper paragraph breaks for narrative flow
            6. Preserve traditional storytelling elements

            Return only the cleaned transcript without any additional commentary.
            """
            
            response = genai_client.generate_content(prompt)
            return response.text.strip()
            
        except Exception:
            return transcript_text  # Return original if enhancement fails


# API Endpoints

@router.post("/transcribe")
async def transcribe_audio_file(
    file: UploadFile = File(...),
    language: str = "sw-KE",  # Default to Swahili (Kenya)
    enhance: bool = True,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Transcribe uploaded audio file"""
    
    # Validate file type
    if not any(file.filename.lower().endswith(ext) for ext in settings.ALLOWED_AUDIO_FORMATS):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported audio format"
        )
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
    
    try:
        # Transcribe audio
        transcript_data = await SpeechToTextProcessor.transcribe_full_audio(temp_file_path, language)
        
        # Enhance transcript if requested
        if enhance and transcript_data["text"]:
            enhanced_text = await GeminiProcessor.enhance_transcript(
                transcript_data["text"], 
                language
            )
            transcript_data["enhanced_text"] = enhanced_text
        
        return {
            "transcript": transcript_data,
            "message": "Transcription completed successfully"
        }
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


@router.post("/generate-speech")
async def generate_speech_from_text(
    text: str = Form(...),
    language: str = Form(default="sw"),  # Default to Swahili
    voice_name: str = Form(default=None),
    multi_speaker: bool = Form(default=False),
    return_file: bool = Form(default=True),
    current_user: User = Depends(get_current_active_user)
):
    """Generate speech audio from text using Gemini TTS"""
    
    try:
        # Generate audio
        audio_data = await TextToSpeechProcessor.generate_speech(
            text=text,
            language=language,
            voice_name=voice_name,
            multi_speaker=multi_speaker
        )
        
        if return_file:
            # Save to temporary file and return as download
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            return FileResponse(
                temp_file_path,
                media_type="audio/wav",
                filename=f"generated_speech_{language}.wav",
                background=BackgroundTasks().add_task(os.unlink, temp_file_path)
            )
        else:
            # Return audio data directly
            return StreamingResponse(
                io.BytesIO(audio_data),
                media_type="audio/wav"
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Speech generation failed: {str(e)}"
        )


@router.post("/generate-story-narration")
async def generate_story_narration(
    story_id: str = Form(...),
    language: str = Form(default="sw"),  # Default to Swahili
    voice_name: str = Form(default=None),
    use_translation: bool = Form(default=False),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Generate audio narration for a story"""
    
    # Get story from database
    result = await db.execute(select(Story).where(Story.id == story_id))
    story = result.scalar_one_or_none()
    
    if not story:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Story not found"
        )
    
    # Get transcript
    transcript_result = await db.execute(
        select(Transcript).where(Transcript.story_id == story_id)
    )
    transcript = transcript_result.scalar_one_or_none()
    
    if not transcript:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No transcript found for this story"
        )
    
    # Determine text to use
    if use_translation and language != story.language.split("-")[0]:
        # Get translation
        translation_result = await db.execute(
            select(Translation).where(
                Translation.story_id == story_id,
                Translation.language == language
            )
        )
        translation = translation_result.scalar_one_or_none()
        
        if translation:
            text_to_speak = translation.translated_text
        else:
            # Generate translation on the fly
            enhanced_text = transcript.transcript_json.get("enhanced_text", 
                                                         transcript.transcript_json.get("original_text", ""))
            translation_result = await TranslationProcessor.translate_text(
                enhanced_text, language, story.language.split("-")[0]
            )
            text_to_speak = translation_result["translated_text"]
    else:
        # Use original enhanced transcript
        text_to_speak = transcript.transcript_json.get("enhanced_text", 
                                                      transcript.transcript_json.get("original_text", ""))
    
    # Add storytelling context for better narration
    narration_text = f"Hadithi, hadithi? Hadithi njoo! {text_to_speak}" if language.startswith("sw") else text_to_speak
    
    try:
        # Generate audio narration
        audio_data = await TextToSpeechProcessor.generate_speech(
            text=narration_text,
            language=language,
            voice_name=voice_name,
            multi_speaker=False
        )
        
        # Save to temporary file and return
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        return FileResponse(
            temp_file_path,
            media_type="audio/wav",
            filename=f"story_narration_{story.title}_{language}.wav",
            background=BackgroundTasks().add_task(os.unlink, temp_file_path)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Narration generation failed: {str(e)}"
        )


@router.post("/process-story")
async def process_complete_story(
    background_tasks: BackgroundTasks,
    request_data: ProcessStoryRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Process a complete story: transcribe, analyze, and translate"""
    
    # Get story from database
    result = await db.execute(select(Story).where(Story.id == request_data.story_id))
    story = result.scalar_one_or_none()
    
    if not story:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Story not found"
        )
    
    if story.contributor_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to process this story"
        )
    
    # Add background task to process the story
    background_tasks.add_task(process_story_background, request_data.story_id, db)
    
    return {
        "message": "Story processing started",
        "story_id": request_data.story_id,
        "status": "processing"
    }


async def process_story_background(story_id: str, db: AsyncSession):
    """Background task to process story completely"""
    try:
        # Get story from database
        result = await db.execute(select(Story).where(Story.id == story_id))
        story = result.scalar_one_or_none()
        
        if not story:
            return
        
        # Download audio file
        audio_file_path = await download_audio_file(story.audio_file_url)
        
        try:
            # Step 1: Transcribe audio
            transcript_data = await SpeechToTextProcessor.transcribe_full_audio(
                audio_file_path, 
                story.language
            )
            
            # Step 2: Enhance transcript
            enhanced_text = await GeminiProcessor.enhance_transcript(
                transcript_data["text"], 
                story.language
            )
            
            # Step 3: Analyze content
            metadata = {
                "title": story.title,
                "origin": story.geo_location,
                "language": story.language,
                "storyteller_name": story.storyteller_name
            }
            
            content_analysis = await GeminiProcessor.analyze_story_content(
                enhanced_text, 
                metadata
            )
            
            # Step 4: Save transcript
            transcript = Transcript(
                story_id=story.id,
                transcript_json={
                    "original_text": transcript_data["text"],
                    "enhanced_text": enhanced_text,
                    "words": transcript_data["words"],
                    "confidence": transcript_data["confidence"],
                    "analysis": content_analysis
                },
                language=story.language
            )
            
            db.add(transcript)
            
            # Step 5: Generate translations
            for target_lang in settings.TRANSLATION_TARGET_LANGUAGES:
                if target_lang != story.language.split("-")[0]:  # Don't translate to same language
                    try:
                        translation_result = await TranslationProcessor.translate_text(
                            enhanced_text, 
                            target_lang, 
                            story.language.split("-")[0]
                        )
                    
                        translation = Translation(
                                story_id=story.id,
                                translated_text=translation_result["translated_text"],
                                language=target_lang
                        )
                        
                        db.add(translation)
            
                    except Exception as e:
                        print(f"Translation to {target_lang} failed: {e}")
            
            # Step 6: Update story status
            story.status = StoryStatus.PUBLISHED
            
            # Step 7: Initialize analytics
            analytics = Analytics(
                story_id=story.id,
                views=0,
                listens=0,
                avg_rating=0.0
            )
            
            db.add(analytics)
            await db.commit()
            
        finally:
            # Clean up downloaded audio file
            if os.path.exists(audio_file_path):
                os.unlink(audio_file_path)
            
    except Exception as e:
        print(f"Story processing failed for {story_id}: {e}")
        # Update story status to reflect error
        if story:
            story.status = StoryStatus.REJECTED
            await db.commit()


async def download_audio_file(audio_url: str) -> str:
    """Download audio file from URL to temporary location"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(audio_url)
            response.raise_for_status()
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                temp_file.write(response.content)
                return temp_file.name
                
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download audio file: {str(e)}"
        )


@router.get("/story/{story_id}/analysis")
async def get_story_analysis(
    story_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get comprehensive analysis of a processed story"""
    
    # Get story with transcript and translations
    result = await db.execute(
        select(Story).where(Story.id == story_id)
    )
    story = result.scalar_one_or_none()
    
    if not story:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Story not found"
        )
    
    # Get transcript
    transcript_result = await db.execute(
        select(Transcript).where(Transcript.story_id == story_id)
    )
    transcript = transcript_result.scalar_one_or_none()
    
    # Get translations
    translations_result = await db.execute(
        select(Translation).where(Translation.story_id == story_id)
    )
    translations = translations_result.scalars().all()
    
    # Get analytics
    analytics_result = await db.execute(
        select(Analytics).where(Analytics.story_id == story_id)
    )
    analytics = analytics_result.scalar_one_or_none()
    
    return {
        "story": {
            "id": str(story.id),
            "title": story.title,
            "description": story.description,
            "storyteller_name": story.storyteller_name,
            "language": story.language,
            "status": story.status,
            "created_at": story.created_at.isoformat()
        },
        "transcript": transcript.transcript_json if transcript else None,
        "translations": [
            {
                "language": t.language,
                "text": t.translated_text
            } for t in translations
        ],
        "analytics": {
            "views": analytics.views if analytics else 0,
            "listens": analytics.listens if analytics else 0,
            "avg_rating": analytics.avg_rating if analytics else 0.0
        } if analytics else None
    }


@router.get("/voices")
async def get_available_voices():
    """Get list of available TTS voices"""
    return {
        "available_voices": ["Zephyr", "Puck", "Charon", "Kore"],
        "swahili_recommended": settings.SWAHILI_VOICES,
        "default_voice": settings.TTS_DEFAULT_VOICE,
        "supported_languages": settings.SUPPORTED_LANGUAGES
    }


@router.get("/health")
async def ai_health_check():
    """Check AI services health"""
    services = {
        "google_cloud_speech": speech_client is not None,
        "google_cloud_translate": translate_instance is not None,
        "google_gemini_content": genai_client is not None,
        "google_gemini_tts": genai_tts_client is not None,
        "spacy_ner": nlp is not None,
        "sentiment_analysis": sentiment_analyzer is not None
    }
    
    return {
        "status": "healthy",
        "services": services,
        "supported_languages": settings.SUPPORTED_LANGUAGES,
        "translation_targets": settings.TRANSLATION_TARGET_LANGUAGES,
        "tts_model": settings.TTS_MODEL,
        "swahili_support": True
    } 