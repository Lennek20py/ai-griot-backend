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
import enum
from datetime import datetime
from pydantic import Field
from sqlalchemy import text

from app.core.database import get_db
from app.core.security import get_current_active_user
from app.core.config import settings
from app.models.user import User
from app.models.story import (
    Story, StoryStatus, Transcript, Translation, Analytics,
    TranscriptCreate, TranslationCreate, Paragraph, Illustration
)

# Import illustration generation modules
from app.tasks.illustration import ParagraphProcessor, IllustrationGenerator

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
    """Handles translation using Google Cloud Translate with Gemini fallback"""
    
    @staticmethod
    async def translate_text(text: str, target_language: str, source_language: str = "auto") -> Dict[str, Any]:
        """Translate text to target language with fallback to Gemini"""
        
        # Try Google Cloud Translate first
        if translate_instance:
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
                    "confidence": 1.0,  # Google Translate doesn't provide confidence scores
                    "method": "google_translate"
                }
            except Exception as e:
                print(f"⚠️ Google Translate failed for {target_language}: {e}")
            
        # Fallback to Gemini for translation
        if genai_client:
            try:
                return await TranslationProcessor.translate_with_gemini(text, target_language, source_language)
            except Exception as e:
                print(f"⚠️ Gemini translation failed for {target_language}: {e}")
        
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Translation failed: No translation service available for {target_language}"
            )
    
    @staticmethod
    async def translate_with_gemini(text: str, target_language: str, source_language: str = "auto") -> Dict[str, Any]:
        """Translate text using Gemini AI"""
        if not genai_client:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Gemini client not configured"
            )
        
        # Language name mapping
        language_names = {
            "sw": "Swahili",
            "en": "English", 
            "fr": "French",
            "es": "Spanish",
            "ar": "Arabic",
            "pt": "Portuguese",
            "de": "German",
            "it": "Italian",
            "hi": "Hindi",
            "ja": "Japanese",
            "zh": "Chinese"
        }
        
        target_name = language_names.get(target_language, target_language)
        source_name = language_names.get(source_language.split("-")[0] if source_language != "auto" else "auto", "the source language")
        
        # Special instructions for different target languages
        special_instructions = ""
        if target_language == "sw":
            special_instructions = """
            Special considerations for Swahili translation:
            - Use proper Swahili grammar and verb conjugations
            - Preserve cultural concepts and traditional terms
            - Include appropriate honorifics (Mzee, Bibi, Bwana, etc.)
            - Maintain storytelling elements (Hadithi, palikuwa, etc.)
            - Use Arabic loanwords where appropriate
            """
        elif target_language == "ar":
            special_instructions = """
            Special considerations for Arabic translation:
            - Use classical Arabic for formal content
            - Preserve Islamic and cultural terminology
            - Maintain proper Arabic grammar and structure
            """
        
        prompt = f"""
        Please translate the following text from {source_name} to {target_name}.
        
        {special_instructions}
        
        Requirements:
        - Provide an accurate, natural translation
        - Preserve the meaning and cultural context
        - Maintain the storytelling style and tone
        - Keep proper nouns and names as appropriate
        
        Text to translate:
        {text}
        
        Please respond with only the translated text, no additional commentary.
        """
        
        try:
            response = genai_client.generate_content(prompt)
            translated_text = response.text.strip()
            
            return {
                "translated_text": translated_text,
                "source_language": source_language,
                "target_language": target_language,
                "confidence": 0.8,  # Estimated confidence for Gemini
                "method": "gemini"
            }
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Gemini translation failed: {str(e)}"
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


class GeminiTranscriptionProcessor:
    """Handles audio transcription using Google Gemini"""
    
    @staticmethod
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
    
    @staticmethod
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
        import re
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
    
    @staticmethod
    async def transcribe_audio_with_gemini(audio_file_path: str, language: str = "sw-KE") -> Dict[str, Any]:
        """Transcribe audio using Gemini's audio capabilities"""
        if not genai_client:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Gemini client not configured"
            )
        
        try:
            # Read audio file
            with open(audio_file_path, "rb") as audio_file:
                audio_content = audio_file.read()
            
            # Convert audio to base64 for Gemini
            audio_base64 = base64.b64encode(audio_content).decode('utf-8')
            
            # Prepare prompt for transcription
            is_swahili = language.startswith('sw')
            language_instructions = ""
            if is_swahili:
                language_instructions = """
                This is a Swahili audio story. Please:
                - Preserve traditional Swahili storytelling elements (hadithi, methali, hekaya)
                - Maintain proper Swahili grammar and verb conjugations
                - Keep Arabic loanwords and Islamic terminology intact
                - Preserve cultural honorifics (Mzee, Bibi, Bwana, etc.)
                - Include traditional expressions and idioms
                """
            
            prompt = f"""
            Please transcribe this audio file accurately. 

            Language: {language}
            {language_instructions}

            Please return the result in this EXACT JSON format:
            {{
                "text": "full transcribed text",
                "confidence": 0.95,
                "language_detected": "{language}",
                "words": [
                    {{"word": "word1", "start_time": 0.0, "end_time": 1.0}},
                    {{"word": "word2", "start_time": 1.0, "end_time": 2.0}}
                ]
            }}

            IMPORTANT: 
            - Return ONLY valid JSON, no extra text or formatting
            - Each word must have "word", "start_time", and "end_time" fields
            - start_time must be less than end_time for each word
            - If you cannot determine exact word timings, provide an estimated breakdown
            """
            
            # Create audio part for Gemini
            audio_part = {
                "mime_type": "audio/wav",
                "data": audio_base64
            }
            
            # Generate transcription
            response = genai_client.generate_content([prompt, audio_part])
            
            try:
                # Clean the response text to handle potential JSON formatting issues
                response_text = response.text.strip()
                
                # Remove any markdown formatting that might be present
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                
                # Parse JSON response
                result = json.loads(response_text)
                
                # Extract and validate the response
                text = result.get("text", "")
                confidence = result.get("confidence", 0.8)
                words = result.get("words", [])
                
                # Clean and validate words array
                cleaned_words = GeminiTranscriptionProcessor.clean_words_array(words)
                
                # If cleaning resulted in significant loss, regenerate from text
                if len(cleaned_words) < len(words) * 0.5 and text:
                    print(f"⚠️  Significant word data loss, regenerating from text...")
                    cleaned_words = GeminiTranscriptionProcessor.regenerate_words_from_text(text)
                
                return {
                    "text": text,
                    "confidence": confidence,
                    "words": cleaned_words,
                    "language": result.get("language_detected", language),
                    "method": "gemini"
                }
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}")
                print(f"Response text: {response.text[:500]}...")
                
                # Fallback: try to extract text and generate basic word structure
                fallback_text = response.text.strip()
                
                # Remove any markdown or formatting
                if fallback_text.startswith("```"):
                    lines = fallback_text.split('\n')
                    fallback_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else fallback_text
                
                # Generate basic word structure
                fallback_words = GeminiTranscriptionProcessor.regenerate_words_from_text(fallback_text)
                
                return {
                    "text": fallback_text,
                    "confidence": 0.7,
                    "words": fallback_words,
                    "language": language,
                    "method": "gemini_fallback"
                }
                
        except Exception as e:
            # Fallback to Google Cloud Speech-to-Text if Gemini fails
            print(f"Gemini transcription failed, falling back to Google Speech-to-Text: {e}")
            return await SpeechToTextProcessor.transcribe_full_audio(audio_file_path, language)


# Add new progress tracking models
class ProcessingStep(str, enum.Enum):
    UPLOADING = "uploading"
    TRANSCRIBING = "transcribing"
    ENHANCING = "enhancing"
    ANALYZING = "analyzing"
    TRANSLATING = "translating"
    ILLUSTRATING = "illustrating"
    PUBLISHED = "published"
    FAILED = "failed"

class StoryProcessingStatus(BaseModel):
    story_id: str
    current_step: ProcessingStep
    progress_percentage: int
    message: str
    error: Optional[str] = None
    transcript_text: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

# Store processing status in memory (in production, use Redis)
PROCESSING_STATUS_STORE: Dict[str, StoryProcessingStatus] = {}


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
        translation_result = await TranslationProcessor.translate_text(
            transcript.transcript_json.get("enhanced_text", 
                                         transcript.transcript_json.get("original_text", "")),
            language, story.language.split("-")[0]
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


async def update_processing_status(story_id: str, step: ProcessingStep, progress: int, message: str, transcript_text: Optional[str] = None, error: Optional[str] = None):
    """Update processing status for a story"""
    status = StoryProcessingStatus(
        story_id=story_id,
        current_step=step,
        progress_percentage=progress,
        message=message,
        transcript_text=transcript_text,
        error=error
    )
    PROCESSING_STATUS_STORE[story_id] = status

async def process_story_background(story_id: str, db: AsyncSession):
    """Enhanced background task to process story with detailed progress tracking"""
    
    # Create a new database session for this background task
    from app.core.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db_session:
        audio_file_path = None
        try:
            # Update status: Starting processing
            await update_processing_status(
                story_id, 
                ProcessingStep.TRANSCRIBING, 
                10, 
                "Starting AI transcription with Gemini..."
            )
            
            # Check Gemini configuration
            if not genai_client:
                error_msg = "Gemini AI client not configured. Please check GEMINI_API_KEY in settings."
                print(f"❌ {error_msg}")
                await update_processing_status(story_id, ProcessingStep.FAILED, 0, "AI service not available", error=error_msg)
                return
            
            # Get story from database
            result = await db_session.execute(select(Story).where(Story.id == story_id))
            story = result.scalar_one_or_none()
            
            if not story:
                error_msg = f"Story not found in database: {story_id}"
                print(f"❌ {error_msg}")
                await update_processing_status(story_id, ProcessingStep.FAILED, 0, "Story not found", error=error_msg)
                return
            
            if not story.audio_file_url:
                error_msg = f"No audio file URL found for story: {story_id}"
                print(f"❌ {error_msg}")
                await update_processing_status(story_id, ProcessingStep.FAILED, 0, "No audio file found", error=error_msg)
                return
            
            # Download audio file
            await update_processing_status(
                story_id, 
                ProcessingStep.TRANSCRIBING, 
                20, 
                "Downloading audio file for processing..."
            )
            
            print(f"🔄 Downloading audio file from: {story.audio_file_url}")
            
            try:
                audio_file_path = await download_audio_file(story.audio_file_url)
                print(f"✅ Audio file downloaded to: {audio_file_path}")
            except Exception as download_error:
                error_msg = f"Failed to download audio file: {str(download_error)}"
                print(f"❌ {error_msg}")
                await update_processing_status(story_id, ProcessingStep.FAILED, 0, "Audio download failed", error=error_msg)
                return
            
            # Step 1: Transcribe audio with Gemini
            await update_processing_status(
                story_id, 
                ProcessingStep.TRANSCRIBING, 
                30, 
                "Transcribing audio with Gemini AI..."
            )
            
            print(f"🔄 Starting transcription for story {story_id} with language: {story.language}")
            
            try:
                transcript_data = await GeminiTranscriptionProcessor.transcribe_audio_with_gemini(
                    audio_file_path, 
                    story.language
                )
                print(f"✅ Transcription completed. Method: {transcript_data.get('method', 'unknown')}")
                print(f"📝 Transcript length: {len(transcript_data.get('text', ''))}")
            except Exception as transcription_error:
                error_msg = f"Transcription failed: {str(transcription_error)}"
                print(f"❌ {error_msg}")
                await update_processing_status(story_id, ProcessingStep.FAILED, 0, "Transcription failed", error=error_msg)
                return
            
            await update_processing_status(
                story_id, 
                ProcessingStep.ENHANCING, 
                50, 
                "Enhancing transcript with AI...",
                transcript_text=transcript_data["text"]
            )
            
            # Step 2: Enhance transcript
            try:
                print(f"🔄 Enhancing transcript...")
                enhanced_text = await GeminiProcessor.enhance_transcript(
                    transcript_data["text"], 
                    story.language
                )
                print(f"✅ Transcript enhanced. Length: {len(enhanced_text)}")
            except Exception as enhancement_error:
                print(f"⚠️ Enhancement failed, using original: {str(enhancement_error)}")
                enhanced_text = transcript_data["text"]  # Fallback to original
            
            await update_processing_status(
                story_id, 
                ProcessingStep.ANALYZING, 
                70, 
                "Analyzing cultural context and content...",
                transcript_text=enhanced_text
            )
            
            # Step 3: Analyze content
            metadata = {
                "title": story.title,
                "origin": story.geo_location,
                "language": story.language,
                "storyteller_name": story.storyteller_name
            }
            
            try:
                print(f"🔄 Analyzing content...")
                content_analysis = await GeminiProcessor.analyze_story_content(
                    enhanced_text, 
                    metadata
                )
                print(f"✅ Content analysis completed")
            except Exception as analysis_error:
                print(f"⚠️ Content analysis failed: {str(analysis_error)}")
                content_analysis = {}  # Fallback to empty analysis
            
            # Step 4: Save transcript
            try:
                print(f"🔄 Saving transcript to database...")
                
                # Validate and clean transcript data before saving
                def validate_and_clean_transcript_data(transcript_data: Dict[str, Any]) -> Dict[str, Any]:
                    """Validate and clean transcript data to ensure proper structure"""
                    cleaned_data = transcript_data.copy()
                    
                    # Ensure words array is properly structured
                    words_array = []
                    if transcript_data.get("words"):
                        words_array = GeminiTranscriptionProcessor.clean_words_array(transcript_data["words"])
                    
                    # If no valid words or significant data loss, regenerate from text
                    if not words_array and transcript_data.get("text"):
                        print(f"⚠️  No valid words found, regenerating from text...")
                        words_array = GeminiTranscriptionProcessor.regenerate_words_from_text(transcript_data["text"])
                    elif len(words_array) < len(transcript_data.get("words", [])) * 0.5:
                        print(f"⚠️  Significant word data loss, regenerating from text...")
                        if transcript_data.get("text"):
                            words_array = GeminiTranscriptionProcessor.regenerate_words_from_text(transcript_data["text"])
                    
                    cleaned_data["words"] = words_array
                    
                    # Ensure required fields exist
                    if "original_text" not in cleaned_data:
                        cleaned_data["original_text"] = transcript_data.get("text", "")
                    
                    if "confidence" not in cleaned_data:
                        cleaned_data["confidence"] = 0.8
                    
                    return cleaned_data
                
                # Clean and validate transcript data
                cleaned_transcript_data = validate_and_clean_transcript_data(transcript_data)
                
                transcript = Transcript(
                    story_id=story.id,
                    transcript_json={
                        "original_text": cleaned_transcript_data["text"],
                        "enhanced_text": enhanced_text,
                        "words": cleaned_transcript_data["words"],
                        "confidence": cleaned_transcript_data["confidence"],
                        "analysis": content_analysis,
                        "transcription_method": cleaned_transcript_data.get("method", "gemini")
                    },
                    language=story.language,
                    confidence_score=cleaned_transcript_data["confidence"]
                )
                
                db_session.add(transcript)
                print(f"✅ Transcript saved with {len(cleaned_transcript_data['words'])} words")
            except Exception as save_error:
                error_msg = f"Failed to save transcript: {str(save_error)}"
                print(f"❌ {error_msg}")
                await update_processing_status(story_id, ProcessingStep.FAILED, 0, "Database save failed", error=error_msg)
                return
            
            await update_processing_status(
                story_id, 
                ProcessingStep.TRANSLATING, 
                85, 
                "Generating translations...",
                transcript_text=enhanced_text
            )
            
            # Step 5: Generate translations
            print(f"🔄 Generating translations...")
            translation_count = 0
            for target_lang in settings.TRANSLATION_TARGET_LANGUAGES:
                if target_lang != story.language.split("-")[0]:  # Don't translate to same language
                    try:
                        translation_result = await TranslationProcessor.translate_text(
                            enhanced_text, 
                            target_lang, 
                            story.language.split("-")[0]
                        )
                        
                        # Generate word-level timestamps for translation
                        translation_words = []
                        original_words_array = cleaned_transcript_data["words"]
                        
                        if original_words_array and len(original_words_array) > 0:
                            # Map original words to translated text
                            original_words = [word["word"] for word in original_words_array]
                            translated_text = translation_result["translated_text"]
                            
                            # Clean and validate the translated text
                            if translated_text and isinstance(translated_text, str):
                                # Simple word mapping - split translated text into words
                                import re
                                translated_words = re.findall(r'\b\w+\b', translated_text)
                                
                                if translated_words:
                                    # Estimate timing for translated words
                                    total_duration = original_words_array[-1]["end_time"] if original_words_array else 0
                                    
                                    # Ensure we have a valid duration
                                    if total_duration <= 0:
                                        total_duration = len(original_words) * 0.5  # Fallback duration
                                    
                                    # Distribute translated words evenly across the total duration
                                    word_duration = total_duration / len(translated_words)
                                    
                                    for i, word in enumerate(translated_words):
                                        start_time = i * word_duration
                                        end_time = (i + 1) * word_duration
                                        
                                        # Ensure timestamps are valid
                                        if start_time < end_time:
                                            translation_words.append({
                                                "word": word,
                                                "start_time": start_time,
                                                "end_time": end_time
                                            })
                                
                                # If no valid words were generated, create a fallback
                                if not translation_words and translated_text:
                                    translation_words = GeminiTranscriptionProcessor.regenerate_words_from_text(
                                        translated_text, 
                                        total_duration if total_duration > 0 else 300.0
                                    )
                        
                        # Ensure we have valid translation words
                        if not translation_words and translation_result.get("translated_text"):
                            # Final fallback: generate from translated text
                            translation_words = GeminiTranscriptionProcessor.regenerate_words_from_text(
                                translation_result["translated_text"]
                            )
                        
                        # Validate and clean translation words
                        cleaned_translation_words = GeminiTranscriptionProcessor.clean_words_array(translation_words)
                        
                        # If cleaning resulted in significant loss, regenerate
                        if len(cleaned_translation_words) < len(translation_words) * 0.5:
                            print(f"⚠️  Significant translation word data loss, regenerating...")
                            if translation_result.get("translated_text"):
                                cleaned_translation_words = GeminiTranscriptionProcessor.regenerate_words_from_text(
                                    translation_result["translated_text"]
                                )
                        
                        translation = Translation(
                            story_id=story.id,
                            translated_text=translation_result["translated_text"],
                            language=target_lang,
                            confidence_score=translation_result.get("confidence", 0.8)
                        )
                        
                        # Store translation with word-level timestamps
                        translation.translation_json = {
                            "text": translation_result["translated_text"],
                            "words": cleaned_translation_words,
                            "confidence": translation_result.get("confidence", 0.8),
                            "method": translation_result.get("method", "google_translate")
                        }
                        
                        db_session.add(translation)
                        translation_count += 1
                        print(f"✅ Translation to {target_lang} completed with {len(cleaned_translation_words)} words")
            
                    except Exception as e:
                        print(f"⚠️ Translation to {target_lang} failed: {e}")
            
            print(f"✅ Generated {translation_count} translations")
            
            # Step 6: Generate illustrations
            await update_processing_status(
                story_id, 
                ProcessingStep.ILLUSTRATING, 
                90, 
                "Starting illustration generation...",
                transcript_text=enhanced_text
            )
            
            print(f"🔄 Generating illustrations for story paragraphs...")
            
            try:
                # Split transcript into paragraphs
                paragraphs = ParagraphProcessor.split_into_paragraphs(
                    enhanced_text, 
                    cleaned_transcript_data.get("words", [])
                )
                
                print(f"📝 Split transcript into {len(paragraphs)} paragraphs")
                
                # Update status to show paragraph splitting
                await update_processing_status(
                    story_id, 
                    ProcessingStep.ILLUSTRATING, 
                    92, 
                    f"Preparing {len(paragraphs)} paragraphs for illustration...",
                    transcript_text=enhanced_text
                )
                
                # Save paragraphs to database
                saved_paragraphs = []
                for i, paragraph_data in enumerate(paragraphs):
                    paragraph = Paragraph(
                        story_id=story.id,
                        sequence_order=i,
                        content=paragraph_data["text"],
                        start_time=paragraph_data.get("start_time", 0.0),
                        end_time=paragraph_data.get("end_time", 0.0),
                        language=story.language,
                        word_count=paragraph_data.get("word_count")
                    )
                    db_session.add(paragraph)
                    saved_paragraphs.append(paragraph)
                
                # Commit paragraphs to get their IDs
                await db_session.commit()
                
                # Update status before starting illustration generation
                await update_processing_status(
                    story_id, 
                    ProcessingStep.ILLUSTRATING, 
                    94, 
                    "Starting AI image generation...",
                    transcript_text=enhanced_text
                )
                
                # Generate illustrations for each paragraph
                illustration_count = 0
                for i, (paragraph, paragraph_data) in enumerate(zip(saved_paragraphs, paragraphs)):
                    try:
                        # Update progress for each illustration with more detailed messages
                        progress = 94 + (i * 4 // len(paragraphs))
                        await update_processing_status(
                            story_id, 
                            ProcessingStep.ILLUSTRATING, 
                            progress, 
                            f"Generating illustration {i+1}/{len(paragraphs)}: '{paragraph_data['text'][:50]}...'",
                            transcript_text=enhanced_text
                        )
                        
                        # Add a small delay to make the process visible
                        await asyncio.sleep(1)
                        
                        # Create illustration prompt
                        prompt = IllustrationGenerator.create_illustration_prompt(
                            paragraph_data["text"], 
                            story.language,
                            {
                                "origin": story.origin,
                                "storyteller_name": story.storyteller_name,
                                "title": story.title
                            }
                        )
                        
                        # Generate image
                        illustration_result = await IllustrationGenerator.generate_image_with_gemini(prompt)
                        
                        if illustration_result:
                            # Save illustration to database
                            illustration = Illustration(
                                paragraph_id=paragraph.id,
                                image_url=illustration_result["image_url"],
                                prompt_used=illustration_result["prompt_used"],
                                style=illustration_result["style_description"],
                                status="completed",
                                generation_time=2.5  # Simulated generation time
                            )
                            db_session.add(illustration)
                            illustration_count += 1
                            print(f"✅ Generated illustration {i+1}/{len(paragraphs)}")
                            
                            # Update status after successful generation
                            await update_processing_status(
                                story_id, 
                                ProcessingStep.ILLUSTRATING, 
                                progress + 1, 
                                f"Completed illustration {i+1}/{len(paragraphs)}",
                                transcript_text=enhanced_text
                            )
                        
                    except Exception as e:
                        print(f"⚠️ Failed to generate illustration for paragraph {i}: {e}")
                        # Update status even on failure
                        await update_processing_status(
                            story_id, 
                            ProcessingStep.ILLUSTRATING, 
                            progress + 1, 
                            f"Failed illustration {i+1}/{len(paragraphs)}, continuing...",
                            transcript_text=enhanced_text
                        )
                
                # Final illustration status update
                await update_processing_status(
                    story_id, 
                    ProcessingStep.ILLUSTRATING, 
                    98, 
                    f"✅ Generated {illustration_count} illustrations successfully!",
                    transcript_text=enhanced_text
                )
                
                print(f"✅ Generated {illustration_count} illustrations")
                
                # Small delay before moving to final step
                await asyncio.sleep(1)
                
            except Exception as illustration_error:
                print(f"⚠️ Illustration generation failed: {str(illustration_error)}")
                # Continue processing even if illustrations fail
                await update_processing_status(
                    story_id, 
                    ProcessingStep.ILLUSTRATING, 
                    98, 
                    "Illustration generation failed, continuing with story...",
                    transcript_text=enhanced_text
                )
            
            # Step 7: Update story status to published
            story.status = StoryStatus.PUBLISHED
            
            # Step 8: Update analytics
            try:
                analytics_result = await db_session.execute(select(Analytics).where(Analytics.story_id == story.id))
                analytics = analytics_result.scalar_one_or_none()
                
                if analytics:
                    # Update existing analytics with AI insights
                    if content_analysis and content_analysis.get("sentiment"):
                        analytics.sentiment_score = content_analysis["sentiment"].get("confidence", 0.5)
                    if content_analysis and content_analysis.get("entities"):
                        analytics.entities = content_analysis["entities"]
                    if content_analysis and content_analysis.get("keywords"):
                        analytics.keywords = content_analysis["keywords"]
                    if content_analysis and content_analysis.get("cultural_context"):
                        analytics.cultural_analysis = content_analysis["cultural_context"]
                    print(f"✅ Analytics updated")
            except Exception as analytics_error:
                print(f"⚠️ Analytics update failed: {str(analytics_error)}")
            
            try:
                await db_session.commit()
                print(f"✅ All data committed to database")
            except Exception as commit_error:
                error_msg = f"Database commit failed: {str(commit_error)}"
                print(f"❌ {error_msg}")
                await update_processing_status(story_id, ProcessingStep.FAILED, 0, "Database commit failed", error=error_msg)
                return
            
            # Final status update
            await update_processing_status(
                story_id, 
                ProcessingStep.PUBLISHED, 
                100, 
                "Story published successfully! ✨",
                transcript_text=enhanced_text
            )
            
            print(f"🎉 Story {story_id} processing completed successfully!")
        
        except Exception as e:
            error_msg = f"Story processing failed with unexpected error: {str(e)}"
            print(f"❌ {error_msg}")
            print(f"❌ Exception type: {type(e).__name__}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            
            await update_processing_status(
                story_id, 
                ProcessingStep.FAILED, 
                0, 
                "Story processing failed",
                error=error_msg
            )
            
            # Update story status to reflect error
            try:
                result = await db_session.execute(select(Story).where(Story.id == story_id))
                story = result.scalar_one_or_none()
                if story:
                    story.status = StoryStatus.REJECTED
                    await db_session.commit()
                    print(f"📝 Story status updated to REJECTED")
            except Exception as status_update_error:
                print(f"❌ Failed to update story status: {str(status_update_error)}")
        
        finally:
            # Clean up downloaded audio file
            if audio_file_path and os.path.exists(audio_file_path):
                try:
                    os.unlink(audio_file_path)
                    print(f"🧹 Cleaned up temporary audio file: {audio_file_path}")
                except Exception as cleanup_error:
                    print(f"⚠️ Failed to cleanup audio file: {str(cleanup_error)}")


async def download_audio_file(audio_url: str) -> str:
    """Download audio file from URL to temporary location with better error handling"""
    try:
        print(f"🔄 Attempting to download from URL: {audio_url}")
        
        # Handle local file URLs
        if audio_url.startswith('/media/files/'):
            # This is a local file path
            local_file_path = f"media/files{audio_url.replace('/media/files', '')}"
            print(f"🔄 Local file path: {local_file_path}")
            
            if os.path.exists(local_file_path):
                # Copy to temporary location
                import shutil
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                    shutil.copy2(local_file_path, temp_file.name)
                    print(f"✅ Local file copied to: {temp_file.name}")
                    return temp_file.name
            else:
                raise FileNotFoundError(f"Local file not found: {local_file_path}")
        
        # Handle full URLs
        elif audio_url.startswith(('http://', 'https://')):
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(audio_url)
                response.raise_for_status()
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                    temp_file.write(response.content)
                    print(f"✅ Remote file downloaded to: {temp_file.name}")
                    return temp_file.name
        
        else:
            raise ValueError(f"Unsupported audio URL format: {audio_url}")
                
    except Exception as e:
        error_msg = f"Failed to download audio file from {audio_url}: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
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


@router.get("/story/{story_id}/processing-status")
async def get_story_processing_status(
    story_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get real-time processing status for a story"""
    
    # Check if we have status in memory
    if story_id in PROCESSING_STATUS_STORE:
        return PROCESSING_STATUS_STORE[story_id]
    
    # Check database status
    result = await db.execute(select(Story).where(Story.id == story_id))
    story = result.scalar_one_or_none()
    
    if not story:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Story not found"
        )
    
    # Determine status from story state
    if story.status == StoryStatus.PROCESSING:
        # Check if transcript exists
        transcript_result = await db.execute(
            select(Transcript).where(Transcript.story_id == story_id)
        )
        transcript = transcript_result.scalar_one_or_none()
        
        if transcript:
            step = ProcessingStep.PUBLISHED
            progress = 100
            message = "Story processing completed"
        else:
            step = ProcessingStep.TRANSCRIBING
            progress = 50
            message = "Processing story with AI transcription..."
    elif story.status == StoryStatus.PUBLISHED:
        step = ProcessingStep.PUBLISHED
        progress = 100
        message = "Story published successfully"
    else:
        step = ProcessingStep.FAILED
        progress = 0
        message = "Story processing failed"
    
    status_obj = StoryProcessingStatus(
        story_id=story_id,
        current_step=step,
        progress_percentage=progress,
        message=message
    )
    
    return status_obj


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


@router.get("/debug/story/{story_id}")
async def debug_story_processing(
    story_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Debug endpoint to check story processing status and details"""
    
    try:
        # Get story from database
        result = await db.execute(select(Story).where(Story.id == story_id))
        story = result.scalar_one_or_none()
        
        if not story:
            return {"error": "Story not found", "story_id": story_id}
        
        # Check if user has access
        if story.contributor_id != current_user.id:
            return {"error": "Access denied", "story_id": story_id}
        
        # Get transcript if exists
        transcript_result = await db.execute(select(Transcript).where(Transcript.story_id == story_id))
        transcript = transcript_result.scalar_one_or_none()
        
        # Check processing status
        processing_status = PROCESSING_STATUS_STORE.get(story_id)
        
        # Check system configuration
        system_config = {
            "gemini_client": genai_client is not None,
            "gemini_api_key_set": bool(settings.GEMINI_API_KEY),
            "google_cloud_speech": speech_client is not None,
            "google_cloud_translate": translate_instance is not None,
            "audio_file_exists": bool(story.audio_file_url),
            "audio_file_url": story.audio_file_url
        }
        
        # Check if audio file is accessible
        audio_accessible = False
        audio_error = None
        if story.audio_file_url:
            try:
                if story.audio_file_url.startswith('/media/files/'):
                    # Local file
                    local_path = f"media/files{story.audio_file_url.replace('/media/files', '')}"
                    audio_accessible = os.path.exists(local_path)
                    if not audio_accessible:
                        audio_error = f"Local file not found: {local_path}"
                elif story.audio_file_url.startswith(('http://', 'https://')):
                    # Remote file - just check URL format
                    audio_accessible = True
                else:
                    audio_error = f"Invalid URL format: {story.audio_file_url}"
            except Exception as e:
                audio_error = str(e)
        
        return {
            "story": {
                "id": str(story.id),
                "title": story.title,
                "language": story.language,
                "status": story.status,
                "audio_file_url": story.audio_file_url,
                "created_at": story.created_at.isoformat()
            },
            "transcript": {
                "exists": transcript is not None,
                "language": transcript.language if transcript else None,
                "confidence": transcript.confidence_score if transcript else None,
                "has_text": bool(transcript and transcript.transcript_json.get("original_text")) if transcript else False
            },
            "processing_status": {
                "in_memory": processing_status.dict() if processing_status else None,
                "current_step": processing_status.current_step if processing_status else None
            },
            "system_config": system_config,
            "audio_file": {
                "accessible": audio_accessible,
                "error": audio_error
            }
        }
        
    except Exception as e:
        import traceback
        return {
            "error": f"Debug failed: {str(e)}",
            "traceback": traceback.format_exc()
        }


@router.get("/debug/system")
async def debug_system_status(current_user: User = Depends(get_current_active_user)):
    """Debug endpoint to check overall system configuration"""
    
    return {
        "ai_services": {
            "gemini_client": genai_client is not None,
            "gemini_tts_client": genai_tts_client is not None,
            "google_cloud_speech": speech_client is not None,
            "google_cloud_translate": translate_instance is not None,
            "spacy_nlp": nlp is not None,
            "sentiment_analyzer": sentiment_analyzer is not None
        },
        "configuration": {
            "gemini_api_key_set": bool(settings.GEMINI_API_KEY),
            "google_credentials_set": bool(settings.GOOGLE_APPLICATION_CREDENTIALS),
            "supported_languages": settings.SUPPORTED_LANGUAGES,
            "translation_targets": settings.TRANSLATION_TARGET_LANGUAGES,
            "tts_model": settings.TTS_MODEL,
            "tts_default_voice": settings.TTS_DEFAULT_VOICE,
            "swahili_voices": settings.SWAHILI_VOICES
        },
        "processing_status_store": {
            "active_stories": len(PROCESSING_STATUS_STORE),
            "story_ids": list(PROCESSING_STATUS_STORE.keys())
        }
    } 


@router.post("/debug/test-translation")
async def test_translation(
    text: str = Form(...),
    target_language: str = Form(...),
    source_language: str = Form(default="auto"),
    current_user: User = Depends(get_current_active_user)
):
    """Test translation service for debugging"""
    
    try:
        result = await TranslationProcessor.translate_text(text, target_language, source_language)
        return {
            "success": True,
            "result": result,
            "services_available": {
                "google_translate": translate_instance is not None,
                "gemini": genai_client is not None
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "services_available": {
                "google_translate": translate_instance is not None,
                "gemini": genai_client is not None
            }
    } 


@router.get("/debug/transcript-status")
async def debug_transcript_status(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Debug endpoint to check transcript data status"""
    try:
        # Get transcript statistics
        result = await db.execute(text("""
            SELECT 
                COUNT(*) as total_transcripts,
                COUNT(CASE WHEN transcript_json->>'words' IS NOT NULL THEN 1 END) as with_words,
                COUNT(CASE WHEN jsonb_array_length(transcript_json->'words') > 0 THEN 1 END) as with_valid_words
            FROM transcripts
        """))
        stats = result.fetchone()
        
        # Get sample transcript data
        sample_result = await db.execute(text("""
            SELECT id, transcript_json->>'original_text' as text_preview, 
                   jsonb_array_length(transcript_json->'words') as word_count
            FROM transcripts 
            LIMIT 3
        """))
        samples = sample_result.fetchall()
        
        return {
            "status": "success",
            "statistics": {
                "total_transcripts": stats[0],
                "transcripts_with_words_field": stats[1],
                "transcripts_with_valid_words": stats[2]
            },
            "sample_data": [
                {
                    "id": str(sample[0]),
                    "text_preview": sample[1][:100] + "..." if sample[1] and len(sample[1]) > 100 else sample[1],
                    "word_count": sample[2]
                }
                for sample in samples
            ]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get transcript status: {str(e)}"
        ) 


@router.post("/debug/test-illustration")
async def test_illustration_generation(
    text: str = Form(...),
    language: str = Form(default="sw-KE"),
    current_user: User = Depends(get_current_active_user)
):
    """Test endpoint for illustration generation"""
    try:
        print(f"🧪 Testing illustration generation for text: {text[:100]}...")
        
        # Create illustration prompt
        prompt = IllustrationGenerator.create_illustration_prompt(
            text,
            language,
            {
                "origin": "Test",
                "storyteller_name": "Test Storyteller",
                "title": "Test Story"
            }
        )
        
        print(f"📝 Generated prompt: {prompt[:200]}...")
        
        # Generate illustration
        result = await IllustrationGenerator.generate_image_with_gemini(prompt)
        
        if result:
            return {
                "success": True,
                "image_url": result["image_url"],
                "prompt_used": result["prompt_used"],
                "style_description": result["style_description"],
                "generation_metadata": result["generation_metadata"]
            }
        else:
            return {
                "success": False,
                "error": "Illustration generation failed"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        } 