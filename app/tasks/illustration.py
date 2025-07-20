import os
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
from pathlib import Path

# Google GenAI imports
from google import genai
from PIL import Image
from io import BytesIO

# Local imports
from app.core.config import settings
from app.models.story import Paragraph, Illustration
from app.core.database import AsyncSessionLocal
from app.models.story import Story
from sqlalchemy import select
from app.core.celery import celery_app

logger = logging.getLogger(__name__)

class ParagraphProcessor:
    """Handles paragraph splitting and processing"""
    
    @staticmethod
    def split_into_paragraphs(transcript_text: str, words: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Split transcript text into paragraphs with timing information
        """
        # Simple paragraph splitting based on double newlines or sentence boundaries
        paragraphs = []
        
        # Split by double newlines first
        raw_paragraphs = transcript_text.split('\n\n')
        
        for i, paragraph_text in enumerate(raw_paragraphs):
            paragraph_text = paragraph_text.strip()
            if not paragraph_text:
                continue
                
            # Calculate timing if words are provided
            start_time = 0.0
            end_time = 0.0
            word_count = len(paragraph_text.split())
            
            if words:
                # Find words that belong to this paragraph
                paragraph_words = []
                current_text = ""
                
                for word in words:
                    current_text += word.get('word', '') + ' '
                    if paragraph_text.lower() in current_text.lower():
                        paragraph_words.append(word)
                        break
                
                if paragraph_words:
                    start_time = paragraph_words[0].get('start_time', 0.0)
                    end_time = paragraph_words[-1].get('end_time', 0.0)
            
            paragraphs.append({
                'text': paragraph_text,
                'start_time': start_time,
                'end_time': end_time,
                'word_count': word_count,
                'sequence_order': i
            })
        
        return paragraphs

class IllustrationGenerator:
    """Handles AI-generated illustration creation using Google GenAI Imagen"""
    
    def __init__(self):
        self.client = None
        self._initialized = False
    
    def _initialize_client(self):
        """Initialize the Google GenAI client"""
        if self._initialized:
            return
            
        try:
            api_key = settings.GEMINI_API_KEY
            logger.info(f"🔍 Checking API key: {'Set' if api_key else 'Not set'}")
            
            if not api_key:
                logger.warning("GEMINI_API_KEY not found in settings")
                self._initialized = True
                return
            
            logger.info("🔧 Initializing Google GenAI client...")
            self.client = genai.Client(api_key=api_key)
            logger.info("✅ Google GenAI client initialized successfully")
            self._initialized = True
        except Exception as e:
            logger.error(f"❌ Failed to initialize Google GenAI client: {e}")
            self.client = None
            self._initialized = True
    
    def _check_available_models(self):
        """Check what models are available for image generation"""
        # Ensure client is initialized
        self._initialize_client()
        
        if not self.client:
            logger.error("❌ Google GenAI client not initialized")
            return None
            
        try:
            models = self.client.models.list()
            available_models = [m.name for m in models]
            logger.info(f"🔍 Available models: {available_models}")
            
            # Look for image generation models
            image_models = [m for m in available_models if 'imagen' in m.lower() or 'image' in m.lower()]
            if image_models:
                logger.info(f"🎨 Found image generation models: {image_models}")
                return image_models[0]  # Return the first available image model
            else:
                logger.warning("⚠️ No image generation models found")
                return None
        except Exception as e:
            logger.warning(f"⚠️ Could not list models: {e}")
            return None
    
    @staticmethod
    def create_illustration_prompt(paragraph_text: str, language: str, metadata: Dict[str, Any]) -> str:
        """
        Create a culturally appropriate illustration prompt
        """
        # Clean and prepare the text
        clean_text = paragraph_text.strip()
        if len(clean_text) > 500:
            clean_text = clean_text[:500] + "..."
        
        # Extract key elements for the prompt
        origin = metadata.get('origin', '')
        storyteller_name = metadata.get('storyteller_name', '')
        title = metadata.get('title', '')
        
        # Create culturally sensitive prompt
        prompt_parts = []
        
        # Style guidance based on language/culture
        if language.startswith('sw'):  # Swahili
            prompt_parts.append("African traditional storytelling style")
            prompt_parts.append("warm earth tones")
            prompt_parts.append("traditional African village setting")
        elif language.startswith('en'):  # English
            prompt_parts.append("universal storytelling style")
            prompt_parts.append("vibrant colors")
        else:
            prompt_parts.append("cultural storytelling style")
            prompt_parts.append("rich colors")
        
        # Add story context
        if origin:
            prompt_parts.append(f"set in {origin}")
        
        # Create the final prompt
        base_prompt = f"Create a beautiful illustration for a traditional story: {clean_text}"
        
        if prompt_parts:
            style_description = ", ".join(prompt_parts)
            final_prompt = f"{base_prompt}. Style: {style_description}. Create a warm, engaging illustration that captures the essence of this traditional story."
        else:
            final_prompt = f"{base_prompt}. Create a warm, engaging illustration that captures the essence of this traditional story."
        
        return final_prompt
    
    async def generate_image_with_imagen(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Generate image using Google GenAI Imagen model
        """
        # Ensure client is initialized
        self._initialize_client()
        
        if not self.client:
            logger.error("❌ Google GenAI client not initialized")
            return None
        
        try:
            logger.info(f"🎨 Generating image with Imagen: {prompt[:100]}...")
            
            # Use the correct Google GenAI API for image generation
            result = self.client.models.generate_image(
                model="models/imagen-4.0-generate-preview-06-06",
                prompt=prompt,
                config=dict(
                    number_of_images=1,
                    output_mime_type="image/jpeg",
                    person_generation="ALLOW_ADULT",
                    aspect_ratio="16:9",  # Better for story illustrations
                ),
            )
            
            if not result.generated_images:
                logger.warning("⚠️ No images generated by Imagen")
                return None
            
            if len(result.generated_images) != 1:
                logger.warning(f"⚠️ Expected 1 image, got {len(result.generated_images)}")
                return None
            
            # Get the generated image using the correct structure
            for generated_image in result.generated_images:
                image_bytes = generated_image.image.image_bytes
                
                # Save the image
                image_filename = f"illustration_{uuid.uuid4()}.jpg"
                media_dir = Path(settings.MEDIA_ROOT) / "illustrations"
                media_dir.mkdir(parents=True, exist_ok=True)
                image_path = media_dir / image_filename
                
                # Save image using PIL
                image = Image.open(BytesIO(image_bytes))
                image.save(image_path, "JPEG", quality=95)
                
                # Create image URL
                image_url = f"/media/illustrations/{image_filename}"
                
                logger.info(f"✅ Generated image saved: {image_path}")
                
                return {
                    "image_url": image_url,
                    "prompt_used": prompt,
                    "style_description": "Imagen AI-generated illustration",
                    "generation_time": 5.0,
                    "model_used": "imagen-4.0-generate-preview-06-06"
                }
            
        except Exception as e:
            error_msg = str(e)
            if "billed users" in error_msg.lower():
                logger.warning("⚠️ Imagen API requires a billing account. Using enhanced fallback images.")
                logger.info("💡 To enable AI image generation, set up billing in Google Cloud Console:")
                logger.info("   1. Go to https://console.cloud.google.com/billing")
                logger.info("   2. Create or link a billing account")
                logger.info("   3. Enable the Imagen API for your project")
                logger.info("   4. The system will automatically use AI generation once billing is set up")
            else:
                logger.error(f"❌ Error generating image with Imagen: {e}")
            return None
    
    async def generate_fallback_image(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Generate a fallback image using Pillow when AI generation fails
        """
        try:
            logger.info("🎨 Creating enhanced fallback illustration...")
            
            # Create a more sophisticated placeholder image
            width, height = 800, 450  # 16:9 aspect ratio
            image = Image.new('RGB', (width, height), color='#f8f9fa')
            
            # Add some decorative elements
            from PIL import ImageDraw, ImageFont
            
            draw = ImageDraw.Draw(image)
            
            # Try to use a font, fallback to default if not available
            try:
                font_large = ImageFont.truetype("arial.ttf", 32)
                font_medium = ImageFont.truetype("arial.ttf", 18)
                font_small = ImageFont.truetype("arial.ttf", 14)
            except:
                font_large = ImageFont.load_default()
                font_medium = ImageFont.load_default()
                font_small = ImageFont.load_default()
            
            # Create a gradient-like background
            for y in range(height):
                # Create a subtle gradient from top to bottom
                color_value = int(248 - (y / height) * 20)  # Subtle gradient
                color = (color_value, color_value + 5, color_value + 10)
                draw.line([(0, y), (width, y)], fill=color)
            
            # Add decorative border with cultural elements
            border_color = '#2c5aa0'  # African blue
            draw.rectangle([10, 10, width-10, height-10], outline=border_color, width=4)
            
            # Add inner decorative border
            draw.rectangle([20, 20, width-20, height-20], outline='#e8c547', width=2)  # Gold accent
            
            # Add title with better positioning
            title = "Story Illustration"
            title_bbox = draw.textbbox((0, 0), title, font=font_large)
            title_width = title_bbox[2] - title_bbox[0]
            title_x = (width - title_width) // 2
            draw.text((title_x, 80), title, fill='#2c5aa0', font=font_large)
            
            # Add subtitle
            subtitle = "Traditional Story Visual"
            subtitle_bbox = draw.textbbox((0, 0), subtitle, font=font_medium)
            subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
            subtitle_x = (width - subtitle_width) // 2
            draw.text((subtitle_x, 120), subtitle, fill='#666666', font=font_medium)
            
            # Add decorative cultural elements (geometric patterns)
            # Top row of decorative elements
            for i in range(6):
                x = 100 + i * 120
                y = 200
                # Draw diamond pattern
                points = [(x, y-15), (x+15, y), (x, y+15), (x-15, y)]
                draw.polygon(points, fill='#e8c547', outline='#2c5aa0')
            
            # Bottom row of decorative elements
            for i in range(6):
                x = 100 + i * 120
                y = 300
                # Draw circle pattern
                draw.ellipse([x-12, y-12, x+12, y+12], fill='#2c5aa0', outline='#e8c547')
            
            # Add some text about the story
            story_text = "A tale of wisdom and tradition"
            text_bbox = draw.textbbox((0, 0), story_text, font=font_small)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = (width - text_width) // 2
            draw.text((text_x, 380), story_text, fill='#666666', font=font_small)
            
            # Add corner decorative elements
            corner_size = 30
            # Top-left corner
            draw.arc([10, 10, 10+corner_size, 10+corner_size], 0, 90, fill='#e8c547', width=3)
            # Top-right corner
            draw.arc([width-10-corner_size, 10, width-10, 10+corner_size], 90, 180, fill='#e8c547', width=3)
            # Bottom-left corner
            draw.arc([10, height-10-corner_size, 10+corner_size, height-10], 270, 360, fill='#e8c547', width=3)
            # Bottom-right corner
            draw.arc([width-10-corner_size, height-10-corner_size, width-10, height-10], 180, 270, fill='#e8c547', width=3)
            
            # Save the image
            image_filename = f"fallback_illustration_{uuid.uuid4()}.jpg"
            media_dir = Path(settings.MEDIA_ROOT) / "illustrations"
            media_dir.mkdir(parents=True, exist_ok=True)
            image_path = media_dir / image_filename
            
            image.save(image_path, "JPEG", quality=95)
            
            # Create image URL
            image_url = f"/media/illustrations/{image_filename}"
            
            logger.info(f"✅ Enhanced fallback image created: {image_path}")
            
            return {
                "image_url": image_url,
                "prompt_used": prompt,
                "style_description": "Enhanced cultural placeholder illustration",
                "generation_time": 2.0,
                "model_used": "fallback-enhanced-pillow"
            }
            
        except Exception as e:
            logger.error(f"❌ Error creating fallback image: {e}")
            return None
    
    async def generate_image_with_gemini(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Generate image using Google GenAI Imagen with fallback
        """
        # Try Imagen first
        result = await self.generate_image_with_imagen(prompt)
        
        if result:
            return result
        
        # Fallback to placeholder image
        logger.warning("⚠️ Imagen generation failed, using fallback")
        return await self.generate_fallback_image(prompt)

# Initialize the generator
illustration_generator = IllustrationGenerator()

@celery_app.task(bind=True, name="generate_illustrations")
async def generate_illustrations_task(self, story_id: str, paragraph_ids: List[str]):
    """Celery task for generating illustrations"""
    try:
        async with AsyncSessionLocal() as db_session:
            # Get story and paragraphs
            story = await db_session.get(Story, story_id)
            if not story:
                raise ValueError(f"Story {story_id} not found")
            
            paragraphs = await db_session.execute(
                select(Paragraph).where(Paragraph.id.in_(paragraph_ids))
            )
            paragraphs = paragraphs.scalars().all()
            
            # Generate illustrations for each paragraph
            for paragraph in paragraphs:
                try:
                    # Create prompt
                    prompt = IllustrationGenerator.create_illustration_prompt(
                        paragraph.content,
                        story.language,
                        {
                            "origin": story.origin,
                            "storyteller_name": story.storyteller_name,
                            "title": story.title
                        }
                    )
                    
                    # Generate image
                    result = await illustration_generator.generate_image_with_gemini(prompt)
                    
                    if result:
                        # Save illustration
                        illustration = Illustration(
                            paragraph_id=paragraph.id,
                            image_url=result["image_url"],
                            prompt_used=result["prompt_used"],
                            style=result["style_description"],
                            status="completed"
                        )
                        db_session.add(illustration)
                        
                except Exception as e:
                    print(f"⚠️ Failed to generate illustration for paragraph {paragraph.id}: {e}")
                    continue
            
            await db_session.commit()
            print(f"✅ Generated illustrations for story {story_id}")
            
    except Exception as e:
        print(f"❌ Illustration generation failed: {e}")
        raise

# For direct usage in API endpoints
async def generate_illustrations_for_story(story_id: str, paragraphs: List[Dict[str, Any]], db_session: AsyncSessionLocal):
    """Generate illustrations for a story's paragraphs"""
    try:
        for i, paragraph_data in enumerate(paragraphs):
            try:
                # Create prompt
                prompt = IllustrationGenerator.create_illustration_prompt(
                    paragraph_data["text"],
                    "sw-KE",  # Default language
                    {
                        "origin": "Kenya",
                        "storyteller_name": "Storyteller",
                        "title": "Traditional Story"
                    }
                )
                
                # Generate image
                result = await illustration_generator.generate_image_with_gemini(prompt)
                
                if result:
                    print(f"✅ Generated illustration {i+1}/{len(paragraphs)}")
                else:
                    print(f"⚠️ Failed to generate illustration {i+1}/{len(paragraphs)}")
                    
            except Exception as e:
                print(f"❌ Error generating illustration {i+1}: {e}")
                continue
                
    except Exception as e:
        print(f"❌ Illustration generation failed: {e}")
        raise 