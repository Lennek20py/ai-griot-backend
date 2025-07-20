#!/usr/bin/env python3
"""
Demo script for the enhanced illustration system
"""

import asyncio
import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.tasks.illustration import IllustrationGenerator, ParagraphProcessor

async def demo_illustration_system():
    """Demo the enhanced illustration system"""
    print("🎨 Digital Griot - Enhanced Illustration System Demo")
    print("=" * 60)
    
    # Initialize the generator
    generator = IllustrationGenerator()
    
    # Sample stories for demonstration
    sample_stories = [
        {
            "text": "Once upon a time, in a small village nestled between rolling hills, there lived a wise grandmother who knew the secrets of the forest. She would often tell stories to the children of the village, stories that had been passed down from generation to generation.",
            "language": "sw-KE",
            "metadata": {
                "origin": "Kenya",
                "storyteller_name": "Grandmother",
                "title": "The Whispering Trees"
            }
        },
        {
            "text": "In the heart of Tanzania, where the savanna meets the mountains, there was a young boy named Juma who dreamed of becoming a great storyteller like his grandfather. Every evening, he would sit by the fire and listen to tales of bravery and wisdom.",
            "language": "sw-TZ",
            "metadata": {
                "origin": "Tanzania",
                "storyteller_name": "Grandfather",
                "title": "Juma's Dream"
            }
        },
        {
            "text": "Long ago, in a village by the sea, there lived a fisherman who could understand the language of the dolphins. They would guide him to the best fishing spots and warn him of approaching storms.",
            "language": "en-US",
            "metadata": {
                "origin": "Coastal Kenya",
                "storyteller_name": "Fisherman",
                "title": "The Dolphin's Gift"
            }
        }
    ]
    
    print(f"\n📚 Processing {len(sample_stories)} sample stories...")
    
    for i, story in enumerate(sample_stories, 1):
        print(f"\n📖 Story {i}: {story['metadata']['title']}")
        print(f"   Language: {story['language']}")
        print(f"   Origin: {story['metadata']['origin']}")
        
        # Split into paragraphs
        paragraphs = ParagraphProcessor.split_into_paragraphs(story['text'])
        print(f"   Paragraphs: {len(paragraphs)}")
        
        # Generate illustrations for each paragraph
        for j, paragraph in enumerate(paragraphs, 1):
            print(f"\n   🎨 Generating illustration {j}/{len(paragraphs)}...")
            
            # Create prompt
            prompt = IllustrationGenerator.create_illustration_prompt(
                paragraph['text'],
                story['language'],
                story['metadata']
            )
            
            print(f"   📝 Prompt: {prompt[:100]}...")
            
            # Generate image
            result = await generator.generate_image_with_gemini(prompt)
            
            if result:
                print(f"   ✅ Generated: {result['image_url']}")
                print(f"   🎨 Style: {result['style_description']}")
                print(f"   ⏱️  Time: {result['generation_time']}s")
                print(f"   🤖 Model: {result['model_used']}")
            else:
                print(f"   ❌ Failed to generate illustration")
    
    print(f"\n🎉 Demo completed!")
    print(f"📁 Check the 'media/illustrations/' directory for generated images")
    print(f"🌐 Images are accessible via: http://localhost:8000/media/illustrations/")

if __name__ == "__main__":
    asyncio.run(demo_illustration_system()) 