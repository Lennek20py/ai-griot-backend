#!/usr/bin/env python3
"""
Create a placeholder image for illustration generation failures
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_placeholder_image():
    """Create a placeholder image for failed illustrations"""
    
    # Create the media directory if it doesn't exist
    os.makedirs("media/illustrations", exist_ok=True)
    
    # Create a simple placeholder image
    width, height = 400, 300
    image = Image.new('RGB', (width, height), color='#f8fafc')
    draw = ImageDraw.Draw(image)
    
    # Add a border
    draw.rectangle([0, 0, width-1, height-1], outline='#e2e8f0', width=2)
    
    # Add a decorative circle
    circle_center = (width // 2, height // 2 - 20)
    circle_radius = 40
    draw.ellipse([
        circle_center[0] - circle_radius,
        circle_center[1] - circle_radius,
        circle_center[0] + circle_radius,
        circle_center[1] + circle_radius
    ], fill='#3b82f6', outline='#1d4ed8', width=2)
    
    # Add text
    try:
        # Try to use a default font
        font = ImageFont.load_default()
    except:
        font = None
    
    text_lines = [
        "Story Illustration",
        "Generating..."
    ]
    
    for i, text in enumerate(text_lines):
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = (width - text_width) // 2
        text_y = height - 80 + (i * 25)
        
        color = '#1f2937' if i == 0 else '#6b7280'
        draw.text((text_x, text_y), text, fill=color, font=font)
    
    # Save the placeholder image
    placeholder_path = "media/placeholder.png"
    image.save(placeholder_path, 'PNG')
    
    print(f"✅ Created placeholder image at {placeholder_path}")

if __name__ == "__main__":
    create_placeholder_image() 