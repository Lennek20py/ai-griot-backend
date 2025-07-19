#!/usr/bin/env python3
"""
Swahili TTS Test Script for Digital Griot

This script demonstrates the Gemini-powered text-to-speech functionality
with a focus on Swahili language content.
"""

import os
import requests
import json
from pathlib import Path


def test_swahili_narration():
    """Test Swahili story narration with traditional storytelling elements"""
    
    # Sample Swahili story text
    swahili_story = """
    Hadithi, hadithi? Hadithi njoo!
    
    Palikuwa na mfalme mkuu wa Afrika Mashariki. Jina lake lilikuwa Mfalme Simba. 
    Alikuwa na moyo mkuu na hekima nyingi. Watu wa kijiji chake walimpenda sana.
    
    Siku moja, mfalme aliamua kutembea msituni kuona wanyamapori wake. 
    Alipotemebea, alikutana na kobe mkongwe aliyekuwa akilia.
    
    "Kwa nini unalia, shangazi?" alimuuliza mfalme kwa huruma.
    
    Kobe akajibu: "Mfalme wangu, nina miaka mingi sana, lakini bado sijapata mzuri wa kweli."
    
    Mfalme Simba akasema: "Shangazi, hekima ni utajiri mkuu kuliko dhahabu na fedha. 
    Wewe una hekima ya miaka mingi. Hiyo ndiyo mali yako ya thamani."
    
    Kobe akafurahi na kuwaambia wanyamapori wengine. Kuanzia siku hiyo, 
    wanyamapori wote waliheshimu kobe kwa hekima yake.
    
    Hadithi inaishia hapa. Funzo: Hekima ni mali ya thamani kuliko utajiri wa kimaduni.
    """
    
    print("🎙️ Testing Swahili TTS with Traditional Story")
    print("=" * 50)
    
    # API endpoint (assuming server is running locally)
    base_url = "http://localhost:8000/api/v1/ai"
    
    # Test data
    test_data = {
        "text": swahili_story,
        "language": "sw",
        "voice_name": "Zephyr",  # Good for Swahili
        "multi_speaker": False,
        "return_file": True
    }
    
    try:
        print("📤 Sending request to generate Swahili speech...")
        
        # Make request to TTS endpoint
        response = requests.post(
            f"{base_url}/generate-speech",
            data=test_data,
            headers={"Authorization": "Bearer YOUR_JWT_TOKEN"}  # Replace with actual token
        )
        
        if response.status_code == 200:
            # Save generated audio
            output_file = "swahili_story_narration.wav"
            with open(output_file, "wb") as f:
                f.write(response.content)
            
            print(f"✅ Success! Audio saved as: {output_file}")
            print(f"📁 File size: {len(response.content)} bytes")
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection error. Make sure the Digital Griot server is running on localhost:8000")
    except Exception as e:
        print(f"❌ Error: {e}")


def test_multi_speaker_dialog():
    """Test multi-speaker dialog in Swahili"""
    
    # Multi-speaker Swahili dialog
    dialog_text = """
    Speaker 1: Hujambo rafiki yangu! Habari za asubuhi?
    Speaker 2: Sijambo, asante sana! Nimefurahi kukuona. Habari za nyumbani?
    Speaker 1: Nyumbani ni sawa. Leo nina hadithi nzuri ya kuwaambia.
    Speaker 2: Hebu niambie! Napenda sana hadithi za zamani.
    Speaker 1: Hadithi hii ni kuhusu mfalme aliyejifunza hekima kutoka kwa kobe mkongwe.
    Speaker 2: Hiyo ni hadithi ya kuvutia! Endelea, tafadhali.
    """
    
    print("\n🗣️ Testing Multi-Speaker Swahili Dialog")
    print("=" * 50)
    
    base_url = "http://localhost:8000/api/v1/ai"
    
    test_data = {
        "text": dialog_text,
        "language": "sw", 
        "multi_speaker": True,
        "return_file": True
    }
    
    try:
        print("📤 Generating multi-speaker dialog...")
        
        response = requests.post(
            f"{base_url}/generate-speech",
            data=test_data,
            headers={"Authorization": "Bearer YOUR_JWT_TOKEN"}
        )
        
        if response.status_code == 200:
            output_file = "swahili_dialog.wav"
            with open(output_file, "wb") as f:
                f.write(response.content)
            
            print(f"✅ Dialog audio saved as: {output_file}")
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def test_voice_options():
    """Test different voice options for Swahili"""
    
    print("\n🎵 Testing Different Voices for Swahili")
    print("=" * 50)
    
    base_url = "http://localhost:8000/api/v1/ai"
    
    # Get available voices
    try:
        response = requests.get(f"{base_url}/voices")
        if response.status_code == 200:
            voices_info = response.json()
            print("Available voices:", voices_info["available_voices"])
            print("Swahili recommended:", voices_info["swahili_recommended"])
        
    except Exception as e:
        print(f"❌ Error getting voice info: {e}")
    
    # Test short phrase with different voices
    test_phrase = "Hongera! Umefanikiwa kutumia teknolojia ya AI kutunza hadithi za kitamaduni."
    
    voices_to_test = ["Zephyr", "Kore"]  # Good for Swahili
    
    for voice in voices_to_test:
        print(f"\n🎤 Testing voice: {voice}")
        
        test_data = {
            "text": test_phrase,
            "language": "sw",
            "voice_name": voice,
            "return_file": True
        }
        
        try:
            response = requests.post(
                f"{base_url}/generate-speech",
                data=test_data,
                headers={"Authorization": "Bearer YOUR_JWT_TOKEN"}
            )
            
            if response.status_code == 200:
                output_file = f"swahili_test_{voice.lower()}.wav"
                with open(output_file, "wb") as f:
                    f.write(response.content)
                print(f"   ✅ Generated: {output_file}")
            else:
                print(f"   ❌ Failed for {voice}")
                
        except Exception as e:
            print(f"   ❌ Error with {voice}: {e}")


def test_story_narration_endpoint():
    """Test the story narration endpoint (requires existing story)"""
    
    print("\n📖 Testing Story Narration Endpoint")
    print("=" * 50)
    
    # This would require an actual story ID from the database
    print("ℹ️  To test story narration:")
    print("1. Upload a story through the regular upload process")
    print("2. Get the story ID from the response")
    print("3. Use the /generate-story-narration endpoint")
    print("")
    print("Example request:")
    print("""
    curl -X POST "http://localhost:8000/api/v1/ai/generate-story-narration" \\
        -H "Authorization: Bearer YOUR_JWT_TOKEN" \\
        -F "story_id=YOUR_STORY_ID" \\
        -F "language=sw" \\
        -F "voice_name=Zephyr" \\
        -F "use_translation=false" \\
        --output story_narration.wav
    """)


def main():
    """Run all TTS tests"""
    
    print("""
    🎭 Digital Griot - Swahili TTS Testing Suite
    ============================================
    
    This script tests the Gemini-powered text-to-speech functionality
    with Swahili language content and traditional storytelling elements.
    
    Make sure:
    1. Digital Griot server is running (python -m uvicorn app.main:app --reload)
    2. You have valid authentication tokens
    3. Gemini TTS is properly configured
    
    """)
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running!")
        else:
            print("⚠️  Server responded but may have issues")
    except:
        print("❌ Server not accessible. Please start the server first.")
        print("   Run: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
        return
    
    # Run tests
    test_swahili_narration()
    test_multi_speaker_dialog()
    test_voice_options()
    test_story_narration_endpoint()
    
    print("\n🎉 Testing completed!")
    print("\nGenerated files:")
    for file in Path(".").glob("swahili_*.wav"):
        print(f"  📁 {file.name}")
    
    print("\n💡 Tips for best results:")
    print("- Use 'Zephyr' or 'Kore' voices for Swahili content")
    print("- Include traditional storytelling elements (Hadithi, hadithi?)")
    print("- Test with different text lengths to find optimal results")
    print("- Consider cultural context in your story content")


if __name__ == "__main__":
    main() 