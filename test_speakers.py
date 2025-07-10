#!/usr/bin/env python3
"""
Test script for speaker diarization functionality
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from speaker_diarization import SpeakerDiarizationService, InteractiveSpeakerNaming

def test_speaker_diarization_service():
    """Test the speaker diarization service"""
    logging.basicConfig(level=logging.INFO)
    
    print("=== Testing Speaker Diarization Service ===")
    
    try:
        # Test initialization
        service = SpeakerDiarizationService()
        print("✅ SpeakerDiarizationService initialized successfully")
        
        # Test service info
        info = service.get_service_info()
        print(f"Service info: {info}")
        
        # Test model loading (requires HuggingFace token)
        print("\n=== Testing Model Loading ===")
        if os.getenv("HUGGINGFACE_TOKEN"):
            success = service.load_pipeline()
            if success:
                print("✅ Pipeline loaded successfully")
            else:
                print("❌ Pipeline loading failed")
        else:
            print("⚠️  HUGGINGFACE_TOKEN not set - skipping pipeline loading test")
            
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Install with: pip install pyannote-audio")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_interactive_naming():
    """Test the interactive speaker naming workflow"""
    print("\n=== Testing Interactive Speaker Naming ===")
    
    # Mock transcript segments
    mock_transcript = [
        {"start": 0.0, "end": 5.0, "text": "Hello everyone, welcome to our meeting"},
        {"start": 5.5, "end": 10.0, "text": "Thanks for organizing this, I'm excited to discuss the project"},
        {"start": 10.5, "end": 15.0, "text": "Let's start with the quarterly updates"},
        {"start": 15.5, "end": 20.0, "text": "I agree, we should focus on the main objectives"}
    ]
    
    # Mock speaker segments
    from speaker_diarization import SpeakerSegment
    mock_speakers = [
        SpeakerSegment(0.0, 5.0, "SPEAKER_00", 0.9),
        SpeakerSegment(5.5, 10.0, "SPEAKER_01", 0.8),
        SpeakerSegment(10.5, 15.0, "SPEAKER_00", 0.9),
        SpeakerSegment(15.5, 20.0, "SPEAKER_02", 0.7)
    ]
    
    try:
        # Test interactive naming workflow
        naming = InteractiveSpeakerNaming(mock_transcript, mock_speakers)
        
        # Test merging transcript with speakers
        merged = naming.merge_transcript_with_speakers()
        print(f"✅ Merged segments: {len(merged)} segments")
        
        # Test getting sample quotes
        quotes = naming.get_speaker_sample_quotes(merged)
        print(f"✅ Sample quotes: {quotes}")
        
        # Test speaker naming (in non-interactive mode)
        speaker_names = naming.prompt_for_speaker_names(quotes, skip_interactive=True)
        print(f"✅ Speaker names: {speaker_names}")
        
        # Test applying names
        updated_segments = naming.apply_speaker_names(merged, speaker_names)
        print(f"✅ Updated segments with names: {len(updated_segments)} segments")
        
        # Print sample output
        print("\n=== Sample Output ===")
        for segment in updated_segments[:3]:
            start_time = f"{int(segment.get('start', 0)//60):02d}:{int(segment.get('start', 0)%60):02d}"
            speaker = segment.get('speaker', 'Unknown')
            text = segment.get('text', '').strip()
            print(f"[{start_time}] {speaker}: {text}")
        
        return True
        
    except Exception as e:
        print(f"❌ Interactive naming test failed: {e}")
        return False

def test_full_workflow():
    """Test the complete workflow with a sample audio file"""
    print("\n=== Testing Full Workflow ===")
    
    # This would require an actual audio file and HuggingFace token
    audio_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    if not audio_file:
        print("⚠️  No audio file provided - skipping full workflow test")
        print("Usage: python test_speakers.py /path/to/audio.mp3")
        return True
    
    if not Path(audio_file).exists():
        print(f"❌ Audio file not found: {audio_file}")
        return False
    
    if not os.getenv("HUGGINGFACE_TOKEN"):
        print("⚠️  HUGGINGFACE_TOKEN not set - skipping full workflow test")
        return True
    
    try:
        service = SpeakerDiarizationService()
        result = service.perform_diarization(audio_file)
        
        if result.speakers:
            print(f"✅ Full workflow test successful:")
            print(f"  - Speakers found: {len(result.speakers)}")
            print(f"  - Segments: {len(result.segments)}")
            print(f"  - Processing time: {result.processing_time:.2f}s")
            
            # Test interactive naming with real results
            mock_transcript = [
                {"start": seg.start_time, "end": seg.end_time, "text": f"Sample text {i}"}
                for i, seg in enumerate(result.segments[:5])
            ]
            
            naming = InteractiveSpeakerNaming(mock_transcript, result.segments)
            merged = naming.merge_transcript_with_speakers()
            quotes = naming.get_speaker_sample_quotes(merged)
            print(f"  - Sample quotes: {quotes}")
            
            return True
        else:
            print("❌ No speakers detected in audio file")
            return False
            
    except Exception as e:
        print(f"❌ Full workflow test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Speaker Diarization Test Suite")
    print("=" * 50)
    
    tests = [
        ("Service Initialization", test_speaker_diarization_service),
        ("Interactive Naming", test_interactive_naming),
        ("Full Workflow", test_full_workflow)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the output above.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)