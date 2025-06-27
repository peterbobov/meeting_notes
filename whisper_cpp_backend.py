#!/usr/bin/env python3
"""
whisper.cpp Backend Implementation
Optimized for M2 Mac using native whisper-cpp binary
"""

import os
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import time

from pydub import AudioSegment


@dataclass
class WhisperCppResult:
    """Result from whisper.cpp transcription"""
    text: str
    segments: List[Dict]
    language: str
    duration: float
    processing_time: float
    model_used: str
    confidence_scores: List[float]
    optimizations_applied: List[str]


class WhisperCppService:
    """
    whisper.cpp backend service optimized for M2 Mac
    Uses native whisper-cpp binary for superior performance
    """
    
    def __init__(self, model_size: str = "medium", models_dir: str = "./whisper_models"):
        """
        Initialize whisper.cpp service
        
        Args:
            model_size: Model size (tiny, base, small, medium, large, large-v2, large-v3)
            models_dir: Directory containing GGML model files
        """
        self.model_size = model_size
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Model mappings
        self.model_files = {
            "tiny": "ggml-tiny.bin",
            "tiny.en": "ggml-tiny.en.bin", 
            "base": "ggml-base.bin",
            "base.en": "ggml-base.en.bin",
            "small": "ggml-small.bin",
            "small.en": "ggml-small.en.bin",
            "medium": "ggml-medium.bin",
            "medium.en": "ggml-medium.en.bin",
            "large": "ggml-large.bin",
            "large-v1": "ggml-large-v1.bin",
            "large-v2": "ggml-large-v2.bin",
            "large-v3": "ggml-large-v3.bin"
        }
        
        # Model download URLs
        self.model_urls = {
            "tiny": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
            "base": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
            "small": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
            "medium": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
            "large": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
            "large-v2": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v2.bin",
            "large-v3": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin"
        }
        
        # Check whisper-cpp installation
        self.whisper_binary = self._find_whisper_binary()
        if not self.whisper_binary:
            raise RuntimeError("whisper-cpp not found. Install with: brew install whisper-cpp")
            
        logging.info(f"Initialized whisper.cpp service with model '{model_size}'")
        logging.info(f"Using binary: {self.whisper_binary}")
        
    def _find_whisper_binary(self) -> Optional[str]:
        """Find whisper-cpp binary"""
        possible_names = ["whisper-cli", "whisper-cpp", "whisper"]
        
        for name in possible_names:
            try:
                result = subprocess.run(["which", name], capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout.strip()
            except Exception:
                continue
        return None
    
    def ensure_model_available(self) -> bool:
        """Ensure the required model is available, download if needed"""
        model_file = self.model_files.get(self.model_size)
        if not model_file:
            logging.error(f"Unknown model size: {self.model_size}")
            return False
            
        model_path = self.models_dir / model_file
        
        if model_path.exists():
            logging.info(f"Model found: {model_path}")
            return True
            
        # Download model
        download_url = self.model_urls.get(self.model_size)
        if not download_url:
            logging.error(f"No download URL for model: {self.model_size}")
            return False
            
        logging.info(f"Downloading model {self.model_size} ({model_file})...")
        try:
            import urllib.request
            urllib.request.urlretrieve(download_url, model_path)
            logging.info(f"Model downloaded successfully: {model_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to download model: {e}")
            return False
    
    def preprocess_audio(self, audio_path: str) -> str:
        """
        Preprocess audio for optimal whisper.cpp performance
        
        Returns:
            Path to preprocessed audio file
        """
        try:
            # Load audio
            audio = AudioSegment.from_file(audio_path)
            
            # Convert to 16kHz mono WAV (whisper.cpp preferred format)
            audio = audio.set_frame_rate(16000).set_channels(1)
            
            # Export to temporary WAV file
            temp_dir = Path(tempfile.gettempdir()) / "plaud_processor"
            temp_dir.mkdir(exist_ok=True)
            
            temp_file = temp_dir / f"preprocessed_{Path(audio_path).stem}.wav"
            audio.export(str(temp_file), format="wav")
            
            logging.info(f"Audio preprocessed: {temp_file}")
            return str(temp_file)
            
        except Exception as e:
            logging.warning(f"Audio preprocessing failed: {e}. Using original file.")
            return audio_path
    
    def transcribe_audio(self, file_path: str, language: Optional[str] = None) -> WhisperCppResult:
        """
        Transcribe audio using whisper.cpp
        
        Args:
            file_path: Path to audio file
            language: Language code (auto-detect if None)
            
        Returns:
            WhisperCppResult with transcription data
        """
        start_time = time.time()
        
        try:
            # Ensure model is available
            if not self.ensure_model_available():
                raise Exception(f"Model {self.model_size} not available")
                
            # Preprocess audio
            processed_audio = self.preprocess_audio(file_path)
            
            # Prepare model path
            model_file = self.model_files[self.model_size]
            model_path = self.models_dir / model_file
            
            # Create temporary output directory
            with tempfile.TemporaryDirectory() as temp_dir:
                output_base = Path(temp_dir) / "whisper_output"
                
                # Build whisper.cpp command
                cmd = [
                    self.whisper_binary,
                    "-m", str(model_path),
                    "-f", processed_audio,
                    "-of", str(output_base),
                    "-oj",  # Output JSON
                    "-ojf", # Full JSON with timestamps
                    "-np",  # No prints (quiet mode)
                    "-t", "8",  # Use 8 threads for M2 performance
                ]
                
                # Add language if specified, otherwise enable auto-detection
                if language:
                    cmd.extend(["-l", language])
                else:
                    cmd.extend(["-l", "auto"])  # Enable auto-detection
                    
                logging.info(f"Running whisper.cpp: {' '.join(cmd)}")
                
                # Run transcription
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout
                )
                
                if result.returncode != 0:
                    raise Exception(f"whisper.cpp failed: {result.stderr}")
                    
                # Read JSON output
                json_file = Path(str(output_base) + ".json")
                if not json_file.exists():
                    raise Exception("whisper.cpp JSON output not found")
                    
                with open(json_file, 'r', encoding='utf-8') as f:
                    whisper_data = json.load(f)
                    
                # Parse results
                text = whisper_data.get("transcription", [])
                if isinstance(text, list):
                    text = " ".join([t.get("text", "") for t in text])
                elif isinstance(text, str):
                    pass
                else:
                    text = str(text)
                    
                # Extract segments with timestamps
                segments = []
                confidence_scores = []
                
                transcription_data = whisper_data.get("transcription", [])
                if isinstance(transcription_data, list):
                    for segment in transcription_data:
                        if isinstance(segment, dict):
                            seg_data = {
                                "start": segment.get("offsets", {}).get("from", 0) / 1000.0,  # Convert ms to seconds
                                "end": segment.get("offsets", {}).get("to", 0) / 1000.0,
                                "text": segment.get("text", "")
                            }
                            segments.append(seg_data)
                            
                            # Approximate confidence from whisper.cpp output
                            confidence_scores.append(0.85)  # whisper.cpp doesn't provide confidence
                            
                # Clean up preprocessed file if it was created
                if processed_audio != file_path and Path(processed_audio).exists():
                    Path(processed_audio).unlink()
                    
                processing_time = time.time() - start_time
                
                # Detect language from output
                detected_language = whisper_data.get("language", "unknown")
                
                # Calculate duration
                if segments:
                    duration = max([seg["end"] for seg in segments])
                else:
                    # Fallback: get duration from audio file
                    try:
                        audio = AudioSegment.from_file(file_path)
                        duration = len(audio) / 1000.0  # Convert to seconds
                    except:
                        duration = 0.0
                        
                logging.info(f"whisper.cpp transcription completed in {processing_time:.2f}s")
                logging.info(f"Detected language: {detected_language}")
                logging.info(f"Generated {len(segments)} segments")
                
                # Track optimizations applied
                optimizations = []
                if processed_audio != file_path:
                    optimizations.append("audio_preprocessing")
                optimizations.extend(["16khz_mono_conversion", "8_thread_processing"])
                
                return WhisperCppResult(
                    text=text.strip(),
                    segments=segments,
                    language=detected_language,
                    duration=duration,
                    processing_time=processing_time,
                    model_used=self.model_size,
                    confidence_scores=confidence_scores,
                    optimizations_applied=optimizations
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            logging.error(f"whisper.cpp transcription failed after {processing_time:.2f}s: {e}")
            
            return WhisperCppResult(
                text="",
                segments=[],
                language="unknown",
                duration=0.0,
                processing_time=processing_time,
                model_used=self.model_size,
                confidence_scores=[],
                optimizations_applied=[]
            )
    
    def format_transcript_with_timestamps(self, segments: List[Dict]) -> str:
        """Format segments with timestamps for markdown"""
        if not segments:
            return "No timestamped segments available."
            
        formatted_lines = []
        for segment in segments:
            start_time = self._format_time(segment.get("start", 0))
            text = segment.get("text", "").strip()
            formatted_lines.append(f"[{start_time}] {text}")
            
        return '\n'.join(formatted_lines)
    
    def _format_time(self, seconds: float) -> str:
        """Format time in MM:SS format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def get_model_info(self) -> Dict:
        """Get information about the whisper.cpp service"""
        model_path = self.models_dir / self.model_files.get(self.model_size, "unknown")
        return {
            "backend": "whisper.cpp",
            "model_size": self.model_size,
            "model_path": str(model_path),
            "model_exists": model_path.exists(),
            "whisper_binary": self.whisper_binary,
            "optimization": "Apple Silicon optimized"
        }


def test_whisper_cpp(audio_file: str = None):
    """Test function for whisper.cpp backend"""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize service
    service = WhisperCppService(model_size="medium")
    
    # Print model info
    info = service.get_model_info()
    print("=== whisper.cpp Information ===")
    for key, value in info.items():
        print(f"{key}: {value}")
    print()
    
    if audio_file and Path(audio_file).exists():
        print(f"Testing transcription with: {audio_file}")
        result = service.transcribe_audio(audio_file)
        
        print("=== Transcription Result ===")
        print(f"Language: {result.language}")
        print(f"Duration: {result.duration:.2f}s")
        print(f"Processing time: {result.processing_time:.2f}s")
        print(f"Model used: {result.model_used}")
        print(f"Segments: {len(result.segments)}")
        print("\n=== Transcript ===")
        print(result.text)
        print("\n=== Formatted with Timestamps ===")
        print(service.format_transcript_with_timestamps(result.segments))
    else:
        print("No audio file provided for testing")
        print("Usage: python whisper_cpp_backend.py /path/to/audio/file.mp3")


if __name__ == "__main__":
    import sys
    audio_file = sys.argv[1] if len(sys.argv) > 1 else None
    test_whisper_cpp(audio_file)
