#!/usr/bin/env python3
"""
Local Speech-to-Text Transcription Service
Optimized for M2 Mac with 16GB RAM using local Whisper models
"""

import os
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile
import json

# Audio processing
import whisper
import torch
from pydub import AudioSegment
from pydub.silence import split_on_silence
import librosa
import soundfile as sf

# Utilities
import gc
from dataclasses import dataclass


@dataclass
class TranscriptionResult:
    """Structured result from transcription process"""
    text: str
    segments: List[Dict]
    language: str
    duration: float
    confidence_scores: List[float]
    processing_time: float
    model_used: str
    optimizations_applied: List[str]


class LocalTranscriptionService:
    """
    Local speech-to-text service using Whisper models
    Optimized for M2 Mac with audio preprocessing and memory management
    """
    
    def __init__(self, 
                 model_size: str = "medium",
                 device: Optional[str] = None,
                 enable_optimizations: bool = True):
        """
        Initialize local transcription service
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large, large-v2, large-v3)
            device: Device to use (auto-detected if None)
            enable_optimizations: Enable audio preprocessing optimizations
        """
        self.model_size = model_size
        self.enable_optimizations = enable_optimizations
        self.model = None
        self.optimizations_applied = []
        
        # Device configuration for M2 Mac
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"  # Metal Performance Shaders for M2
                logging.info("Using MPS (Metal) acceleration on M2 Mac")
            else:
                self.device = "cpu"
                logging.info("Using CPU (MPS not available)")
        else:
            self.device = device
            
        # Model specifications
        self.model_specs = {
            "tiny": {"size": "39 MB", "speed": "~32x", "accuracy": "Good for simple audio"},
            "base": {"size": "74 MB", "speed": "~16x", "accuracy": "Better accuracy"},
            "small": {"size": "244 MB", "speed": "~6x", "accuracy": "Good balance"},
            "medium": {"size": "769 MB", "speed": "~2x", "accuracy": "High accuracy (recommended)"},
            "large": {"size": "1550 MB", "speed": "1x", "accuracy": "Highest accuracy"},
            "large-v2": {"size": "1550 MB", "speed": "1x", "accuracy": "Improved large model"},
            "large-v3": {"size": "1550 MB", "speed": "1x", "accuracy": "Latest large model"}
        }
        
        # Audio processing settings
        self.audio_settings = {
            "target_sample_rate": 16000,  # Whisper's expected sample rate
            "silence_threshold": -40,     # dBFS threshold for silence detection
            "min_silence_len": 500,       # Minimum silence length in ms
            "keep_silence": 100,          # Keep this much silence around chunks
            "chunk_length_ms": 30000,     # Maximum chunk length (30 seconds)
            "max_file_size_mb": 500       # Maximum file size to process
        }
        
        logging.info(f"Initialized LocalTranscriptionService with model '{model_size}' on device '{self.device}'")
        
    def load_model(self) -> bool:
        """Load Whisper model with error handling"""
        try:
            logging.info(f"Loading Whisper model '{self.model_size}'...")
            model_info = self.model_specs.get(self.model_size, {})
            logging.info(f"Model specs: {model_info}")
            
            start_time = time.time()
            self.model = whisper.load_model(self.model_size, device=self.device)
            load_time = time.time() - start_time
            
            logging.info(f"Model loaded successfully in {load_time:.2f} seconds")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load Whisper model '{self.model_size}': {e}")
            # Try fallback to smaller model
            if self.model_size != "base":
                logging.info("Attempting fallback to 'base' model...")
                self.model_size = "base"
                return self.load_model()
            return False
    
    def optimize_audio(self, audio_path: str) -> Tuple[str, List[str]]:
        """
        Optimize audio file for better transcription
        
        Returns:
            Tuple of (optimized_file_path, list_of_optimizations_applied)
        """
        optimizations = []
        
        try:
            # Load audio file
            audio = AudioSegment.from_file(audio_path)
            original_duration = len(audio)
            
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
                optimizations.append("converted_to_mono")
                
            # Normalize sample rate to 16kHz (Whisper's preferred rate)
            if audio.frame_rate != self.audio_settings["target_sample_rate"]:
                audio = audio.set_frame_rate(self.audio_settings["target_sample_rate"])
                optimizations.append(f"resampled_to_{self.audio_settings['target_sample_rate']}hz")
            
            # Normalize audio levels
            audio = audio.normalize()
            optimizations.append("normalized_levels")
            
            # Remove silence (if enabled and beneficial)
            if self.enable_optimizations:
                # Split on silence and rejoin with minimal silence
                chunks = split_on_silence(
                    audio,
                    min_silence_len=self.audio_settings["min_silence_len"],
                    silence_thresh=self.audio_settings["silence_threshold"],
                    keep_silence=self.audio_settings["keep_silence"]
                )
                
                if len(chunks) > 1:  # Only if silence was actually found
                    audio = AudioSegment.empty()
                    for chunk in chunks:
                        audio += chunk + AudioSegment.silent(duration=self.audio_settings["keep_silence"])
                    
                    new_duration = len(audio)
                    reduction_percent = ((original_duration - new_duration) / original_duration) * 100
                    
                    if reduction_percent > 5:  # Only if significant reduction
                        optimizations.append(f"removed_silence_{reduction_percent:.1f}%_reduction")
                    else:
                        # Reload original if minimal benefit
                        audio = AudioSegment.from_file(audio_path)
                        optimizations = [opt for opt in optimizations if "removed_silence" not in opt]
            
            # Export optimized audio to temporary file
            temp_dir = Path(tempfile.gettempdir()) / "plaud_processor"
            temp_dir.mkdir(exist_ok=True)
            
            optimized_path = temp_dir / f"optimized_{Path(audio_path).stem}.wav"
            audio.export(str(optimized_path), format="wav")
            
            logging.info(f"Audio optimized: {len(optimizations)} optimizations applied")
            return str(optimized_path), optimizations
            
        except Exception as e:
            logging.warning(f"Audio optimization failed: {e}. Using original file.")
            return audio_path, ["optimization_failed_using_original"]
    
    def transcribe_audio(self, file_path: str, language: Optional[str] = None) -> TranscriptionResult:
        """
        Transcribe audio file with optimizations
        
        Args:
            file_path: Path to audio file
            language: Language code (auto-detected if None)
            
        Returns:
            TranscriptionResult with comprehensive information
        """
        start_time = time.time()
        
        try:
            # Ensure model is loaded
            if self.model is None:
                if not self.load_model():
                    raise Exception("Failed to load Whisper model")
            
            # Validate file
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Audio file not found: {file_path}")
                
            file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            if file_size_mb > self.audio_settings["max_file_size_mb"]:
                raise ValueError(f"File too large: {file_size_mb:.1f}MB > {self.audio_settings['max_file_size_mb']}MB")
            
            # Optimize audio if enabled
            audio_path = file_path
            optimizations = []
            
            if self.enable_optimizations:
                audio_path, optimizations = self.optimize_audio(file_path)
                self.optimizations_applied = optimizations
            
            # Transcribe with Whisper
            logging.info(f"Starting transcription with model '{self.model_size}'...")
            
            # Configure transcription options
            transcribe_options = {
                "verbose": True,
                "language": language,
                "task": "transcribe",
                "fp16": False,  # More stable on some systems
                "temperature": 0.0,  # Deterministic output
                "best_of": 1,    # Single pass for speed
                "beam_size": 1,  # Greedy decoding for speed
            }
            
            # Perform transcription
            result = self.model.transcribe(audio_path, **transcribe_options)
            
            # Calculate confidence scores (approximation based on segment probabilities)
            confidence_scores = []
            for segment in result.get("segments", []):
                # Use token-level probabilities if available
                if "avg_logprob" in segment:
                    confidence = min(max(segment["avg_logprob"] + 1.0, 0.0), 1.0)
                else:
                    confidence = 0.8  # Default confidence
                confidence_scores.append(confidence)
            
            processing_time = time.time() - start_time
            
            # Clean up temporary files
            if audio_path != file_path and Path(audio_path).exists():
                Path(audio_path).unlink()
            
            # Force garbage collection to free memory
            gc.collect()
            if self.device == "mps":
                torch.mps.empty_cache()
            
            # Create result object
            transcription_result = TranscriptionResult(
                text=result["text"].strip(),
                segments=result.get("segments", []),
                language=result.get("language", "unknown"),
                duration=result.get("duration", 0.0),
                confidence_scores=confidence_scores,
                processing_time=processing_time,
                model_used=self.model_size,
                optimizations_applied=optimizations
            )
            
            logging.info(f"Transcription completed in {processing_time:.2f}s")
            logging.info(f"Detected language: {transcription_result.language}")
            logging.info(f"Average confidence: {sum(confidence_scores)/len(confidence_scores):.2f}" if confidence_scores else "N/A")
            
            return transcription_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logging.error(f"Transcription failed after {processing_time:.2f}s: {e}")
            
            # Return error result
            return TranscriptionResult(
                text="",
                segments=[],
                language="unknown",
                duration=0.0,
                confidence_scores=[],
                processing_time=processing_time,
                model_used=self.model_size,
                optimizations_applied=["error_occurred"]
            )
    
    def format_transcript_with_timestamps(self, segments: List[Dict], include_confidence: bool = False) -> str:
        """Format segments with timestamps for markdown"""
        if not segments:
            return "No timestamped segments available."
            
        formatted_lines = []
        for i, segment in enumerate(segments):
            start_time = self._format_time(segment.get("start", 0))
            end_time = self._format_time(segment.get("end", 0))
            text = segment.get("text", "").strip()
            
            if include_confidence and hasattr(self, 'optimizations_applied'):
                # Add confidence indicator if available
                confidence = segment.get("avg_logprob", -1)
                if confidence > -0.5:
                    conf_indicator = "🟢"  # High confidence
                elif confidence > -1.0:
                    conf_indicator = "🟡"  # Medium confidence  
                else:
                    conf_indicator = "🔴"  # Low confidence
                formatted_lines.append(f"[{start_time}-{end_time}] {conf_indicator} {text}")
            else:
                formatted_lines.append(f"[{start_time}] {text}")
                
        return '\n'.join(formatted_lines)
    
    def _format_time(self, seconds: float) -> str:
        """Format time in MM:SS format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        return {
            "model_size": self.model_size,
            "device": self.device,
            "specifications": self.model_specs.get(self.model_size, {}),
            "optimizations_enabled": self.enable_optimizations,
            "audio_settings": self.audio_settings
        }
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            if self.device == "mps":
                torch.mps.empty_cache()
            logging.info("Model unloaded and memory cleared")


def test_local_transcription(audio_file: str = None):
    """Test function for local transcription"""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize service
    service = LocalTranscriptionService(
        model_size="medium",  # Good balance for M2 Mac
        enable_optimizations=True
    )
    
    # Print model info
    info = service.get_model_info()
    print("=== Model Information ===")
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
        print(f"Optimizations: {result.optimizations_applied}")
        print(f"Average confidence: {sum(result.confidence_scores)/len(result.confidence_scores):.2f}" if result.confidence_scores else "N/A")
        print("\n=== Transcript ===")
        print(result.text)
        print("\n=== Formatted with Timestamps ===")
        print(service.format_transcript_with_timestamps(result.segments))
    else:
        print("No audio file provided for testing")
        print("Usage: python local_transcription.py /path/to/audio/file.mp3")


if __name__ == "__main__":
    import sys
    audio_file = sys.argv[1] if len(sys.argv) > 1 else None
    test_local_transcription(audio_file)
