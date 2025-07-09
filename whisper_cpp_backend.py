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
import re

from pydub import AudioSegment


class SimpleProgressBar:
    """Simple progress bar for whisper.cpp transcription"""
    
    def __init__(self, total_duration: float, description: str = "Transcribing"):
        self.total_duration = total_duration
        self.description = description
        self.start_time = time.time()
        self.last_update = 0
        self.bar_width = 40
        self.last_progress = 0.0
        self.shown_progress = False
        
    def update_from_timestamp(self, timestamp_line: str):
        """Extract timestamp from whisper.cpp output and update progress"""
        try:
            # Parse timestamp like "[00:12:34.100 --> 00:12:34.100]"
            if '[' in timestamp_line and '-->' in timestamp_line:
                # Extract the end timestamp
                end_part = timestamp_line.split('-->')[1].split(']')[0].strip()
                current_seconds = self._parse_timestamp(end_part)
                
                if current_seconds > 0 and self.total_duration > 0:
                    progress = min(current_seconds / self.total_duration, 1.0)
                    
                    # Only update every 2 seconds or if progress increased significantly
                    current_time = time.time()
                    progress_diff = progress - self.last_progress
                    
                    if (current_time - self.last_update >= 2.0 or 
                        progress_diff >= 0.02 or  # 2% progress change
                        progress >= 1.0):
                        self._display_progress(progress, current_seconds)
                        self.last_update = current_time
                        self.last_progress = progress
        except Exception:
            pass  # Ignore parsing errors
    
    def _parse_timestamp(self, timestamp_str: str) -> float:
        """Parse HH:MM:SS.mmm to seconds"""
        try:
            parts = timestamp_str.split(':')
            if len(parts) == 3:
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds
        except:
            pass
        return 0.0
    
    def _display_progress(self, progress: float, current_seconds: float):
        """Display progress bar that updates in place like tqdm"""
        import sys
        
        # Only show progress bar in non-verbose mode (when console level is INFO, not DEBUG)
        root_logger = logging.getLogger()
        console_handler = None
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream.name == '<stderr>':
                console_handler = handler
                break
        
        # Skip progress bar if in verbose mode (DEBUG level) or if no console handler found
        if console_handler and console_handler.level == logging.DEBUG:
            return
            
        # Calculate time estimates
        elapsed_time = time.time() - self.start_time
        if progress > 0.01:  # Avoid division by zero
            estimated_total = elapsed_time / progress
            remaining_time = max(0, estimated_total - elapsed_time)
        else:
            remaining_time = 0
        
        # Create progress bar
        filled_width = int(self.bar_width * progress)
        bar = '█' * filled_width + '░' * (self.bar_width - filled_width)
        
        # Format time
        remaining_str = self._format_time(remaining_time)
        current_str = self._format_time(current_seconds)
        total_str = self._format_time(self.total_duration)
        
        # Create progress line
        percentage = progress * 100
        progress_line = f"{self.description}: {bar} {percentage:.0f}% ({current_str}/{total_str}, {remaining_str} remaining)"
        
        # Print to stdout with carriage return to overwrite the same line
        print(f"\r{progress_line}", end='', flush=True)
        self.shown_progress = True
    
    def finish(self, success: bool = True):
        """Complete the progress bar"""
        elapsed = time.time() - self.start_time
        elapsed_str = self._format_time(elapsed)
        
        if success:
            bar = '█' * self.bar_width
            final_line = f"{self.description}: {bar} 100% (completed in {elapsed_str})"
        else:
            final_line = f"{self.description}: Failed after {elapsed_str}"
        
        # Only show final progress if we've shown progress before
        if self.shown_progress or success:
            print(f"\r{final_line}")  # Final line with newline
        elif self.shown_progress:
            print()  # Just add a newline if we showed progress
    
    def _format_time(self, seconds: float) -> str:
        """Format time in MM:SS format"""
        if seconds < 60:
            return f"{int(seconds)}s"
        else:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}:{secs:02d}"


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
            logging.info(f"[WHISPER_CPP] Preprocessing audio: {Path(audio_path).name}")
            
            # Load audio
            audio = AudioSegment.from_file(audio_path)
            original_rate = audio.frame_rate
            original_channels = audio.channels
            
            logging.info(f"[WHISPER_CPP] Original audio specs: {original_rate}Hz, {original_channels} channel(s)")
            
            # Convert to 16kHz mono WAV (whisper.cpp preferred format)
            audio = audio.set_frame_rate(16000).set_channels(1)
            
            # Export to temporary WAV file
            temp_dir = Path(tempfile.gettempdir()) / "plaud_processor"
            temp_dir.mkdir(exist_ok=True)
            
            temp_file = temp_dir / f"preprocessed_{Path(audio_path).stem}.wav"
            audio.export(str(temp_file), format="wav")
            
            logging.info(f"[WHISPER_CPP] Audio preprocessed to 16kHz mono: {temp_file}")
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
            logging.info(f"[WHISPER_CPP] Starting transcription process for {Path(file_path).name}")
            logging.info(f"[WHISPER_CPP] Model: {self.model_size}")
            logging.info(f"[WHISPER_CPP] Language: {language if language else 'auto-detect'}")
            
            # Ensure model is available
            logging.info("[WHISPER_CPP] Checking model availability...")
            if not self.ensure_model_available():
                raise Exception(f"Model {self.model_size} not available")
                
            # Preprocess audio
            processed_audio = self.preprocess_audio(file_path)
            
            # Get audio duration for progress bar
            try:
                audio = AudioSegment.from_file(file_path)
                audio_duration_seconds = len(audio) / 1000.0
            except Exception as e:
                logging.warning(f"[WHISPER_CPP] Could not get audio duration: {e}")
                audio_duration_seconds = 0
            
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
                    "-t", "8",  # Use 8 threads for M2 performance
                ]
                
                # Add language if specified, otherwise enable auto-detection
                if language:
                    cmd.extend(["-l", language])
                else:
                    cmd.extend(["-l", "auto"])  # Enable auto-detection
                    
                logging.info(f"[WHISPER_CPP] Running whisper.cpp command...")
                logging.info(f"[WHISPER_CPP] Command: {' '.join(cmd)}")
                
                # Initialize progress bar (will be used in non-verbose mode)
                progress_bar = None
                if audio_duration_seconds > 0:
                    progress_bar = SimpleProgressBar(audio_duration_seconds, "Transcribing audio")
                
                # Run transcription with real-time progress capture and robust encoding handling
                logging.info("[WHISPER_CPP] Starting transcription with real-time progress...")
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1
                )
                
                # Capture and log progress in real-time with encoding fallback
                output_lines = []
                raw_output = b''
                progress_patterns = ['progress', '%', 'whisper_full', 'processing', 'load_state', 'model']
                
                # Save all output for recovery purposes
                partial_transcript_lines = []
                
                while True:
                    chunk = process.stdout.read(1024)
                    if not chunk and process.poll() is not None:
                        break
                    if chunk:
                        raw_output += chunk
                        
                        # Try to decode chunk with multiple encoding options
                        decoded_chunk = self._decode_with_fallback(chunk)
                        if decoded_chunk:
                            lines = decoded_chunk.split('\n')
                            for line in lines[:-1]:  # Process complete lines
                                line = line.strip()
                                if line:
                                    output_lines.append(line)
                                    
                                    # Save transcript lines for recovery
                                    if '-->' in line and ']' in line:  # Transcript line with timestamp
                                        partial_transcript_lines.append(line)
                                        
                                        # Update progress bar if available
                                        if progress_bar:
                                            progress_bar.update_from_timestamp(line)
                                    
                                    # Parse and log progress for relevant lines (in verbose mode)
                                    line_lower = line.lower()
                                    if any(pattern in line_lower for pattern in progress_patterns):
                                        logging.info(f"[WHISPER_CPP] {line}")
                                    elif line and not line.startswith('whisper_') and len(line) < 100:
                                        # Log other relevant short messages (in verbose mode)
                                        logging.info(f"[WHISPER_CPP] {line}")
                
                return_code = process.poll()
                
                # Finish progress bar
                if progress_bar:
                    progress_bar.finish(success=(return_code == 0))
                
                # Save partial transcript for recovery even if process fails
                if partial_transcript_lines:
                    logging.info(f"[WHISPER_CPP] Captured {len(partial_transcript_lines)} transcript lines for recovery")
                    self._save_partial_transcript(output_base, partial_transcript_lines)
                
                if return_code != 0:
                    # Try to recover partial results before failing
                    if partial_transcript_lines:
                        logging.warning(f"[WHISPER_CPP] Process failed but attempting to recover {len(partial_transcript_lines)} transcript lines")
                        return self._create_result_from_partial(partial_transcript_lines, start_time, file_path)
                    
                    full_output = '\n'.join(output_lines[-10:])  # Last 10 lines for error context
                    raise Exception(f"whisper.cpp failed with return code {return_code}. Last output: {full_output}")
                    
                # Read JSON output with encoding fallback
                json_file = Path(str(output_base) + ".json")
                if not json_file.exists():
                    # If JSON not found but we have partial transcript, try to recover
                    if partial_transcript_lines:
                        logging.warning(f"[WHISPER_CPP] JSON not found but attempting recovery from {len(partial_transcript_lines)} partial lines")
                        return self._create_result_from_partial(partial_transcript_lines, start_time, file_path)
                    raise Exception("whisper.cpp JSON output not found")
                
                # Try reading JSON with multiple encodings
                whisper_data = None
                json_read_error = None
                
                for encoding in ['utf-8', 'cp1251', 'latin-1']:
                    try:
                        with open(json_file, 'r', encoding=encoding, errors='replace') as f:
                            whisper_data = json.load(f)
                        logging.info(f"[WHISPER_CPP] Successfully read JSON with {encoding} encoding")
                        break
                    except (UnicodeDecodeError, json.JSONDecodeError) as e:
                        json_read_error = e
                        logging.warning(f"[WHISPER_CPP] Failed to read JSON with {encoding}: {e}")
                        continue
                
                if whisper_data is None:
                    # Final fallback: try to recover from partial transcript
                    if partial_transcript_lines:
                        logging.warning(f"[WHISPER_CPP] JSON parsing completely failed, recovering from {len(partial_transcript_lines)} partial lines")
                        return self._create_result_from_partial(partial_transcript_lines, start_time, file_path)
                    raise Exception(f"Failed to read JSON output with any encoding. Last error: {json_read_error}")
                    
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
                    logging.info("[WHISPER_CPP] Cleaning up temporary files...")
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
                        
                avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.85
                
                logging.info(f"[WHISPER_CPP] Transcription completed:")
                logging.info(f"[WHISPER_CPP] - Processing time: {processing_time:.2f}s")
                logging.info(f"[WHISPER_CPP] - Detected language: {detected_language}")
                logging.info(f"[WHISPER_CPP] - Segments generated: {len(segments)}")
                logging.info(f"[WHISPER_CPP] - Text length: {len(text)} characters")
                logging.info(f"[WHISPER_CPP] - Average confidence: {avg_confidence:.2f}")
                logging.info(f"[WHISPER_CPP] - Audio duration: {duration:.1f}s")
                
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
    
    def _decode_with_fallback(self, chunk: bytes) -> str:
        """
        Decode bytes with multiple encoding fallbacks to handle Cyrillic text
        """
        encodings = ['utf-8', 'cp1251', 'latin-1', 'utf-16']
        
        for encoding in encodings:
            try:
                return chunk.decode(encoding, errors='ignore')
            except (UnicodeDecodeError, LookupError):
                continue
        
        # Last resort: decode with errors replaced
        try:
            return chunk.decode('utf-8', errors='replace')
        except:
            return chunk.decode('latin-1', errors='replace')
    
    def _save_partial_transcript(self, output_base: Path, transcript_lines: List[str]):
        """
        Save partial transcript lines for recovery purposes
        """
        try:
            partial_file = Path(str(output_base) + "_partial.txt")
            with open(partial_file, 'w', encoding='utf-8', errors='replace') as f:
                for line in transcript_lines:
                    f.write(line + '\n')
            logging.info(f"[WHISPER_CPP] Saved partial transcript to {partial_file}")
        except Exception as e:
            logging.warning(f"[WHISPER_CPP] Failed to save partial transcript: {e}")
    
    def _create_result_from_partial(self, transcript_lines: List[str], start_time: float, file_path: str) -> 'WhisperCppResult':
        """
        Create a result object from partial transcript lines when JSON parsing fails
        """
        try:
            # Parse transcript lines to extract text and segments
            full_text = []
            segments = []
            
            for line in transcript_lines:
                if '-->' in line and ']' in line:
                    # Parse timestamp format: [00:12:34.100 --> 00:12:34.100]   Text
                    try:
                        # Extract timestamp and text
                        timestamp_part = line[line.find('['):line.find(']')+1]
                        text_part = line[line.find(']')+1:].strip()
                        
                        if text_part:
                            full_text.append(text_part)
                            
                            # Parse timestamps
                            timestamps = timestamp_part.strip('[]').split(' --> ')
                            if len(timestamps) == 2:
                                start_str, end_str = timestamps
                                start_seconds = self._parse_timestamp(start_str)
                                end_seconds = self._parse_timestamp(end_str)
                                
                                segments.append({
                                    "start": start_seconds,
                                    "end": end_seconds,
                                    "text": text_part
                                })
                    except Exception as e:
                        logging.warning(f"[WHISPER_CPP] Failed to parse line '{line}': {e}")
                        # Still include the text even if timestamp parsing fails
                        text_part = line.split(']')[-1].strip() if ']' in line else line.strip()
                        if text_part:
                            full_text.append(text_part)
            
            combined_text = ' '.join(full_text)
            processing_time = time.time() - start_time
            
            # Estimate duration from segments or audio file
            duration = 0.0
            if segments:
                duration = max([seg["end"] for seg in segments])
            else:
                try:
                    from pydub import AudioSegment
                    audio = AudioSegment.from_file(file_path)
                    duration = len(audio) / 1000.0
                except:
                    duration = 0.0
            
            logging.info(f"[WHISPER_CPP] Recovery successful: {len(combined_text)} characters, {len(segments)} segments")
            
            return WhisperCppResult(
                text=combined_text,
                segments=segments,
                language="unknown",  # Can't detect from partial data
                duration=duration,
                processing_time=processing_time,
                model_used=self.model_size,
                confidence_scores=[0.8] * len(segments),  # Approximate confidence
                optimizations_applied=["partial_recovery", "encoding_fallback"]
            )
            
        except Exception as e:
            logging.error(f"[WHISPER_CPP] Failed to create result from partial transcript: {e}")
            # Return minimal result to prevent complete failure
            return WhisperCppResult(
                text=' '.join([line.split(']')[-1].strip() if ']' in line else line.strip() 
                              for line in transcript_lines if line.strip()]),
                segments=[],
                language="unknown",
                duration=0.0,
                processing_time=time.time() - start_time,
                model_used=self.model_size,
                confidence_scores=[],
                optimizations_applied=["emergency_recovery"]
            )
    
    def _parse_timestamp(self, timestamp_str: str) -> float:
        """
        Parse timestamp string (HH:MM:SS.mmm) to seconds
        """
        try:
            parts = timestamp_str.split(':')
            if len(parts) == 3:
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds
            return 0.0
        except:
            return 0.0

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
