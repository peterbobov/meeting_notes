"""Transcription service with whisper.cpp backend and OpenAI API fallback."""

import os
import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

from pydub import AudioSegment
from openai import OpenAI


@dataclass
class TranscriptionResult:
    """Result from transcription process."""
    text: str
    segments: List[Dict]
    language: str
    duration: float
    processing_time: float
    model_used: str
    confidence_scores: List[float]
    optimizations_applied: List[str]


class ProgressBar:
    """Progress bar for transcription with time estimation."""

    def __init__(self, total_duration: float, description: str = "Transcribing"):
        self.total_duration = total_duration
        self.description = description
        self.start_time = time.time()
        self.last_update = 0
        self.bar_width = 40
        self.last_progress = 0.0
        self.shown_progress = False

    def update_from_timestamp(self, timestamp_line: str):
        """Extract timestamp from whisper.cpp output and update progress."""
        try:
            if '[' in timestamp_line and '-->' in timestamp_line:
                end_part = timestamp_line.split('-->')[1].split(']')[0].strip()
                current_seconds = self._parse_timestamp(end_part)

                if current_seconds > 0 and self.total_duration > 0:
                    progress = min(current_seconds / self.total_duration, 1.0)
                    current_time = time.time()
                    progress_diff = progress - self.last_progress

                    if (current_time - self.last_update >= 2.0 or
                        progress_diff >= 0.02 or progress >= 1.0):
                        self._display_progress(progress, current_seconds)
                        self.last_update = current_time
                        self.last_progress = progress
        except Exception:
            pass

    def _parse_timestamp(self, timestamp_str: str) -> float:
        """Parse HH:MM:SS.mmm to seconds."""
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
        """Display progress bar."""
        # Skip in verbose mode
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and hasattr(handler.stream, 'name'):
                if handler.stream.name == '<stderr>' and handler.level == logging.DEBUG:
                    return

        elapsed_time = time.time() - self.start_time
        if progress > 0.01:
            estimated_total = elapsed_time / progress
            remaining_time = max(0, estimated_total - elapsed_time)
        else:
            remaining_time = 0

        filled_width = int(self.bar_width * progress)
        bar = '█' * filled_width + '░' * (self.bar_width - filled_width)
        remaining_str = self._format_time(remaining_time)
        current_str = self._format_time(current_seconds)
        total_str = self._format_time(self.total_duration)

        percentage = progress * 100
        progress_line = f"{self.description}: {bar} {percentage:.0f}% ({current_str}/{total_str}, {remaining_str} remaining)"
        print(f"\r{progress_line}", end='', flush=True)
        self.shown_progress = True

    def finish(self, success: bool = True):
        """Complete the progress bar."""
        elapsed = time.time() - self.start_time
        elapsed_str = self._format_time(elapsed)

        if success:
            bar = '█' * self.bar_width
            final_line = f"{self.description}: {bar} 100% (completed in {elapsed_str})"
        else:
            final_line = f"{self.description}: Failed after {elapsed_str}"

        if self.shown_progress or success:
            print(f"\r{final_line}")

    def _format_time(self, seconds: float) -> str:
        """Format time in MM:SS format."""
        if seconds < 60:
            return f"{int(seconds)}s"
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"


class WhisperCppBackend:
    """whisper.cpp backend optimized for Apple Silicon."""

    MODEL_FILES = {
        "tiny": "ggml-tiny.bin",
        "base": "ggml-base.bin",
        "small": "ggml-small.bin",
        "medium": "ggml-medium.bin",
        "large": "ggml-large.bin",
        "large-v2": "ggml-large-v2.bin",
        "large-v3": "ggml-large-v3.bin"
    }

    MODEL_URLS = {
        "tiny": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
        "base": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
        "small": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
        "medium": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
        "large-v2": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v2.bin",
        "large-v3": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin"
    }

    def __init__(self, model_size: str = "medium", models_dir: str = "./whisper_models"):
        self.model_size = model_size
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.whisper_binary = self._find_whisper_binary()

        if not self.whisper_binary:
            raise RuntimeError("whisper-cpp not found. Install with: brew install whisper-cpp")

        logging.info(f"Initialized whisper.cpp with model '{model_size}'")

    def _find_whisper_binary(self) -> Optional[str]:
        """Find whisper-cpp binary."""
        for name in ["whisper-cli", "whisper-cpp", "whisper"]:
            try:
                result = subprocess.run(["which", name], capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout.strip()
            except Exception:
                continue
        return None

    def ensure_model_available(self) -> bool:
        """Ensure model is available, download if needed."""
        model_file = self.MODEL_FILES.get(self.model_size)
        if not model_file:
            logging.error(f"Unknown model size: {self.model_size}")
            return False

        model_path = self.models_dir / model_file
        if model_path.exists():
            return True

        download_url = self.MODEL_URLS.get(self.model_size)
        if not download_url:
            logging.error(f"No download URL for model: {self.model_size}")
            return False

        logging.info(f"Downloading model {self.model_size}...")
        try:
            import urllib.request
            urllib.request.urlretrieve(download_url, model_path)
            logging.info(f"Model downloaded: {model_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to download model: {e}")
            return False

    def preprocess_audio(self, audio_path: str) -> str:
        """Convert audio to 16kHz mono WAV for whisper.cpp."""
        try:
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_frame_rate(16000).set_channels(1)

            temp_dir = Path(tempfile.gettempdir()) / "plaud_processor"
            temp_dir.mkdir(exist_ok=True)
            temp_file = temp_dir / f"preprocessed_{Path(audio_path).stem}.wav"
            audio.export(str(temp_file), format="wav")
            return str(temp_file)
        except Exception as e:
            logging.warning(f"Audio preprocessing failed: {e}")
            return audio_path

    def transcribe(self, file_path: str, language: Optional[str] = None) -> TranscriptionResult:
        """Transcribe audio using whisper.cpp."""
        start_time = time.time()

        try:
            if not self.ensure_model_available():
                raise Exception(f"Model {self.model_size} not available")

            processed_audio = self.preprocess_audio(file_path)

            # Get audio duration
            try:
                audio = AudioSegment.from_file(file_path)
                audio_duration = len(audio) / 1000.0
            except:
                audio_duration = 0

            model_path = self.models_dir / self.MODEL_FILES[self.model_size]

            with tempfile.TemporaryDirectory() as temp_dir:
                output_base = Path(temp_dir) / "whisper_output"

                cmd = [
                    self.whisper_binary,
                    "-m", str(model_path),
                    "-f", processed_audio,
                    "-of", str(output_base),
                    "-oj", "-ojf",
                    "-t", "8",
                    "-l", language if language else "auto"
                ]

                progress_bar = ProgressBar(audio_duration, "Transcribing audio") if audio_duration > 0 else None
                partial_lines = []

                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

                while True:
                    chunk = process.stdout.read(1024)
                    if not chunk and process.poll() is not None:
                        break
                    if chunk:
                        decoded = self._decode_chunk(chunk)
                        for line in decoded.split('\n'):
                            line = line.strip()
                            if line and '-->' in line and ']' in line:
                                partial_lines.append(line)
                                if progress_bar:
                                    progress_bar.update_from_timestamp(line)

                if progress_bar:
                    progress_bar.finish(success=(process.returncode == 0))

                if process.returncode != 0 and partial_lines:
                    return self._create_result_from_partial(partial_lines, start_time, file_path)

                # Parse JSON output
                json_file = Path(str(output_base) + ".json")
                if not json_file.exists():
                    if partial_lines:
                        return self._create_result_from_partial(partial_lines, start_time, file_path)
                    raise Exception("whisper.cpp JSON output not found")

                whisper_data = self._read_json(json_file)
                if whisper_data is None and partial_lines:
                    return self._create_result_from_partial(partial_lines, start_time, file_path)

                # Extract results
                text = whisper_data.get("transcription", [])
                if isinstance(text, list):
                    text = " ".join([t.get("text", "") for t in text])

                segments = []
                for seg in whisper_data.get("transcription", []):
                    if isinstance(seg, dict):
                        segments.append({
                            "start": seg.get("offsets", {}).get("from", 0) / 1000.0,
                            "end": seg.get("offsets", {}).get("to", 0) / 1000.0,
                            "text": seg.get("text", "")
                        })

                # Cleanup
                if processed_audio != file_path and Path(processed_audio).exists():
                    Path(processed_audio).unlink()

                duration = max([s["end"] for s in segments]) if segments else audio_duration

                return TranscriptionResult(
                    text=text.strip(),
                    segments=segments,
                    language=whisper_data.get("language", "unknown"),
                    duration=duration,
                    processing_time=time.time() - start_time,
                    model_used=self.model_size,
                    confidence_scores=[0.85] * len(segments),
                    optimizations_applied=["whisper_cpp", "16khz_mono", "8_threads"]
                )

        except Exception as e:
            logging.error(f"whisper.cpp transcription failed: {e}")
            return TranscriptionResult(
                text="", segments=[], language="unknown", duration=0.0,
                processing_time=time.time() - start_time, model_used=self.model_size,
                confidence_scores=[], optimizations_applied=[]
            )

    def _decode_chunk(self, chunk: bytes) -> str:
        """Decode bytes with encoding fallbacks."""
        for encoding in ['utf-8', 'cp1251', 'latin-1']:
            try:
                return chunk.decode(encoding, errors='ignore')
            except:
                continue
        return chunk.decode('utf-8', errors='replace')

    def _read_json(self, json_file: Path) -> Optional[Dict]:
        """Read JSON with encoding fallbacks."""
        for encoding in ['utf-8', 'cp1251', 'latin-1']:
            try:
                with open(json_file, 'r', encoding=encoding, errors='replace') as f:
                    return json.load(f)
            except:
                continue
        return None

    def _create_result_from_partial(self, lines: List[str], start_time: float, file_path: str) -> TranscriptionResult:
        """Create result from partial transcript lines."""
        full_text = []
        segments = []

        for line in lines:
            try:
                if '-->' in line and ']' in line:
                    timestamp_part = line[line.find('['):line.find(']')+1]
                    text_part = line[line.find(']')+1:].strip()
                    if text_part:
                        full_text.append(text_part)
                        timestamps = timestamp_part.strip('[]').split(' --> ')
                        if len(timestamps) == 2:
                            segments.append({
                                "start": self._parse_timestamp(timestamps[0]),
                                "end": self._parse_timestamp(timestamps[1]),
                                "text": text_part
                            })
            except:
                continue

        duration = max([s["end"] for s in segments]) if segments else 0.0

        return TranscriptionResult(
            text=' '.join(full_text),
            segments=segments,
            language="unknown",
            duration=duration,
            processing_time=time.time() - start_time,
            model_used=self.model_size,
            confidence_scores=[0.8] * len(segments),
            optimizations_applied=["partial_recovery"]
        )

    def _parse_timestamp(self, ts: str) -> float:
        """Parse HH:MM:SS.mmm to seconds."""
        try:
            parts = ts.split(':')
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        except:
            pass
        return 0.0

    def get_model_info(self) -> Dict:
        """Get service info."""
        model_path = self.models_dir / self.MODEL_FILES.get(self.model_size, "unknown")
        return {
            "backend": "whisper.cpp",
            "model_size": self.model_size,
            "model_exists": model_path.exists(),
            "optimization": "Apple Silicon"
        }


class TranscriptionService:
    """Hybrid transcription service: whisper.cpp primary, OpenAI API fallback."""

    def __init__(self, api_key: str, use_local: bool = True, model: str = "medium"):
        self.api_key = api_key
        self.use_local = use_local

        if use_local:
            self.backend = WhisperCppBackend(model_size=model)
            logging.info(f"Using whisper.cpp with model '{model}'")
        else:
            self.client = OpenAI(api_key=api_key)
            logging.info("Using OpenAI Whisper API")

    def transcribe(self, file_path: str, language: Optional[str] = None) -> Dict:
        """Transcribe audio file."""
        if self.use_local:
            result = self.backend.transcribe(file_path, language)
            if result.text:
                return self._result_to_dict(result, "local_whisper")
            # Fallback to API
            logging.info("Local transcription failed, falling back to API...")

        return self._transcribe_api(file_path, language)

    def _transcribe_api(self, file_path: str, language: Optional[str] = None) -> Dict:
        """Transcribe using OpenAI Whisper API."""
        try:
            if not hasattr(self, 'client'):
                self.client = OpenAI(api_key=self.api_key)

            with open(file_path, "rb") as f:
                params = {
                    "model": "whisper-1",
                    "file": f,
                    "response_format": "verbose_json",
                    "timestamp_granularities": ["segment"]
                }
                if language:
                    params["language"] = language

                transcript = self.client.audio.transcriptions.create(**params)

            return {
                'text': transcript.text,
                'segments': transcript.segments if hasattr(transcript, 'segments') else [],
                'duration': transcript.duration if hasattr(transcript, 'duration') else 0,
                'language': transcript.language if hasattr(transcript, 'language') else 'unknown',
                'processing_info': {'method': 'openai_api', 'model': 'whisper-1'}
            }

        except Exception as e:
            logging.error(f"API transcription failed: {e}")
            return {'text': '', 'segments': [], 'duration': 0, 'language': 'unknown', 'error': str(e)}

    def _result_to_dict(self, result: TranscriptionResult, method: str) -> Dict:
        """Convert TranscriptionResult to dict."""
        return {
            'text': result.text,
            'segments': result.segments,
            'duration': result.duration,
            'language': result.language,
            'processing_info': {
                'method': method,
                'model': result.model_used,
                'processing_time': result.processing_time,
                'optimizations': result.optimizations_applied
            }
        }

    def format_transcript(self, segments: List) -> str:
        """Format segments with timestamps for markdown."""
        if not segments:
            return "No transcript available."

        lines = []
        for seg in segments:
            start = seg.get('start', 0) if isinstance(seg, dict) else getattr(seg, 'start', 0)
            text = seg.get('text', '').strip() if isinstance(seg, dict) else getattr(seg, 'text', '').strip()
            minutes, seconds = int(start // 60), int(start % 60)
            lines.append(f"[{minutes:02d}:{seconds:02d}] {text}")

        return '\n'.join(lines)

    def get_info(self) -> Dict:
        """Get service configuration info."""
        if self.use_local and hasattr(self, 'backend'):
            return {'mode': 'local', **self.backend.get_model_info()}
        return {'mode': 'api', 'model': 'whisper-1'}
