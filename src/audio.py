"""Audio file processing, validation, and monitoring."""

import os
import re
import shutil
import datetime
import logging
import tempfile
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torchaudio
from pydub import AudioSegment
from silero_vad import load_silero_vad, get_speech_timestamps

# Suppress torchaudio deprecation warnings about TorchCodec migration
warnings.filterwarnings("ignore", message=".*torchaudio.load_with_torchcodec.*")


class AudioProcessor:
    """Handles audio file processing, validation, and monitoring."""

    def __init__(self, config):
        self.config = config
        self.supported_formats = config['SETTINGS']['supported_formats'].split(',')
        self.max_size_mb = int(config['SETTINGS']['max_file_size_mb'])

        # Get input folder with environment variable substitution
        input_folder_str = config['PATHS'].get('input_folder', './input_audio')
        input_folder_str = self._expand_env_vars(input_folder_str)
        self.input_folder = Path(input_folder_str).expanduser()

        self.processed_folder = Path(config['PATHS']['processed_folder'])

    def _expand_env_vars(self, path_str: str) -> str:
        """Replace ${VAR_NAME} with environment variable values."""
        if '${' not in path_str:
            return path_str

        env_vars = re.findall(r'\$\{([^}]+)\}', path_str)
        for env_var in env_vars:
            env_value = os.getenv(env_var, '')
            if env_value:
                path_str = path_str.replace(f'${{{env_var}}}', env_value)
            else:
                raise ValueError(f"Environment variable {env_var} not found. Please set {env_var} in your .env file")
        return path_str

    def detect_new_files(self, folder_path: str) -> List[str]:
        """Monitor folder for new audio files."""
        folder = Path(folder_path)
        if not folder.exists():
            return []

        audio_files = []
        for ext in self.supported_formats:
            audio_files.extend(folder.glob(f"*.{ext}"))

        return [str(f) for f in audio_files if self.validate_audio_file(str(f))]

    def validate_audio_file(self, file_path: str) -> bool:
        """Check if file is valid audio and within size limits."""
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                return False

            # Check file size
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb > self.max_size_mb:
                logging.warning(f"File {file_path} exceeds size limit: {size_mb:.1f}MB")
                return False

            # Check if it's a valid audio file
            audio = AudioSegment.from_file(str(file_path))
            return len(audio) > 0

        except Exception as e:
            logging.error(f"Error validating audio file {file_path}: {e}")
            return False

    def get_audio_duration(self, file_path: str) -> float:
        """Get duration in minutes."""
        try:
            audio = AudioSegment.from_file(file_path)
            return len(audio) / (1000 * 60)  # Convert ms to minutes
        except Exception as e:
            logging.error(f"Error getting duration for {file_path}: {e}")
            return 0.0

    def move_to_processed(self, file_path: str) -> str:
        """Move processed file to archive folder."""
        try:
            source = Path(file_path)
            dest_folder = self.processed_folder / datetime.date.today().strftime("%Y-%m")
            dest_folder.mkdir(parents=True, exist_ok=True)

            dest_path = dest_folder / source.name
            shutil.move(str(source), str(dest_path))
            return str(dest_path)

        except Exception as e:
            logging.error(f"Error moving file to processed folder: {e}")
            return file_path


@dataclass
class SpeechSegment:
    """A segment of speech with original timestamps."""
    start: float  # seconds in original audio
    end: float    # seconds in original audio


@dataclass
class FilteredAudioResult:
    """Result of filtering silence from audio."""
    filtered_path: str  # Path to the filtered audio file
    speech_segments: List[SpeechSegment]  # Original timestamps of speech
    original_duration: float  # Duration of original audio in seconds
    filtered_duration: float  # Duration of filtered audio in seconds


class SilenceFilter:
    """Filter silence from audio using Silero VAD to prevent Whisper hallucinations."""

    def __init__(
        self,
        min_silence_duration: float = 0.5,
        speech_pad: float = 0.1,
        threshold: float = 0.5
    ):
        """
        Initialize silence filter.

        Args:
            min_silence_duration: Minimum silence duration (seconds) to filter out
            speech_pad: Padding to add around speech segments (seconds)
            threshold: VAD threshold (0-1), higher = more strict
        """
        self.min_silence_duration = min_silence_duration
        self.speech_pad = speech_pad
        self.threshold = threshold
        self._model = None

    def _get_model(self):
        """Lazy load the VAD model."""
        if self._model is None:
            logging.info("Loading Silero VAD model...")
            self._model = load_silero_vad()
        return self._model

    def _convert_to_wav_for_vad(self, audio_path: str) -> str:
        """
        Convert audio to 16kHz mono WAV for VAD processing.

        torchaudio doesn't support all formats (e.g., m4a), so we use pydub
        which leverages ffmpeg for broad format support.
        """
        temp_dir = Path(tempfile.gettempdir()) / "plaud_processor"
        temp_dir.mkdir(exist_ok=True)
        wav_path = str(temp_dir / f"vad_{Path(audio_path).stem}.wav")

        # Use pydub to convert (supports m4a, mp3, etc. via ffmpeg)
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(wav_path, format="wav")

        return wav_path

    def detect_speech_segments(self, audio_path: str) -> List[SpeechSegment]:
        """
        Detect speech segments in audio file using Silero VAD.

        Args:
            audio_path: Path to audio file

        Returns:
            List of SpeechSegment with start/end times in seconds
        """
        model = self._get_model()
        converted_path = None

        try:
            # Try loading directly with torchaudio first
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
            except Exception as e:
                # Fall back to converting via pydub/ffmpeg for unsupported formats
                logging.debug(f"torchaudio can't load {audio_path}, converting via ffmpeg: {e}")
                converted_path = self._convert_to_wav_for_vad(audio_path)
                waveform, sample_rate = torchaudio.load(converted_path)

            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Get speech timestamps
            speech_timestamps = get_speech_timestamps(
                waveform.squeeze(),
                model,
                threshold=self.threshold,
                sampling_rate=sample_rate,
                min_silence_duration_ms=int(self.min_silence_duration * 1000),
                speech_pad_ms=int(self.speech_pad * 1000),
                return_seconds=True
            )

            segments = [
                SpeechSegment(start=ts['start'], end=ts['end'])
                for ts in speech_timestamps
            ]

            logging.debug(f"Detected {len(segments)} speech segments")
            return segments

        finally:
            # Cleanup converted file
            if converted_path and Path(converted_path).exists():
                Path(converted_path).unlink()

    def filter_silence(self, audio_path: str, output_path: Optional[str] = None) -> FilteredAudioResult:
        """
        Remove silence from audio file, keeping only speech segments.

        Args:
            audio_path: Path to input audio file
            output_path: Optional output path (will create temp file if not provided)

        Returns:
            FilteredAudioResult with path to filtered audio and segment mapping
        """
        # Detect speech segments
        speech_segments = self.detect_speech_segments(audio_path)

        if not speech_segments:
            logging.warning("No speech detected in audio file")
            # Return original file if no speech detected
            audio = AudioSegment.from_file(audio_path)
            return FilteredAudioResult(
                filtered_path=audio_path,
                speech_segments=[SpeechSegment(0, len(audio) / 1000)],
                original_duration=len(audio) / 1000,
                filtered_duration=len(audio) / 1000
            )

        # Load original audio with pydub for manipulation
        audio = AudioSegment.from_file(audio_path)
        original_duration = len(audio) / 1000  # Convert ms to seconds

        # Extract and concatenate speech segments
        filtered_audio = AudioSegment.empty()
        for segment in speech_segments:
            start_ms = int(segment.start * 1000)
            end_ms = int(segment.end * 1000)
            filtered_audio += audio[start_ms:end_ms]

        filtered_duration = len(filtered_audio) / 1000

        # Determine output path
        if output_path is None:
            temp_dir = Path(tempfile.gettempdir()) / "plaud_processor"
            temp_dir.mkdir(exist_ok=True)
            output_path = str(temp_dir / f"filtered_{Path(audio_path).stem}.wav")

        # Export filtered audio
        filtered_audio.export(output_path, format="wav")

        silence_removed = original_duration - filtered_duration
        logging.info(
            f"Filtered audio: {original_duration:.1f}s → {filtered_duration:.1f}s "
            f"(removed {silence_removed:.1f}s of silence, {len(speech_segments)} speech segments)"
        )

        return FilteredAudioResult(
            filtered_path=output_path,
            speech_segments=speech_segments,
            original_duration=original_duration,
            filtered_duration=filtered_duration
        )


class TimestampRemapper:
    """Remap timestamps from filtered audio back to original audio time."""

    def __init__(self, speech_segments: List[SpeechSegment]):
        """
        Initialize with speech segments from the original audio.

        Args:
            speech_segments: List of SpeechSegment with original timestamps
        """
        self.speech_segments = speech_segments
        self._build_mapping()

    def _build_mapping(self):
        """Build cumulative offset mapping for fast lookups."""
        self.filtered_offsets = []  # Start time in filtered audio for each segment
        cumulative = 0.0

        for segment in self.speech_segments:
            self.filtered_offsets.append(cumulative)
            cumulative += segment.end - segment.start

    def remap_timestamp(self, filtered_time: float) -> float:
        """
        Convert a timestamp from filtered audio to original audio time.

        Args:
            filtered_time: Timestamp in the filtered (silence-removed) audio

        Returns:
            Corresponding timestamp in the original audio
        """
        if not self.speech_segments:
            return filtered_time

        # Find which segment this time falls into
        for i, (segment, offset) in enumerate(zip(self.speech_segments, self.filtered_offsets)):
            segment_duration = segment.end - segment.start
            next_offset = offset + segment_duration

            if filtered_time < next_offset or i == len(self.speech_segments) - 1:
                # Time falls within this segment
                time_into_segment = filtered_time - offset
                # Clamp to segment duration
                time_into_segment = min(time_into_segment, segment_duration)
                return segment.start + time_into_segment

        # Fallback: return last segment end
        return self.speech_segments[-1].end

    def remap_segments(self, segments: List[dict]) -> List[dict]:
        """
        Remap all timestamps in a list of transcript segments.

        Args:
            segments: List of dicts with 'start' and 'end' keys

        Returns:
            List of segments with remapped timestamps
        """
        remapped = []
        for seg in segments:
            remapped_seg = seg.copy()
            remapped_seg['start'] = self.remap_timestamp(seg.get('start', 0))
            remapped_seg['end'] = self.remap_timestamp(seg.get('end', 0))
            remapped.append(remapped_seg)
        return remapped
