"""Speaker diarization service using pyannote-audio."""

import os
import gc
import time
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

import torch
from pydub import AudioSegment

try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    logging.warning("pyannote-audio not installed. Speaker diarization disabled.")


@dataclass
class SpeakerSegment:
    """A segment attributed to a speaker."""
    start_time: float
    end_time: float
    speaker_id: str
    confidence: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class DiarizationResult:
    """Result from speaker diarization."""
    speakers: List[str]
    segments: List[SpeakerSegment]
    speaker_stats: Dict[str, Dict]
    total_duration: float
    processing_time: float


class SpeakerNaming:
    """Interactive workflow for naming speakers based on sample quotes."""

    def __init__(self, transcript_segments: List[Dict], speaker_segments: List[SpeakerSegment],
                 alignment_tolerance: float = 0.5):
        self.transcript_segments = transcript_segments
        self.speaker_segments = speaker_segments
        self.alignment_tolerance = alignment_tolerance

    def merge_transcript_with_speakers(self) -> List[Dict]:
        """Merge transcript segments with speaker information."""
        merged = []

        for seg in self.transcript_segments:
            t_start = seg.get("start", 0)
            t_end = seg.get("end", 0)
            t_duration = t_end - t_start

            assigned_speaker = "Unknown"
            best_quality = 0

            for speaker_seg in self.speaker_segments:
                # Calculate overlap with tolerance
                overlap_start = max(t_start, speaker_seg.start_time - self.alignment_tolerance)
                overlap_end = min(t_end, speaker_seg.end_time + self.alignment_tolerance)
                overlap = max(0, overlap_end - overlap_start)

                if overlap > 0:
                    coverage = overlap / max(0.1, t_duration)
                    quality = overlap * coverage
                    if quality > best_quality:
                        best_quality = quality
                        assigned_speaker = speaker_seg.speaker_id

            result = seg.copy()
            result["speaker"] = assigned_speaker
            merged.append(result)

        return merged

    def get_sample_quotes(self, merged_segments: List[Dict], max_length: int = 150) -> Dict[str, Tuple[str, str]]:
        """Get sample quotes for each speaker for identification."""
        speaker_segs = defaultdict(list)

        for seg in merged_segments:
            text = seg.get("text", "").strip()
            if text:
                speaker_segs[seg.get("speaker", "Unknown")].append(seg)

        quotes = {}
        for speaker_id, segments in speaker_segs.items():
            # Sort by start time, prioritize early segments
            segments.sort(key=lambda s: s.get("start", 0))

            best_seg = None
            best_score = -1

            for seg in segments:
                text = seg.get("text", "").strip()
                start = seg.get("start", 0)
                duration = seg.get("end", start) - start

                # Skip suspicious timestamps
                if start == 0 and len(segments) > 1 and any(s.get("start", 0) > 0 for s in segments):
                    continue

                score = 0
                # Prefer early segments
                if start <= 300:
                    score += 20
                elif start <= 600:
                    score += 10
                # Good length
                if 30 <= len(text) <= max_length:
                    score += 5
                # Good duration
                if 2 <= duration <= 10:
                    score += 3
                # Complete sentence
                if text.endswith(('.', '?', '!')):
                    score += 2

                if score > best_score:
                    best_score = score
                    best_seg = seg

            if best_seg:
                text = best_seg.get("text", "").strip()
                start = best_seg.get("start", 0)
                timestamp = f"{int(start // 60):02d}:{int(start % 60):02d}"
                quotes[speaker_id] = (text[:max_length] + "..." if len(text) > max_length else text, timestamp)

        return quotes

    def prompt_for_names(self, sample_quotes: Dict[str, Tuple[str, str]], skip_interactive: bool = False) -> Dict[str, str]:
        """Prompt user for speaker names."""
        # Sort chronologically
        sorted_speakers = sorted(sample_quotes.keys(),
                                 key=lambda s: int(sample_quotes[s][1].split(':')[0]) * 60 + int(sample_quotes[s][1].split(':')[1]))

        speaker_mapping = {sid: f"Speaker {i+1}" for i, sid in enumerate(sorted_speakers)}

        if skip_interactive:
            return speaker_mapping

        print(f"\nFound {len(sample_quotes)} speakers in this meeting.")
        print("Speakers are in chronological order. Press Enter to keep default name.\n")

        names = {}
        for speaker_id in sorted_speakers:
            quote, timestamp = sample_quotes[speaker_id]
            display_name = speaker_mapping[speaker_id]

            print(f'{display_name} said at [{timestamp}]: "{quote}"')
            user_input = input(f"Name for {display_name}: ").strip()

            names[speaker_id] = user_input if user_input else display_name
            print()

        return names

    def apply_names(self, merged_segments: List[Dict], speaker_names: Dict[str, str]) -> List[Dict]:
        """Apply speaker names to segments."""
        result = []
        for seg in merged_segments:
            updated = seg.copy()
            speaker_id = seg.get("speaker", "Unknown")
            updated["speaker"] = speaker_names.get(speaker_id, speaker_id)
            result.append(updated)
        return result


class SpeakerDiarizationService:
    """Speaker diarization using pyannote-audio, optimized for Apple Silicon."""

    def __init__(self, confidence_threshold: float = 0.7, min_segment_duration: float = 1.0):
        if not PYANNOTE_AVAILABLE:
            raise ImportError("pyannote-audio required. Install with: pip install pyannote-audio")

        self.confidence_threshold = confidence_threshold
        self.min_segment_duration = min_segment_duration
        self.pipeline = None

        # Device selection
        if torch.backends.mps.is_available():
            self.device = "mps"
            logging.info("Using MPS (Metal) for speaker diarization")
        else:
            self.device = "cpu"

    def _load_pipeline(self) -> bool:
        """Load pyannote-audio pipeline."""
        try:
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
            )
            if self.device == "mps":
                self.pipeline.to(torch.device("mps"))
            return True
        except Exception as e:
            logging.error(f"Failed to load diarization pipeline: {e}")
            logging.error("Set HUGGINGFACE_TOKEN and accept model license.")
            return False

    def _optimize_audio(self, audio_path: str) -> str:
        """Optimize audio for diarization."""
        try:
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_channels(1).set_frame_rate(16000).normalize()

            temp_dir = Path(tempfile.gettempdir()) / "plaud_processor"
            temp_dir.mkdir(exist_ok=True)
            out_path = temp_dir / f"diarization_{Path(audio_path).stem}.wav"
            audio.export(str(out_path), format="wav")
            return str(out_path)
        except Exception as e:
            logging.warning(f"Audio optimization failed: {e}")
            return audio_path

    def perform_diarization(self, audio_path: str) -> DiarizationResult:
        """Perform speaker diarization."""
        start_time = time.time()

        try:
            if self.pipeline is None and not self._load_pipeline():
                raise Exception("Pipeline not loaded")

            optimized = self._optimize_audio(audio_path)
            audio = AudioSegment.from_file(audio_path)
            total_duration = len(audio) / 1000.0

            logging.info(f"[SPEAKER_DIARIZATION] Processing {total_duration:.1f}s of audio...")
            diarization = self.pipeline(optimized)

            segments = []
            stats = defaultdict(lambda: {"total_time": 0.0, "segments": 0})

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if turn.duration >= self.min_segment_duration:
                    seg = SpeakerSegment(
                        start_time=turn.start,
                        end_time=turn.end,
                        speaker_id=speaker,
                        confidence=0.8
                    )
                    segments.append(seg)
                    stats[speaker]["total_time"] += seg.duration
                    stats[speaker]["segments"] += 1

            speakers = list(set(s.speaker_id for s in segments))
            for sid in speakers:
                s = stats[sid]
                s["percentage"] = (s["total_time"] / total_duration) * 100
                s["avg_segment_duration"] = s["total_time"] / s["segments"] if s["segments"] else 0

            # Cleanup
            if optimized != audio_path and Path(optimized).exists():
                Path(optimized).unlink()
            gc.collect()
            if self.device == "mps":
                torch.mps.empty_cache()

            logging.info(f"[SPEAKER_DIARIZATION] Found {len(speakers)} speakers, {len(segments)} segments")

            return DiarizationResult(
                speakers=speakers,
                segments=segments,
                speaker_stats=dict(stats),
                total_duration=total_duration,
                processing_time=time.time() - start_time
            )

        except Exception as e:
            logging.error(f"[SPEAKER_DIARIZATION] Failed: {e}")
            return DiarizationResult(
                speakers=[], segments=[], speaker_stats={},
                total_duration=0.0, processing_time=time.time() - start_time
            )
