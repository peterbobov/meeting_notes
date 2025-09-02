#!/usr/bin/env python3
"""
Speaker Diarization Service
Optimized for M2 Mac with pyannote-audio for voice activity detection and speaker clustering
"""

import os
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile
import json
from dataclasses import dataclass

# Audio processing
import torch
from pydub import AudioSegment
import librosa
import soundfile as sf

# Speaker diarization
try:
    from pyannote.audio import Pipeline
    from pyannote.core import Segment, Annotation
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    logging.warning("pyannote-audio not installed. Speaker diarization will be disabled.")

# Utilities
import gc
from collections import defaultdict


@dataclass
class SpeakerSegment:
    """Information about a speaker segment"""
    start_time: float
    end_time: float
    speaker_id: str
    confidence: float
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class SpeakerDiarizationResult:
    """Result from speaker diarization process"""
    speakers: List[str]  # List of unique speaker IDs
    segments: List[SpeakerSegment]  # All speaker segments
    speaker_stats: Dict[str, Dict]  # Statistics per speaker
    total_duration: float
    processing_time: float
    confidence_threshold: float
    optimizations_applied: List[str]


class InteractiveSpeakerNaming:
    """
    Interactive workflow for naming speakers based on sample quotes
    """
    
    def __init__(self, transcript_segments: List[Dict], speaker_segments: List[SpeakerSegment]):
        """
        Initialize interactive speaker naming
        
        Args:
            transcript_segments: List of transcript segments with timestamps
            speaker_segments: List of speaker segments from diarization
        """
        self.transcript_segments = transcript_segments
        self.speaker_segments = speaker_segments
        self.speaker_names = {}
        
    def merge_transcript_with_speakers(self) -> List[Dict]:
        """
        Merge transcript segments with speaker information
        
        Returns:
            List of transcript segments with speaker IDs
        """
        logging.info("[SPEAKER_DIARIZATION] Merging transcript segments with speaker information...")
        
        merged_segments = []
        speaker_assignments = defaultdict(int)
        
        for transcript_seg in self.transcript_segments:
            transcript_start = transcript_seg.get("start", 0)
            transcript_end = transcript_seg.get("end", 0)
            transcript_mid = (transcript_start + transcript_end) / 2
            transcript_duration = transcript_end - transcript_start
            
            # Skip segments with suspicious timestamps (exactly 0 start when others exist)
            if transcript_start == 0 and transcript_end > 0:
                # Check if there are other segments with non-zero starts
                has_non_zero_segments = any(seg.get("start", 0) > 0 for seg in self.transcript_segments)
                if has_non_zero_segments:
                    # This might be a spurious 00:00 segment, be more selective
                    logging.warning(f"[SPEAKER_DIARIZATION] Found suspicious 00:00 segment: '{transcript_seg.get('text', '')[:50]}...'")
            
            # Find the speaker segment that contains this transcript segment
            assigned_speaker = "Unknown"
            best_overlap = 0
            best_match_quality = 0
            
            for speaker_seg in self.speaker_segments:
                # Check if transcript segment overlaps with speaker segment
                overlap_start = max(transcript_start, speaker_seg.start_time)
                overlap_end = min(transcript_end, speaker_seg.end_time)
                overlap_duration = max(0, overlap_end - overlap_start)
                
                # Calculate match quality (prioritize segments that are well contained)
                if overlap_duration > 0:
                    # How much of the transcript segment is covered by this speaker?
                    coverage = overlap_duration / max(0.1, transcript_duration)
                    
                    # Prefer speakers where the transcript is well-contained
                    match_quality = overlap_duration * coverage
                    
                    if match_quality > best_match_quality:
                        best_match_quality = match_quality
                        best_overlap = overlap_duration
                        assigned_speaker = speaker_seg.speaker_id
            
            merged_segment = transcript_seg.copy()
            merged_segment["speaker"] = assigned_speaker
            merged_segments.append(merged_segment)
            
            # Track speaker assignments
            speaker_assignments[assigned_speaker] += 1
        
        # Log speaker assignment summary
        unique_speakers = list(speaker_assignments.keys())
        logging.info(f"[SPEAKER_DIARIZATION] Speaker assignment complete:")
        logging.info(f"[SPEAKER_DIARIZATION] - Total transcript segments: {len(merged_segments)}")
        logging.info(f"[SPEAKER_DIARIZATION] - Unique speakers in transcript: {len(unique_speakers)}")
        
        for speaker_id in sorted(unique_speakers):
            count = speaker_assignments[speaker_id]
            percentage = (count / len(merged_segments)) * 100
            logging.info(f"[SPEAKER_DIARIZATION] - {speaker_id}: {count} segments ({percentage:.1f}%)")
        
        return merged_segments
    
    def get_speaker_sample_quotes(self, merged_segments: List[Dict], max_quote_length: int = 150) -> Dict[str, Tuple[str, str]]:
        """
        Get sample quotes with timestamps for each speaker for identification
        Prioritizes quotes from early in the meeting and validates speaker purity
        
        Args:
            merged_segments: Transcript segments with speaker information
            max_quote_length: Maximum length of sample quote
            
        Returns:
            Dictionary mapping speaker IDs to tuples of (quote, timestamp)
        """
        speaker_segments = defaultdict(list)
        
        # Collect all segments for each speaker, INCLUDING "Unknown" speakers
        for segment in merged_segments:
            speaker_id = segment.get("speaker", "Unknown")
            text = segment.get("text", "").strip()
            start_time = segment.get("start", 0)
            
            # Skip segments with invalid timestamps (00:00 unless actually at start)
            if start_time == 0 and text and len([s for s in merged_segments if s.get("start", 0) > 0]) > 0:
                # Only allow 00:00 if there are no other segments with valid timestamps
                logging.warning(f"[SPEAKER_DIARIZATION] Skipping segment with suspicious 00:00 timestamp: '{text[:50]}...'")
                continue
                
            if text:  # Only require text, include ALL speakers including "Unknown"
                speaker_segments[speaker_id].append(segment)
        
        # Select best sample quote with timestamp for each speaker
        logging.info("[SPEAKER_DIARIZATION] Selecting sample quotes for speaker identification...")
        
        sample_quotes = {}
        for speaker_id, segments in speaker_segments.items():
            if segments:
                logging.info(f"[SPEAKER_DIARIZATION] Processing {len(segments)} segments for {speaker_id}")
                
                # Sort segments by start time to prioritize early segments
                segments_sorted = sorted(segments, key=lambda s: s.get("start", 0))
                
                # Try to find a good representative quote, prioritizing early segments
                best_segment = None
                best_score = -1
                early_segments_count = sum(1 for s in segments if s.get("start", 0) <= 300)  # First 5 minutes
                
                for segment in segments_sorted:
                    text = segment.get("text", "").strip()
                    start_time = segment.get("start", 0)
                    end_time = segment.get("end", start_time)
                    
                    # Check for multi-speaker contamination in this segment
                    if self._is_multi_speaker_segment(segment, speaker_id, merged_segments):
                        logging.debug(f"[SPEAKER_DIARIZATION] Skipping multi-speaker segment for {speaker_id}: '{text[:50]}...'")
                        continue
                    
                    # Base scoring criteria
                    score = 0
                    
                    # Heavily prioritize early segments (first 5 minutes = 300 seconds)
                    if start_time <= 300:  # First 5 minutes
                        score += 20  # Strong preference for early segments
                    elif start_time <= 600:  # First 10 minutes
                        score += 10  # Moderate preference
                    elif start_time <= 1200:  # First 20 minutes
                        score += 5   # Slight preference
                    
                    # Quality scoring
                    if text.endswith('.') or text.endswith('?') or text.endswith('!'):
                        score += 3  # Complete sentence
                    if 30 <= len(text) <= max_quote_length:
                        score += 2  # Good length
                    if len(text) > 50:
                        score += 1  # Substantial content
                    
                    # Bonus for segments with good duration (not too short, not too long)
                    duration = end_time - start_time
                    if 2 <= duration <= 10:  # 2-10 seconds is usually a good single-speaker segment
                        score += 2
                    elif duration > 15:  # Very long segments are more likely multi-speaker
                        score -= 3
                    
                    # Penalize very short text
                    if len(text) < 20:
                        score -= 2
                    
                    # Bonus for segments that don't start at exactly 00:00 (unless it's the very first)
                    if start_time > 0.1:  # Allow small floating point errors
                        score += 1
                    
                    if score > best_score:
                        best_score = score
                        best_segment = segment
                
                if best_segment:
                    text = best_segment.get("text", "").strip()
                    start_time = best_segment.get("start", 0)
                    
                    # Format timestamp as MM:SS
                    minutes = int(start_time // 60)
                    seconds = int(start_time % 60)
                    timestamp = f"{minutes:02d}:{seconds:02d}"
                    
                    # Truncate if needed
                    if len(text) > max_quote_length:
                        text = text[:max_quote_length] + "..."
                    
                    sample_quotes[speaker_id] = (text, timestamp)
                    
                    early_note = f" [NOTE: Only speaks late in meeting]" if early_segments_count == 0 else ""
                    logging.info(f"[SPEAKER_DIARIZATION] - {speaker_id}: Selected quote at {timestamp} (score: {best_score}, early segments: {early_segments_count}){early_note}")
        
        logging.info(f"[SPEAKER_DIARIZATION] Sample quote selection complete: {len(sample_quotes)} speakers will be asked for names")
        
        return sample_quotes
    
    def _is_multi_speaker_segment(self, target_segment: Dict, target_speaker: str, all_segments: List[Dict]) -> bool:
        """
        Check if a transcript segment likely contains speech from multiple speakers
        
        Args:
            target_segment: The segment to check
            target_speaker: The speaker ID we're checking for
            all_segments: All merged segments to check for overlaps
            
        Returns:
            True if the segment likely contains multiple speakers
        """
        target_start = target_segment.get("start", 0)
        target_end = target_segment.get("end", target_start)
        target_duration = target_end - target_start
        
        # Very long segments (>15 seconds) are suspicious for single speaker
        if target_duration > 15:
            return True
            
        # Count how many different speakers have segments that overlap with this one
        overlapping_speakers = set()
        
        for segment in all_segments:
            if segment == target_segment:
                continue
                
            seg_start = segment.get("start", 0)
            seg_end = segment.get("end", seg_start)
            seg_speaker = segment.get("speaker", "Unknown")
            
            # Check for temporal overlap
            overlap_start = max(target_start, seg_start)
            overlap_end = min(target_end, seg_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            # If there's significant overlap with a different speaker, it's suspicious
            if overlap_duration > 0.5 and seg_speaker != target_speaker:  # 0.5 second overlap threshold
                overlapping_speakers.add(seg_speaker)
        
        # If multiple other speakers overlap significantly, this segment is contaminated
        return len(overlapping_speakers) >= 1
    
    def prompt_for_speaker_names(self, sample_quotes: Dict[str, Tuple[str, str]], skip_interactive: bool = False) -> Dict[str, str]:
        """
        Prompt user to provide names for speakers
        
        Args:
            sample_quotes: Sample quotes with timestamps for each speaker
            skip_interactive: If True, keep default speaker names
            
        Returns:
            Dictionary mapping speaker IDs to user-provided names
        """
        # Sort speakers chronologically by their first appearance (earliest quotes first)
        def get_timestamp_seconds(speaker_id):
            _, timestamp_str = sample_quotes[speaker_id]
            # Convert MM:SS to seconds for proper sorting
            try:
                minutes, seconds = map(int, timestamp_str.split(':'))
                return minutes * 60 + seconds
            except (ValueError, AttributeError):
                return float('inf')  # Put any malformed timestamps at the end
        
        sorted_speakers = sorted(sample_quotes.keys(), key=get_timestamp_seconds)
        
        logging.info(f"[SPEAKER_DIARIZATION] Speaker chronological order: {[(sid, sample_quotes[sid][1]) for sid in sorted_speakers]}")
        
        # Convert speaker IDs to progressive numbered names based on chronological order
        speaker_mapping = {}
        
        for idx, speaker_id in enumerate(sorted_speakers):
            speaker_number = idx + 1  # Start from 1 instead of 0
            speaker_mapping[speaker_id] = f"Speaker {speaker_number}"
        
        if skip_interactive:
            logging.info("Skipping interactive speaker naming, keeping default names")
            return speaker_mapping
        
        logging.info(f"[SPEAKER_DIARIZATION] Starting interactive speaker naming for {len(sample_quotes)} speakers")
        
        print(f"\nFound {len(sample_quotes)} speakers in this meeting.")
        print("Please provide names for each speaker (press Enter to keep default name):")
        print("Speakers are presented in chronological order (first to speak → last to speak).")
        print("Note: If you enter the same name for multiple speakers, they will be merged.\n")
        
        speaker_names = {}
        entered_names = {}  # Track which names have been entered
        
        for speaker_id in sorted_speakers:
            quote, timestamp = sample_quotes[speaker_id]
            display_name = speaker_mapping[speaker_id]
            
            print(f"{display_name} said at [{timestamp}]: \"{quote}\"")
            user_input = input(f"Please enter name for {display_name}: ").strip()
            
            if user_input:
                # Check if this name was already entered
                if user_input in entered_names.values():
                    # Find the first speaker with this name
                    original_speaker = None
                    for sid, name in entered_names.items():
                        if name == user_input:
                            original_speaker = sid
                            break
                    
                    # Map this speaker to the same name as the original
                    speaker_names[speaker_id] = user_input
                    print(f"  → Will merge with previously named '{user_input}'")
                    logging.info(f"[SPEAKER_DIARIZATION] {speaker_id} ({display_name}) -> '{user_input}' (merged)")
                else:
                    speaker_names[speaker_id] = user_input
                    entered_names[speaker_id] = user_input
                    logging.info(f"[SPEAKER_DIARIZATION] {speaker_id} ({display_name}) -> '{user_input}'")
            else:
                speaker_names[speaker_id] = display_name
                entered_names[speaker_id] = display_name
                logging.info(f"[SPEAKER_DIARIZATION] {speaker_id} -> kept default name '{display_name}'")
            
            print()  # Add blank line for readability
        
        logging.info(f"[SPEAKER_DIARIZATION] Interactive speaker naming complete: {len(set(speaker_names.values()))} unique names assigned")
        
        return speaker_names
    
    def apply_speaker_names(self, merged_segments: List[Dict], speaker_names: Dict[str, str]) -> List[Dict]:
        """
        Apply user-provided speaker names to transcript segments
        
        Args:
            merged_segments: Transcript segments with speaker IDs
            speaker_names: Mapping of speaker IDs to names
            
        Returns:
            Updated transcript segments with speaker names
        """
        logging.info("[SPEAKER_DIARIZATION] Applying speaker names to transcript segments...")
        
        updated_segments = []
        final_speaker_counts = defaultdict(int)
        unmapped_speakers = set()
        
        for segment in merged_segments:
            updated_segment = segment.copy()
            speaker_id = segment.get("speaker", "Unknown")
            speaker_name = speaker_names.get(speaker_id, speaker_id)
            
            if speaker_id not in speaker_names and speaker_id != "Unknown":
                unmapped_speakers.add(speaker_id)
                
            updated_segment["speaker"] = speaker_name
            updated_segments.append(updated_segment)
            
            # Count final speaker assignments
            final_speaker_counts[speaker_name] += 1
        
        # Store names for future reference
        self.speaker_names = speaker_names
        
        # Log final results
        logging.info(f"[SPEAKER_DIARIZATION] Speaker name application complete:")
        logging.info(f"[SPEAKER_DIARIZATION] - Total segments updated: {len(updated_segments)}")
        logging.info(f"[SPEAKER_DIARIZATION] - Final speaker distribution:")
        
        for speaker_name in sorted(final_speaker_counts.keys()):
            count = final_speaker_counts[speaker_name]
            percentage = (count / len(updated_segments)) * 100
            logging.info(f"[SPEAKER_DIARIZATION] - {speaker_name}: {count} segments ({percentage:.1f}%)")
        
        if unmapped_speakers:
            logging.warning(f"[SPEAKER_DIARIZATION] Warning: {len(unmapped_speakers)} speakers not mapped: {list(unmapped_speakers)}")
        
        # Check for remaining Unknown speakers
        if "Unknown" in final_speaker_counts:
            unknown_count = final_speaker_counts["Unknown"]
            logging.warning(f"[SPEAKER_DIARIZATION] Warning: {unknown_count} segments still assigned to 'Unknown' speaker")
        else:
            logging.info("[SPEAKER_DIARIZATION] Success: All speakers have been identified and named")
        
        return updated_segments


class SpeakerDiarizationService:
    """
    Speaker diarization service using pyannote-audio
    Optimized for M2 Mac with Metal Performance Shaders
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.7,
                 min_segment_duration: float = 1.0,
                 device: Optional[str] = None):
        """
        Initialize speaker diarization service
        
        Args:
            confidence_threshold: Minimum confidence for speaker segments
            min_segment_duration: Minimum duration for speaker segments (seconds)
            device: Device to use (auto-detected if None)
        """
        if not PYANNOTE_AVAILABLE:
            raise ImportError("pyannote-audio is required for speaker diarization. Install with: pip install pyannote-audio")
        
        self.confidence_threshold = confidence_threshold
        self.min_segment_duration = min_segment_duration
        self.pipeline = None
        self.optimizations_applied = []
        
        # Device configuration for M2 Mac
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"  # Metal Performance Shaders for M2
                logging.info("Using MPS (Metal) acceleration for speaker diarization")
            else:
                self.device = "cpu"
                logging.info("Using CPU for speaker diarization (MPS not available)")
        else:
            self.device = device
            
        logging.info(f"Initialized SpeakerDiarizationService with device '{self.device}'")
        
    def load_pipeline(self) -> bool:
        """Load pyannote-audio pipeline"""
        try:
            logging.info("[SPEAKER_DIARIZATION] Loading pyannote-audio pipeline...")
            
            # Note: This requires a HuggingFace token for the pretrained model
            # Users will need to accept the user agreement and provide a token
            start_time = time.time()
            
            # Load the speaker diarization pipeline
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
            )
            
            # Move to the appropriate device
            if self.device == "mps":
                self.pipeline.to(torch.device("mps"))
            
            load_time = time.time() - start_time
            logging.info(f"[SPEAKER_DIARIZATION] Pipeline loaded successfully in {load_time:.2f}s")
            return True
            
        except Exception as e:
            logging.error(f"[SPEAKER_DIARIZATION] Failed to load pipeline: {e}")
            logging.error("Note: Speaker diarization requires a HuggingFace token and model access.")
            logging.error("Please set HUGGINGFACE_TOKEN environment variable and accept the model license.")
            return False
    
    def optimize_audio_for_diarization(self, audio_path: str) -> Tuple[str, List[str]]:
        """
        Optimize audio for speaker diarization
        
        Args:
            audio_path: Path to input audio file
            
        Returns:
            Tuple of (optimized_audio_path, optimizations_applied)
        """
        optimizations = []
        
        try:
            logging.info(f"[SPEAKER_DIARIZATION] Optimizing audio for diarization: {Path(audio_path).name}")
            
            # Load audio
            audio = AudioSegment.from_file(audio_path)
            
            # Convert to mono if stereo
            if audio.channels > 1:
                logging.info("[SPEAKER_DIARIZATION] Converting to mono...")
                audio = audio.set_channels(1)
                optimizations.append("converted_to_mono")
            
            # Normalize sample rate (16kHz is good for speaker diarization)
            if audio.frame_rate != 16000:
                logging.info("[SPEAKER_DIARIZATION] Resampling to 16kHz...")
                audio = audio.set_frame_rate(16000)
                optimizations.append("resampled_to_16khz")
            
            # Normalize audio levels
            audio = audio.normalize()
            optimizations.append("normalized_levels")
            
            # Export optimized audio
            temp_dir = Path(tempfile.gettempdir()) / "plaud_processor"
            temp_dir.mkdir(exist_ok=True)
            
            optimized_path = temp_dir / f"diarization_{Path(audio_path).stem}.wav"
            audio.export(str(optimized_path), format="wav")
            
            logging.info(f"[SPEAKER_DIARIZATION] Audio optimized: {optimizations}")
            return str(optimized_path), optimizations
            
        except Exception as e:
            logging.warning(f"[SPEAKER_DIARIZATION] Audio optimization failed: {e}")
            return audio_path, ["optimization_failed"]
    
    def perform_diarization(self, audio_path: str) -> SpeakerDiarizationResult:
        """
        Perform speaker diarization on audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            SpeakerDiarizationResult with speaker information
        """
        start_time = time.time()
        
        try:
            logging.info(f"[SPEAKER_DIARIZATION] Starting diarization process for {Path(audio_path).name}")
            
            # Ensure pipeline is loaded
            if self.pipeline is None:
                if not self.load_pipeline():
                    raise Exception("Failed to load diarization pipeline")
            
            # Optimize audio
            optimized_audio, optimizations = self.optimize_audio_for_diarization(audio_path)
            self.optimizations_applied = optimizations
            
            # Get audio duration
            audio = AudioSegment.from_file(audio_path)
            total_duration = len(audio) / 1000.0  # Convert to seconds
            
            logging.info(f"[SPEAKER_DIARIZATION] Processing {total_duration:.1f}s of audio...")
            
            # Perform diarization
            diarization_start = time.time()
            diarization = self.pipeline(optimized_audio)
            diarization_time = time.time() - diarization_start
            
            # Process results
            speaker_segments = []
            speaker_stats = defaultdict(lambda: {"total_time": 0.0, "segments": 0})
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Filter segments by duration and confidence
                if turn.duration >= self.min_segment_duration:
                    segment = SpeakerSegment(
                        start_time=turn.start,
                        end_time=turn.end,
                        speaker_id=speaker,
                        confidence=0.8  # pyannote doesn't provide confidence scores directly
                    )
                    speaker_segments.append(segment)
                    
                    # Update statistics
                    speaker_stats[speaker]["total_time"] += segment.duration
                    speaker_stats[speaker]["segments"] += 1
            
            # Get unique speakers
            speakers = list(set(seg.speaker_id for seg in speaker_segments))
            
            # Calculate speaker statistics
            for speaker_id in speakers:
                stats = speaker_stats[speaker_id]
                stats["percentage"] = (stats["total_time"] / total_duration) * 100
                stats["avg_segment_duration"] = stats["total_time"] / stats["segments"] if stats["segments"] > 0 else 0
            
            # Clean up temporary files
            if optimized_audio != audio_path and Path(optimized_audio).exists():
                Path(optimized_audio).unlink()
            
            # Force garbage collection
            gc.collect()
            if self.device == "mps":
                torch.mps.empty_cache()
            
            processing_time = time.time() - start_time
            
            logging.info(f"[SPEAKER_DIARIZATION] Diarization completed:")
            logging.info(f"[SPEAKER_DIARIZATION] - Processing time: {processing_time:.2f}s")
            logging.info(f"[SPEAKER_DIARIZATION] - Diarization time: {diarization_time:.2f}s")
            logging.info(f"[SPEAKER_DIARIZATION] - Speakers detected: {len(speakers)}")
            logging.info(f"[SPEAKER_DIARIZATION] - Total segments: {len(speaker_segments)}")
            
            for speaker_id, stats in speaker_stats.items():
                logging.info(f"[SPEAKER_DIARIZATION] - {speaker_id}: {stats['percentage']:.1f}% ({stats['total_time']:.1f}s)")
            
            return SpeakerDiarizationResult(
                speakers=speakers,
                segments=speaker_segments,
                speaker_stats=dict(speaker_stats),
                total_duration=total_duration,
                processing_time=processing_time,
                confidence_threshold=self.confidence_threshold,
                optimizations_applied=self.optimizations_applied
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logging.error(f"[SPEAKER_DIARIZATION] Diarization failed after {processing_time:.2f}s: {e}")
            
            # Return empty result
            return SpeakerDiarizationResult(
                speakers=[],
                segments=[],
                speaker_stats={},
                total_duration=0.0,
                processing_time=processing_time,
                confidence_threshold=self.confidence_threshold,
                optimizations_applied=["diarization_failed"]
            )
    
    def get_service_info(self) -> Dict:
        """Get information about the diarization service"""
        return {
            "backend": "pyannote-audio",
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "min_segment_duration": self.min_segment_duration,
            "pipeline_loaded": self.pipeline is not None,
            "optimizations_applied": self.optimizations_applied
        }


def test_speaker_diarization(audio_file: str = None):
    """Test function for speaker diarization"""
    logging.basicConfig(level=logging.INFO)
    
    if not PYANNOTE_AVAILABLE:
        print("pyannote-audio not available. Please install with: pip install pyannote-audio")
        return
    
    # Initialize service
    service = SpeakerDiarizationService(
        confidence_threshold=0.7,
        min_segment_duration=1.0
    )
    
    # Print service info
    info = service.get_service_info()
    print("=== Speaker Diarization Service Info ===")
    for key, value in info.items():
        print(f"{key}: {value}")
    print()
    
    if audio_file and Path(audio_file).exists():
        print(f"Testing diarization with: {audio_file}")
        result = service.perform_diarization(audio_file)
        
        print("=== Diarization Result ===")
        print(f"Speakers detected: {len(result.speakers)}")
        print(f"Total segments: {len(result.segments)}")
        print(f"Processing time: {result.processing_time:.2f}s")
        print(f"Total duration: {result.total_duration:.1f}s")
        
        print("\n=== Speaker Statistics ===")
        for speaker_id, stats in result.speaker_stats.items():
            print(f"{speaker_id}:")
            print(f"  - Speaking time: {stats['total_time']:.1f}s ({stats['percentage']:.1f}%)")
            print(f"  - Segments: {stats['segments']}")
            print(f"  - Average segment: {stats['avg_segment_duration']:.1f}s")
        
        print("\n=== Speaker Timeline ===")
        for i, segment in enumerate(result.segments[:10]):  # Show first 10 segments
            print(f"[{segment.start_time:.1f}s-{segment.end_time:.1f}s] {segment.speaker_id}")
            if i == 9 and len(result.segments) > 10:
                print(f"... and {len(result.segments) - 10} more segments")
    else:
        print("No audio file provided for testing")
        print("Usage: python speaker_diarization.py /path/to/audio/file.mp3")


if __name__ == "__main__":
    import sys
    audio_file = sys.argv[1] if len(sys.argv) > 1 else None
    test_speaker_diarization(audio_file)