#!/usr/bin/env python3
"""
Plaud Pin to Obsidian Processing Pipeline
Processes audio recordings from Plaud Pin device, transcribes using OpenAI Whisper,
and generates structured Markdown files for Obsidian import.
"""

import os
import json
import datetime
from pathlib import Path
import configparser
import logging
from typing import Dict, List, Optional
import time
import hashlib
import shutil
import re
import sys
import threading

# Audio processing
import librosa
from pydub import AudioSegment

# AI APIs
from openai import OpenAI

# Local transcription
from local_transcription import LocalTranscriptionService, TranscriptionResult

# whisper.cpp backend
from whisper_cpp_backend import WhisperCppService

# Environment
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class TranscriptionProgressBar:
    """
    Progress bar for transcription tasks with time estimation
    """
    
    def __init__(self, total_duration_seconds: float, description: str = "Transcribing"):
        self.total_duration = total_duration_seconds
        self.description = description
        self.start_time = time.time()
        self.current_progress = 0.0
        self.is_running = False
        self.last_update = 0
        self.bar_width = 40
        
    def start(self):
        """Start the progress bar"""
        self.start_time = time.time()
        self.is_running = True
        self.current_progress = 0.0
        
    def update(self, current_seconds: float):
        """
        Update progress based on current timestamp in the audio
        
        Args:
            current_seconds: Current position in the audio file (in seconds)
        """
        if not self.is_running:
            return
            
        self.current_progress = min(current_seconds / self.total_duration, 1.0) if self.total_duration > 0 else 0.0
        
        # Only update display every 2 seconds to avoid spam
        current_time = time.time()
        if current_time - self.last_update >= 2.0 or self.current_progress >= 1.0:
            self._display_progress()
            self.last_update = current_time
    
    def _display_progress(self):
        """Display the current progress bar"""
        if not self.is_running:
            return
            
        # Calculate time estimates
        elapsed_time = time.time() - self.start_time
        if self.current_progress > 0:
            estimated_total = elapsed_time / self.current_progress
            remaining_time = max(0, estimated_total - elapsed_time)
        else:
            remaining_time = 0
        
        # Create progress bar
        filled_width = int(self.bar_width * self.current_progress)
        bar = '█' * filled_width + '░' * (self.bar_width - filled_width)
        
        # Format time
        remaining_str = self._format_time(remaining_time)
        
        # Create progress line
        percentage = self.current_progress * 100
        progress_line = f"\r{self.description}: {bar} {percentage:.0f}% ({remaining_str} remaining)"
        
        # Print without newline, flush immediately
        print(progress_line, end='', file=sys.stderr, flush=True)
    
    def finish(self, success: bool = True):
        """Complete the progress bar"""
        if not self.is_running:
            return
            
        self.is_running = False
        self.current_progress = 1.0
        
        if success:
            # Show completed bar
            bar = '█' * self.bar_width
            elapsed = time.time() - self.start_time
            elapsed_str = self._format_time(elapsed)
            final_line = f"\r{self.description}: {bar} 100% (completed in {elapsed_str})"
            print(final_line, file=sys.stderr)
        else:
            # Show failed bar
            print(f"\r{self.description}: Failed after {self._format_time(time.time() - self.start_time)}", file=sys.stderr)
    
    def _format_time(self, seconds: float) -> str:
        """Format time in MM:SS format"""
        if seconds < 60:
            return f"{int(seconds)}s"
        else:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}:{secs:02d}"


class AudioProcessor:
    """Handles audio file processing, validation, and monitoring."""
    
    def __init__(self, config):
        self.config = config
        self.supported_formats = config['SETTINGS']['supported_formats'].split(',')
        self.max_size_mb = int(config['SETTINGS']['max_file_size_mb'])
        
        # Get single input folder with environment variable substitution
        input_folder_str = config['PATHS'].get('input_folder', './input_audio')
        
        # Replace ${VAR_NAME} with environment variable values
        if '${' in input_folder_str and '}' in input_folder_str:
            import re
            env_vars = re.findall(r'\$\{([^}]+)\}', input_folder_str)
            for env_var in env_vars:
                env_value = os.getenv(env_var, '')
                if env_value:
                    input_folder_str = input_folder_str.replace(f'${{{env_var}}}', env_value)
                else:
                    raise ValueError(f"Environment variable {env_var} not found. Please set {env_var} in your .env file")
            
        self.input_folder = Path(input_folder_str).expanduser()
        self.processed_folder = Path(config['PATHS']['processed_folder'])
        
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
            
            # Check if file exists
            if not file_path.exists():
                return False
                
            # Check file size
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb > self.max_size_mb:
                logging.warning(f"File {file_path} exceeds size limit: {size_mb:.1f}MB")
                return False
                
            # Check if it's a valid audio file by trying to load it
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


class HybridTranscriptionService:
    """Hybrid transcription service that can use either local Whisper or OpenAI API."""
    
    def __init__(self, api_key: str, use_local: bool = True, local_model: str = "medium"):
        self.api_key = api_key
        self.use_local = use_local
        self.local_model = local_model
        
        if use_local:
            # Use whisper.cpp backend for better M2 compatibility
            self.local_service = WhisperCppService(
                model_size=local_model
            )
            logging.info(f"Initialized whisper.cpp transcription with model '{local_model}'") 
        else:
            self.client = OpenAI(api_key=api_key)
            logging.info("Initialized OpenAI API transcription service")
        
    def transcribe_audio(self, file_path: str, language: Optional[str] = None) -> Dict:
        """
        Transcribe audio using either local Whisper or OpenAI API.
        Returns: {
            'text': full_transcript,
            'segments': timestamped_segments,
            'duration': duration_seconds,
            'language': detected_language,
            'processing_info': additional_processing_information
        }
        """
        if self.use_local:
            return self._transcribe_local(file_path, language)
        else:
            return self._transcribe_api(file_path, language)
    
    def _transcribe_local(self, file_path: str, language: Optional[str] = None) -> Dict:
        """Transcribe using local Whisper model."""
        try:
            result = self.local_service.transcribe_audio(file_path, language)
            
            # Convert TranscriptionResult to expected format
            return {
                'text': result.text,
                'segments': result.segments,
                'duration': result.duration,
                'language': result.language,
                'processing_info': {
                    'method': 'local_whisper',
                    'model': result.model_used,
                    'processing_time': result.processing_time,
                    'optimizations': result.optimizations_applied,
                    'confidence_scores': result.confidence_scores
                }
            }
            
        except Exception as e:
            logging.error(f"Local transcription failed: {e}")
            # Fallback to API if local fails
            if hasattr(self, 'api_key'):
                logging.info("Falling back to OpenAI API...")
                return self._transcribe_api(file_path, language)
            else:
                return {
                    'text': '',
                    'segments': [],
                    'duration': 0,
                    'language': 'unknown',
                    'error': str(e),
                    'processing_info': {'method': 'local_whisper_failed'}
                }
    
    def _transcribe_api(self, file_path: str, language: Optional[str] = None) -> Dict:
        """Transcribe using OpenAI Whisper API."""
        try:
            if not hasattr(self, 'client'):
                self.client = OpenAI(api_key=self.api_key)
                
            with open(file_path, "rb") as audio_file:
                # Build transcription parameters
                transcribe_params = {
                    "model": "whisper-1",
                    "file": audio_file,
                    "response_format": "verbose_json",
                    "timestamp_granularities": ["segment"]
                }
                
                # Add language if specified
                if language:
                    transcribe_params["language"] = language
                    
                # Transcribe with timestamps
                transcript = self.client.audio.transcriptions.create(**transcribe_params)
                
            return {
                'text': transcript.text,
                'segments': transcript.segments if hasattr(transcript, 'segments') else [],
                'duration': transcript.duration if hasattr(transcript, 'duration') else 0,
                'language': transcript.language if hasattr(transcript, 'language') else 'unknown',
                'processing_info': {
                    'method': 'openai_api',
                    'model': 'whisper-1'
                }
            }
            
        except Exception as e:
            logging.error(f"Error transcribing audio {file_path}: {e}")
            return {
                'text': '',
                'segments': [],
                'duration': 0,
                'language': 'unknown',
                'error': str(e),
                'processing_info': {'method': 'api_failed'}
            }
            
    def format_transcript_with_timestamps(self, segments: List, include_confidence: bool = False) -> str:
        """Format segments with timestamps for markdown."""
        if not segments:
            return "No timestamped segments available."
            
        # Use local service formatting if available and requested
        if self.use_local and hasattr(self, 'local_service') and include_confidence:
            return self.local_service.format_transcript_with_timestamps(segments, include_confidence)
            
        formatted_lines = []
        for segment in segments:
            # Handle both dict and object formats
            if hasattr(segment, 'start'):
                start_time = self._format_time(getattr(segment, 'start', 0))
                text = getattr(segment, 'text', '').strip()
            else:
                start_time = self._format_time(segment.get('start', 0))
                text = segment.get('text', '').strip()
            formatted_lines.append(f"[{start_time}] {text}")
            
        return '\n'.join(formatted_lines)
        
    def _format_time(self, seconds: float) -> str:
        """Format time in MM:SS format."""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
        
    def get_transcription_info(self) -> Dict:
        """Get information about the transcription service configuration."""
        if self.use_local and hasattr(self, 'local_service'):
            return {
                'mode': 'local',
                'model_info': self.local_service.get_model_info()
            }
        else:
            return {
                'mode': 'api',
                'model': 'whisper-1'
            }


class AIProcessor:
    """Handles AI processing for summaries and action items using GPT-4o."""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        
    def generate_summary(self, transcript: str, prompt: str, context: Optional[str] = None, target_language: Optional[str] = None) -> str:
        """Generate meeting summary using GPT-4o."""
        try:
            # Create the user message with optional context and language specification
            user_message = "Please analyze this meeting transcript and create a summary:"
            if context:
                user_message = f"Context: {context}\n\nPlease analyze this meeting transcript and create a summary based on the provided context:"
            if target_language:
                user_message += f"\n\nIMPORTANT: Please write the summary in {target_language}, regardless of the transcript language."
            user_message += f"\n\n{transcript}"
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logging.error(f"Error generating summary: {e}")
            return f"Error generating summary: {e}"
            
    def extract_action_items(self, transcript: str, prompt: str, target_language: Optional[str] = None) -> str:
        """Extract and format action items using GPT-4o."""
        try:
            user_message = f"Please extract action items from this meeting transcript:\n\n{transcript}"
            if target_language:
                user_message += f"\n\nIMPORTANT: Please write the action items in {target_language}, regardless of the transcript language."
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=800,
                temperature=0.2
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logging.error(f"Error extracting action items: {e}")
            return f"Error extracting action items: {e}"
            
    def generate_meeting_title(self, transcript: str, prompt: str) -> str:
        """Generate appropriate meeting title from content."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Generate a meeting title for this transcript:\n\n{transcript[:1000]}..."}
                ],
                max_tokens=50,
                temperature=0.3
            )
            return response.choices[0].message.content.strip().strip('"\'')
            
        except Exception as e:
            logging.error(f"Error generating title: {e}")
            return "Meeting Recording"
            
    def detect_participants(self, transcript: str, prompt: str) -> List[str]:
        """Attempt to identify meeting participants."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Identify participants in this transcript:\n\n{transcript}"}
                ],
                max_tokens=200,
                temperature=0.2
            )
            
            participants_text = response.choices[0].message.content.strip()
            participants = [p.strip() for p in participants_text.split(',') if p.strip()]
            return participants
            
        except Exception as e:
            logging.error(f"Error detecting participants: {e}")
            return ["Unknown"]
    
    def translate_transcript(self, transcript: str, target_language: str = "Russian") -> str:
        """Translate transcript to target language while preserving timestamps and names."""
        try:
            prompt = f"""Translate this meeting transcript to {target_language}. 
            
IMPORTANT RULES:
1. Keep ALL timestamps exactly as they are: [MM:SS]
2. Preserve proper names (like "Petia Bobov") exactly as written
3. Translate only the spoken content, not the names or timestamps
4. Maintain the same format and structure
5. Ensure natural, fluent {target_language} translation

Original transcript:"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a professional translator specializing in meeting transcripts. You preserve names and timestamps while providing accurate translations."},
                    {"role": "user", "content": f"{prompt}\n\n{transcript}"}
                ],
                max_tokens=4000,
                temperature=0.1
            )
            
            translated_text = response.choices[0].message.content.strip()
            logging.info(f"Successfully translated transcript to {target_language}")
            return translated_text
            
        except Exception as e:
            logging.error(f"Error translating transcript to {target_language}: {e}")
            return transcript  # Return original if translation fails


class ObsidianGenerator:
    """Generates structured markdown files for Obsidian import."""
    
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    def create_meeting_folder(self, meeting_title: str, date: str) -> Path:
        """Create folder structure for meeting."""
        # Clean title for folder name
        clean_title = re.sub(r'[^\w\s-]', '', meeting_title).strip()
        clean_title = re.sub(r'[-\s]+', '-', clean_title)
        
        folder_name = f"{date}_{clean_title}"
        folder_path = self.output_path / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        
        return folder_path
        
    def generate_meta_file(self, meeting_data: Dict, include_action_items: bool = True) -> str:
        """Generate meeting_meta.md content."""
        navigation_links = [
            "- [[2_meeting_transcript]] - Full transcript with timestamps",
            "- [[3_meeting_summary]] - Key points and decisions"
        ]
        
        if include_action_items:
            navigation_links.append("- [[4_meeting_action_items]] - Tasks and follow-ups")
        
        navigation_section = "\n".join(navigation_links)
        
        template = f"""# Meeting: {meeting_data['title']}

**Date:** {meeting_data['date']}
**Duration:** {meeting_data['duration']} minutes
**Participants:** {', '.join(meeting_data['participants'])}
**Language:** {meeting_data.get('language', 'unknown').title()}
**Status:** #meeting/processed

## Quick Navigation
{navigation_section}

## Meeting Overview
{meeting_data.get('overview', 'Auto-generated meeting recording processed for analysis and action item tracking.')}

---
*Processed on: {meeting_data['processed_timestamp']}*
*Audio file: {meeting_data['original_filename']}*"""
        return template
        
    def generate_transcript_file(self, transcript_data: Dict) -> str:
        """Generate meeting_transcript.md content."""
        template = f"""# Meeting Transcript - {transcript_data['title']}

**Date:** {transcript_data['date']}
**Duration:** {transcript_data['duration_formatted']}
**Tags:** #transcript #meeting

## Transcript

{transcript_data['formatted_transcript']}

---
**Related:** [[1_meeting_meta]] | [[3_meeting_summary]] | [[4_meeting_action_items]]
"""
        return template
        
    def generate_summary_file(self, summary_data: Dict) -> str:
        """Generate meeting_summary.md content."""
        template = f"""# Meeting Summary - {summary_data['title']}

**Date:** {summary_data['date']}
**Tags:** #summary #meeting

{summary_data['summary_content']}

---
**Related:** [[1_meeting_meta]] | [[2_meeting_transcript]] | [[4_meeting_action_items]]
"""
        return template
        
    def generate_action_items_file(self, action_items_data: Dict) -> str:
        """Generate meeting_action_items.md content."""
        template = f"""# Action Items - {action_items_data['title']}

**Date:** {action_items_data['date']}
**Tags:** #actionitems #meeting #tasks

{action_items_data['action_items_content']}

---
**Related:** [[1_meeting_meta]] | [[2_meeting_transcript]] | [[3_meeting_summary]]
"""
        return template
        
    def write_files(self, folder_path: Path, file_contents: Dict, write_action_items: bool = True):
        """Write all markdown files to the meeting folder."""
        files_to_write = {
            '1_meeting_meta.md': file_contents['meta'],
            '2_meeting_transcript.md': file_contents['transcript'],
            '3_meeting_summary.md': file_contents['summary']
        }
        
        # Only include action items file if requested
        if write_action_items and 'action_items' in file_contents:
            files_to_write['4_meeting_action_items.md'] = file_contents['action_items']
        
        for filename, content in files_to_write.items():
            file_path = folder_path / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
        logging.info(f"Created meeting files in: {folder_path}")


class PlaudProcessor:
    """Main processor pipeline for Plaud Pin recordings."""
    
    def __init__(self, config_path: str = "config.ini", custom_summary_prompt: Optional[str] = None, \
                 context: Optional[str] = None, generate_action_items: bool = True, \
                 summary_language: Optional[str] = None, transcription_language: Optional[str] = None, \
                 whisper_model: Optional[str] = None, verbose: bool = False):
        self.config = self.load_config(config_path)
        self.custom_summary_prompt = custom_summary_prompt
        self.context = context
        self.generate_action_items = generate_action_items
        self.summary_language = summary_language
        self.transcription_language = transcription_language
        self.whisper_model = whisper_model
        self.verbose = verbose
        self.setup_logging()
        
        # Initialize components
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")
            
        self.audio_processor = AudioProcessor(self.config)
        
        # Use local transcription by default, with API fallback
        use_local = self.config.getboolean('SETTINGS', 'use_local_transcription', fallback=True)
        
        # CLI model override takes precedence, then config
        local_model = self.whisper_model
        if not local_model:
            local_model = self.config.get('SETTINGS', 'local_whisper_model', fallback='medium')
        
        if local_model:
            logging.info(f"Using Whisper model: {local_model}")
        
        self.transcription_service = HybridTranscriptionService(
            api_key=api_key,
            use_local=use_local,
            local_model=local_model
        )
        self.ai_processor = AIProcessor(api_key)
        
        # Handle environment variable substitution for output folder
        output_folder_str = self.config['PATHS'].get('output_folder', './meeting_data')
        if '${' in output_folder_str and '}' in output_folder_str:
            import re
            env_vars = re.findall(r'\$\{([^}]+)\}', output_folder_str)
            for env_var in env_vars:
                env_value = os.getenv(env_var, '')
                if env_value:
                    output_folder_str = output_folder_str.replace(f'${{{env_var}}}', env_value)
                else:
                    # Fallback to default if environment variable not set
                    output_folder_str = './meeting_data'
                    logging.warning(f"Environment variable {env_var} not found. Using default: {output_folder_str}")
        
        self.obsidian_generator = ObsidianGenerator(output_folder_str)
        
        # Create input folder if it doesn't exist
        self.audio_processor.input_folder.mkdir(parents=True, exist_ok=True)
        logging.info(f"Input folder ready: {self.audio_processor.input_folder}")
        
    def load_config(self, config_path: str) -> configparser.ConfigParser:
        """Load configuration from file."""
        config = configparser.ConfigParser()
        
        # Try to load existing config
        if Path(config_path).exists():
            config.read(config_path)
        else:
            # Use example config as template
            example_path = config_path + '.example'
            if Path(example_path).exists():
                config.read(example_path)
                logging.warning(f"Using example config. Copy {example_path} to {config_path} and customize.")
            else:
                raise FileNotFoundError(f"Config file not found: {config_path}")
                
        return config
        
    def setup_logging(self):
        """Setup logging configuration based on verbose mode."""
        logs_folder = Path(self.config['PATHS']['logs_folder'])
        logs_folder.mkdir(parents=True, exist_ok=True)
        
        log_file = logs_folder / f"plaud_processor_{datetime.date.today().strftime('%Y%m%d')}.log"
        
        # Determine log level based on verbose flag
        console_level = logging.DEBUG if self.verbose else logging.INFO
        
        # Clear any existing handlers to avoid duplicates
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Setup logging with appropriate level
        logging.basicConfig(
            level=logging.DEBUG,  # Always log everything to file
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        # Set console handler to appropriate level
        console_handler = logging.getLogger().handlers[-1]  # StreamHandler is last
        console_handler.setLevel(console_level)
        
        if not self.verbose:
            # In non-verbose mode, create a custom filter for cleaner output
            console_handler.addFilter(self._clean_output_filter)
    
    def _clean_output_filter(self, record):
        """
        Filter log records for clean, non-verbose output.
        Only shows important progress messages, hides detailed technical logs.
        """
        message = record.getMessage()
        
        # Always show error and warning messages
        if record.levelno >= logging.WARNING:
            return True
        
        # Progress bars are now handled directly via stdout, not logging
        # So we don't need to filter them here anymore
        
        # Show main pipeline progress
        if any(phase in message for phase in ['[PIPELINE]', '[VALIDATION]', '[TRANSCRIPTION]', '[AI_PROCESSING]', '[FILE_GENERATION]', '[CLEANUP]']):
            # But hide the detailed sub-phases in non-verbose mode
            if any(detail in message.lower() for detail in [
                'model loaded', 'audio loaded', 'optimization', 'generating meeting title',
                'generating summary', 'extracting action items', 'detecting participants',
                'generating markdown', 'writing files', 'archiving', 'cleaning up'
            ]):
                return False
            return True
        
        # Hide all detailed technical logs from LOCAL_TRANSCRIPTION and WHISPER_CPP in non-verbose mode
        if any(prefix in message for prefix in ['[LOCAL_TRANSCRIPTION]', '[WHISPER_CPP]']):
            return False
        
        # Show important general messages
        if any(keyword in message.lower() for keyword in [
            'processing complete', 'successfully processed', 'created meeting files',
            'failed', 'error', 'starting', 'completed'
        ]):
            return True
        
        # Hide HTTP requests and other detailed logs
        if any(detail in message for detail in ['HTTP Request:', 'OpenAI', 'api.openai.com']):
            return False
        
        # Default: show other INFO messages
        return True
        
    def log_progress(self, phase: str, message: str, progress: int = None):
        """
        Centralized progress logging with consistent formatting.
        
        Args:
            phase: Processing phase (VALIDATION, TRANSCRIPTION, AI_PROCESSING, etc.)
            message: Progress message
            progress: Optional progress percentage (0-100)
        """
        if progress is not None:
            logging.info(f"[{phase}] {message} ({progress}%)")
        else:
            logging.info(f"[{phase}] {message}")
    
    def _validate_transcript(self, transcript_data: Dict) -> bool:
        """
        Validate transcript content to ensure it's not empty or corrupted
        
        Args:
            transcript_data: Dictionary containing transcript text and metadata
            
        Returns:
            bool: True if transcript is valid, False otherwise
        """
        try:
            # Check if transcript_data has required structure
            if not isinstance(transcript_data, dict):
                logging.error("Transcript data is not a dictionary")
                return False
            
            # Check for text content
            text = transcript_data.get('text', '')
            if not text or not isinstance(text, str):
                logging.error("Transcript contains no text content")
                return False
            
            # Check minimum length (at least 10 characters of actual content)
            cleaned_text = text.strip()
            if len(cleaned_text) < 10:
                logging.error(f"Transcript too short: {len(cleaned_text)} characters")
                return False
            
            # Check for segments if available
            segments = transcript_data.get('segments', [])
            if segments and len(segments) == 0:
                logging.warning("No segments found in transcript")
            
            # Check for obvious error indicators
            error_phrases = [
                "failed to transcribe",
                "no audio detected",
                "transcription error",
                "unable to process",
                "im sorry but i cant generate",
                "i need more information"
            ]
            
            text_lower = cleaned_text.lower()
            for error_phrase in error_phrases:
                if error_phrase in text_lower:
                    logging.error(f"Transcript contains error indicator: '{error_phrase}'")
                    return False
            
            # Additional validation for recovered transcripts
            optimizations = transcript_data.get('optimizations_applied', [])
            if 'emergency_recovery' in optimizations:
                logging.warning("Transcript was recovered using emergency methods - quality may be reduced")
                # Still allow it through but warn user
            
            logging.info(f"Transcript validation passed: {len(cleaned_text)} characters, {len(segments)} segments")
            return True
            
        except Exception as e:
            logging.error(f"Transcript validation failed with error: {e}")
            return False
        
    def process_audio_file(self, file_path: str):
        """Main processing pipeline for a single audio file."""
        try:
            self.log_progress("PIPELINE", f"Starting processing of {Path(file_path).name}")
            
            # Phase 1: Validation
            self.log_progress("VALIDATION", "Validating audio file...")
            if not self.audio_processor.validate_audio_file(file_path):
                logging.error(f"Invalid audio file: {file_path}")
                return False
            self.log_progress("VALIDATION", "Audio file validation complete")
                
            # Get duration for cost estimation
            duration = self.audio_processor.get_audio_duration(file_path)
            self.log_progress("VALIDATION", f"Audio duration: {duration:.1f} minutes")
            
            # Phase 2: Transcription Setup
            self.log_progress("TRANSCRIPTION", "Preparing transcription...")
            
            # Get language settings - CLI override takes precedence, then config
            transcription_language = self.transcription_language
            if not transcription_language:
                transcription_language = self.config.get('SETTINGS', 'transcription_language', fallback=None)
                if transcription_language and transcription_language.strip():
                    transcription_language = transcription_language.strip()
                else:
                    transcription_language = None
            
            if transcription_language:
                self.log_progress("TRANSCRIPTION", f"Using transcription language: {transcription_language}")
            
            # Phase 3: Transcription Processing
            self.log_progress("TRANSCRIPTION", "Starting audio transcription...")
            transcript_data = self.transcription_service.transcribe_audio(file_path, transcription_language)
            
            if 'error' in transcript_data:
                logging.error(f"Transcription failed: {transcript_data['error']}")
                return False
            
            # Validate transcript content before proceeding
            if not self._validate_transcript(transcript_data):
                logging.error("Transcript validation failed - content is empty or invalid")
                return False
            
            self.log_progress("TRANSCRIPTION", "Transcription completed successfully")
                
            # Phase 4: AI Processing
            self.log_progress("AI_PROCESSING", "Generating AI content...")
            prompts = self.config['PROMPTS']
            
            self.log_progress("AI_PROCESSING", "Generating meeting title...")
            title = self.ai_processor.generate_meeting_title(
                transcript_data['text'], 
                prompts['title_generation_prompt']
            )
            
            # Use custom prompt if provided, otherwise use config prompt
            summary_prompt = self.custom_summary_prompt if self.custom_summary_prompt else prompts['summary_prompt']
            
            # Determine target language for summary
            detected_language = transcript_data.get('language', 'unknown')
            
            # If detected language is unknown, use the transcription language that was actually used
            if detected_language == 'unknown':
                # First check if language was specified via command line
                if hasattr(self, 'transcription_language') and self.transcription_language:
                    detected_language = self.transcription_language
                    logging.info(f"Using command-line transcription language '{detected_language}' since detection failed")
                else:
                    # Fallback to configured transcription language
                    config_transcription_language = self.config.get('SETTINGS', 'transcription_language', fallback='unknown')
                    if config_transcription_language and config_transcription_language.strip():
                        detected_language = config_transcription_language.strip()
                        logging.info(f"Using configured transcription language '{detected_language}' since detection failed")
            
            # Check config for summary language setting
            config_summary_language = self.config.get('SETTINGS', 'summary_language', fallback=None)
            if config_summary_language and config_summary_language.strip():
                config_summary_language = config_summary_language.strip()
            else:
                config_summary_language = None
                
            # Use priority: command-line > config > detected language
            target_language = self.summary_language or config_summary_language or detected_language
            
            # Convert language codes to full names for clarity
            language_map = {
                'ru': 'Russian',
                'russian': 'Russian',
                'rus': 'Russian',
                'en': 'English',
                'english': 'English', 
                'eng': 'English',
                'es': 'Spanish',
                'spanish': 'Spanish',
                'fr': 'French',
                'french': 'French',
                'de': 'German',
                'german': 'German',
                'it': 'Italian',
                'italian': 'Italian',
                'pt': 'Portuguese',
                'portuguese': 'Portuguese'
            }
            
            # Normalize language code for mapping
            target_language_lower = target_language.lower() if target_language else 'unknown'
            
            if target_language_lower in language_map:
                target_language_name = language_map[target_language_lower]
            elif target_language != 'unknown':
                target_language_name = target_language.title()
            else:
                target_language_name = 'the same language as the transcript'
            
            # Add debug logging to see what language is being detected
            logging.info(f"Language detection: detected='{detected_language}', target='{target_language}', mapped='{target_language_name}'")
            
            self.log_progress("AI_PROCESSING", f"Generating summary in {target_language_name}...")
            summary = self.ai_processor.generate_summary(
                transcript_data['text'], 
                summary_prompt,
                self.context,
                target_language_name
            )
            
            # Only generate action items if requested
            if self.generate_action_items:
                self.log_progress("AI_PROCESSING", "Extracting action items...")
                action_items = self.ai_processor.extract_action_items(
                    transcript_data['text'], 
                    prompts['action_items_prompt'],
                    target_language_name
                )
            else:
                action_items = "Action items generation disabled for this meeting."
            
            self.log_progress("AI_PROCESSING", "Detecting participants...")
            participants = self.ai_processor.detect_participants(
                transcript_data['text'], 
                prompts['participant_detection_prompt']
            )
            
            # Phase 5: File Generation
            self.log_progress("FILE_GENERATION", "Creating output structure...")
            date_str = datetime.date.today().strftime('%Y-%m-%d')
            meeting_folder = self.obsidian_generator.create_meeting_folder(title, date_str)
            
            # Prepare data for file generation
            formatted_transcript = self.transcription_service.format_transcript_with_timestamps(
                transcript_data['segments']
            )
            
            meeting_data = {
                'title': title,
                'date': date_str,
                'duration': int(duration),
                'participants': participants,
                'language': transcript_data.get('language', 'unknown'),
                'overview': f"Meeting recording processed from {Path(file_path).name}",
                'processed_timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'original_filename': Path(file_path).name
            }
            
            transcript_file_data = {
                'title': title,
                'date': date_str,
                'duration_formatted': f"{int(duration//60):02d}:{int(duration%60):02d}",
                'formatted_transcript': formatted_transcript
            }
            
            summary_file_data = {
                'title': title,
                'date': date_str,
                'summary_content': summary
            }
            
            action_items_file_data = {
                'title': title,
                'date': date_str,
                'action_items_content': action_items
            }
            
            # Generate file contents
            self.log_progress("FILE_GENERATION", "Generating markdown files...")
            files_content = {
                'meta': self.obsidian_generator.generate_meta_file(meeting_data, self.generate_action_items),
                'transcript': self.obsidian_generator.generate_transcript_file(transcript_file_data),
                'summary': self.obsidian_generator.generate_summary_file(summary_file_data)
            }
            
            # Only include action items in files_content if they're being generated
            if self.generate_action_items:
                files_content['action_items'] = self.obsidian_generator.generate_action_items_file(action_items_file_data)
            
            # Write files
            self.log_progress("FILE_GENERATION", "Writing files to disk...")
            self.obsidian_generator.write_files(meeting_folder, files_content, self.generate_action_items)
            
            # Phase 6: Cleanup
            self.log_progress("CLEANUP", "Archiving processed file...")
            processed_path = self.audio_processor.move_to_processed(file_path)
            
            self.log_progress("PIPELINE", f"Processing complete! Output: {meeting_folder}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            return False
            
    def run_continuous_monitoring(self):
        """Monitor input folder and process new files automatically."""
        input_folder = self.audio_processor.input_folder
        logging.info(f"Starting continuous monitoring of folder: {input_folder}")
        
        try:
            while True:
                if input_folder.exists():
                    new_files = self.audio_processor.detect_new_files(str(input_folder))
                    
                    for file_path in new_files:
                        logging.info(f"Detected new file: {file_path}")
                        self.process_audio_file(file_path)
                
                time.sleep(10)  # Check every 10 seconds
                
        except KeyboardInterrupt:
            logging.info("Monitoring stopped by user")
        except Exception as e:
            logging.error(f"Error in continuous monitoring: {e}")
            
    def process_single_file(self, file_path: str):
        """Process a single file (for manual processing)."""
        return self.process_audio_file(file_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process Plaud Pin recordings for Obsidian')
    parser.add_argument('--file', help='Process single audio file')
    parser.add_argument('--monitor', action='store_true', help='Monitor folder for new files')
    parser.add_argument('--config', default='config.ini', help='Config file path')
    parser.add_argument('--custom-prompt', help='Custom prompt for meeting summarization')
    parser.add_argument('--context', help='Context about the meeting to guide summarization')
    parser.add_argument('--no-action-items', action='store_true', help='Disable action items generation')
    parser.add_argument('--language', help='Override transcription language (e.g., en, ru, es). Defaults to config setting')
    parser.add_argument('--model', help='Override Whisper model (tiny, base, small, medium, large, large-v2, large-v3). Defaults to config setting')
    parser.add_argument('--summary-language', help='Language for summary and action items (e.g., en, ru, es). Defaults to transcript language')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging for debugging (default: clean progress display)')
    
    args = parser.parse_args()
    
    print("Plaud Pin to Obsidian Processor")
    print("=" * 40)
    
    try:
        processor = PlaudProcessor(
            config_path=args.config,
            custom_summary_prompt=getattr(args, 'custom_prompt', None),
            context=getattr(args, 'context', None),
            generate_action_items=not args.no_action_items,
            summary_language=getattr(args, 'summary_language', None),
            transcription_language=getattr(args, 'language', None),
            whisper_model=getattr(args, 'model', None),
            verbose=args.verbose
        )
        
        if args.file:
            if Path(args.file).exists():
                print(f"Processing single file: {args.file}")
                success = processor.process_single_file(args.file)
                if success:
                    print("✅ File processed successfully!")
                else:
                    print("❌ File processing failed. Check logs for details.")
            else:
                print(f"❌ File not found: {args.file}")
                
        elif args.monitor:
            print("🔍 Starting continuous monitoring...")
            print("Press Ctrl+C to stop monitoring")
            processor.run_continuous_monitoring()
            
        else:
            print("Please specify --file or --monitor mode")
            print("\nExamples:")
            print("  python main.py --file path/to/audio.mp3")
            print("  python main.py --monitor")
            print("  python main.py --monitor --config custom_config.ini")
            print("  python main.py --file audio.mp3 --custom-prompt 'Create a technical summary focusing on decisions and blockers'")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have:")
        print("1. Created a .env file with OPENAI_API_KEY")
        print("2. Copied config.ini.example to config.ini")
        print("3. Installed dependencies: pip install -r requirements.txt")