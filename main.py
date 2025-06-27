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

# Audio processing
import librosa
from pydub import AudioSegment

# AI APIs
from openai import OpenAI

# Local transcription
from local_transcription import LocalTranscriptionService, TranscriptionResult

# Environment
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


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
            self.local_service = LocalTranscriptionService(
                model_size=local_model,
                enable_optimizations=True
            )
            logging.info(f"Initialized local transcription with model '{local_model}'")
        else:
            self.client = OpenAI(api_key=api_key)
            logging.info("Initialized OpenAI API transcription service")
        
    def transcribe_audio(self, file_path: str) -> Dict:
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
            return self._transcribe_local(file_path)
        else:
            return self._transcribe_api(file_path)
    
    def _transcribe_local(self, file_path: str) -> Dict:
        """Transcribe using local Whisper model."""
        try:
            result = self.local_service.transcribe_audio(file_path)
            
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
                return self._transcribe_api(file_path)
            else:
                return {
                    'text': '',
                    'segments': [],
                    'duration': 0,
                    'language': 'unknown',
                    'error': str(e),
                    'processing_info': {'method': 'local_whisper_failed'}
                }
    
    def _transcribe_api(self, file_path: str) -> Dict:
        """Transcribe using OpenAI Whisper API."""
        try:
            if not hasattr(self, 'client'):
                self.client = OpenAI(api_key=self.api_key)
                
            with open(file_path, "rb") as audio_file:
                # Transcribe with timestamps
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )
                
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
        
    def generate_summary(self, transcript: str, prompt: str) -> str:
        """Generate meeting summary using GPT-4o."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Please analyze this meeting transcript and create a summary:\n\n{transcript}"}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logging.error(f"Error generating summary: {e}")
            return f"Error generating summary: {e}"
            
    def extract_action_items(self, transcript: str, prompt: str) -> str:
        """Extract and format action items using GPT-4o."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Please extract action items from this meeting transcript:\n\n{transcript}"}
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
        
    def generate_meta_file(self, meeting_data: Dict) -> str:
        """Generate meeting_meta.md content."""
        template = f"""# Meeting: {meeting_data['title']}

**Date:** {meeting_data['date']}
**Duration:** {meeting_data['duration']} minutes
**Participants:** {', '.join(meeting_data['participants'])}
**Language:** {meeting_data.get('language', 'unknown').title()}
**Status:** #meeting/processed

## Quick Navigation
- [[2_meeting_transcript]] - Full transcript with timestamps
- [[3_meeting_summary]] - Key points and decisions
- [[4_meeting_action_items]] - Tasks and follow-ups

## Meeting Overview
{meeting_data.get('overview', 'Auto-generated meeting recording processed for analysis and action item tracking.')}

---
*Processed on: {meeting_data['processed_timestamp']}*
*Audio file: {meeting_data['original_filename']}*
"""
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
        
    def write_files(self, folder_path: Path, file_contents: Dict):
        """Write all markdown files to the meeting folder."""
        files_to_write = {
            '1_meeting_meta.md': file_contents['meta'],
            '2_meeting_transcript.md': file_contents['transcript'],
            '3_meeting_summary.md': file_contents['summary'],
            '4_meeting_action_items.md': file_contents['action_items']
        }
        
        for filename, content in files_to_write.items():
            file_path = folder_path / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
        logging.info(f"Created meeting files in: {folder_path}")


class PlaudProcessor:
    """Main processor pipeline for Plaud Pin recordings."""
    
    def __init__(self, config_path: str = "config.ini", custom_summary_prompt: Optional[str] = None):
        self.config = self.load_config(config_path)
        self.custom_summary_prompt = custom_summary_prompt
        self.setup_logging()
        
        # Initialize components
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")
            
        self.audio_processor = AudioProcessor(self.config)
        
        # Use local transcription by default, with API fallback
        use_local = self.config.getboolean('SETTINGS', 'use_local_transcription', fallback=True)
        local_model = self.config.get('SETTINGS', 'local_whisper_model', fallback='medium')
        
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
        """Setup logging configuration."""
        logs_folder = Path(self.config['PATHS']['logs_folder'])
        logs_folder.mkdir(parents=True, exist_ok=True)
        
        log_file = logs_folder / f"plaud_processor_{datetime.date.today().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def process_audio_file(self, file_path: str):
        """Main processing pipeline for a single audio file."""
        try:
            logging.info(f"Processing audio file: {file_path}")
            
            # Validate audio file
            if not self.audio_processor.validate_audio_file(file_path):
                logging.error(f"Invalid audio file: {file_path}")
                return False
                
            # Get duration for cost estimation
            duration = self.audio_processor.get_audio_duration(file_path)
            logging.info(f"Audio duration: {duration:.1f} minutes")
            
            # Transcribe audio
            logging.info("Starting transcription...")
            transcript_data = self.transcription_service.transcribe_audio(file_path)
            
            if 'error' in transcript_data:
                logging.error(f"Transcription failed: {transcript_data['error']}")
                return False
                
            # Generate AI content
            logging.info("Generating AI content...")
            prompts = self.config['PROMPTS']
            
            title = self.ai_processor.generate_meeting_title(
                transcript_data['text'], 
                prompts['title_generation_prompt']
            )
            
            # Use custom prompt if provided, otherwise use config prompt
            summary_prompt = self.custom_summary_prompt if self.custom_summary_prompt else prompts['summary_prompt']
            summary = self.ai_processor.generate_summary(
                transcript_data['text'], 
                summary_prompt
            )
            
            action_items = self.ai_processor.extract_action_items(
                transcript_data['text'], 
                prompts['action_items_prompt']
            )
            
            participants = self.ai_processor.detect_participants(
                transcript_data['text'], 
                prompts['participant_detection_prompt']
            )
            
            # Create output folder and files
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
            files_content = {
                'meta': self.obsidian_generator.generate_meta_file(meeting_data),
                'transcript': self.obsidian_generator.generate_transcript_file(transcript_file_data),
                'summary': self.obsidian_generator.generate_summary_file(summary_file_data),
                'action_items': self.obsidian_generator.generate_action_items_file(action_items_file_data)
            }
            
            # Write files
            self.obsidian_generator.write_files(meeting_folder, files_content)
            
            # Move processed file
            processed_path = self.audio_processor.move_to_processed(file_path)
            logging.info(f"Moved processed file to: {processed_path}")
            
            logging.info(f"Successfully processed: {file_path}")
            logging.info(f"Output folder: {meeting_folder}")
            
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
    
    args = parser.parse_args()
    
    print("Plaud Pin to Obsidian Processor")
    print("=" * 40)
    
    try:
        processor = PlaudProcessor(args.config, getattr(args, 'custom_prompt', None))
        
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