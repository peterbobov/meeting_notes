"""Main processor pipeline for audio to Obsidian conversion."""

import os
import re
import datetime
import logging
import time
import configparser
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

from .audio import AudioProcessor
from .transcribe import TranscriptionService
from .ai import AIProcessor, get_language_name
from .obsidian import ObsidianGenerator, TemplateManager
from .speakers import SpeakerDiarizationService, SpeakerNaming

load_dotenv()


class PlaudProcessor:
    """Main pipeline for processing audio recordings into Obsidian notes."""

    def __init__(
        self,
        config_path: str = "config.ini",
        context: Optional[str] = None,
        generate_action_items: bool = True,
        summary_language: Optional[str] = None,
        transcription_language: Optional[str] = None,
        whisper_model: Optional[str] = None,
        verbose: bool = False,
        enable_speakers: bool = False,
        skip_interactive_naming: bool = False,
        template_name: Optional[str] = None,
        ai_provider: Optional[str] = None,
        transcribe_only: bool = False
    ):
        self.config = self._load_config(config_path)
        self.context = context
        self.generate_action_items = generate_action_items
        self.summary_language = summary_language
        self.transcription_language = transcription_language
        self.verbose = verbose
        self.enable_speakers = enable_speakers
        self.skip_interactive_naming = skip_interactive_naming
        self.template_name = template_name
        self.transcribe_only = transcribe_only

        self._setup_logging()

        # Initialize components
        self.audio_processor = AudioProcessor(self.config)

        # Transcription setup
        use_local = self.config.getboolean('SETTINGS', 'use_local_transcription', fallback=True)
        model = whisper_model or self.config.get('SETTINGS', 'local_whisper_model', fallback='medium')
        logging.info(f"Using Whisper model: {model}")

        # AI provider setup (skip if transcribe_only)
        self.ai = None
        if transcribe_only:
            # For transcribe-only mode, we just need transcription service
            openai_key = os.getenv('OPENAI_API_KEY', 'dummy')
            self.transcription = TranscriptionService(openai_key, use_local, model)
            logging.info("Transcribe-only mode: AI processing disabled")
        else:
            provider = (ai_provider or os.getenv('AI_PROVIDER', 'openai')).lower()

            if provider == 'yandex':
                yandex_key = os.getenv('YANDEX_API_KEY')
                yandex_folder = os.getenv('YANDEX_FOLDER_ID')
                if not yandex_key or not yandex_folder:
                    raise ValueError("YandexGPT requires YANDEX_API_KEY and YANDEX_FOLDER_ID")
                openai_key = os.getenv('OPENAI_API_KEY', 'dummy')
                self.transcription = TranscriptionService(openai_key, use_local, model)
                self.ai = AIProcessor(yandex_key, 'yandex', yandex_folder)
                logging.info(f"Using YandexGPT for AI processing")
            else:
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OPENAI_API_KEY required")
                self.transcription = TranscriptionService(api_key, use_local, model)
                self.ai = AIProcessor(api_key)
                logging.info("Using OpenAI GPT-4o for AI processing")

        # Output folder with env var expansion
        output_folder = self._expand_env_vars(
            self.config['PATHS'].get('output_folder', './meeting_data')
        )
        self.obsidian = ObsidianGenerator(output_folder)
        self.template_manager = TemplateManager()

        # Load template
        self.template = None
        if template_name:
            try:
                self.template = self.template_manager.load(template_name)
                logging.info(f"Using template: {self.template['name']}")
            except Exception as e:
                logging.error(f"Failed to load template '{template_name}': {e}")
                raise

        # Speaker diarization
        self.speaker_service = None
        if enable_speakers:
            try:
                self.speaker_service = SpeakerDiarizationService()
                logging.info("Speaker diarization enabled")
            except ImportError as e:
                logging.warning(f"Speaker diarization not available: {e}")
                self.enable_speakers = False

        self.audio_processor.input_folder.mkdir(parents=True, exist_ok=True)
        logging.info(f"Input folder: {self.audio_processor.input_folder}")

    def _load_config(self, path: str) -> configparser.ConfigParser:
        """Load configuration file."""
        config = configparser.ConfigParser()
        if Path(path).exists():
            config.read(path)
        elif Path(f"{path}.example").exists():
            config.read(f"{path}.example")
            logging.warning(f"Using example config. Copy {path}.example to {path}")
        else:
            raise FileNotFoundError(f"Config not found: {path}")
        return config

    def _expand_env_vars(self, path: str) -> str:
        """Expand ${VAR} in path strings."""
        if '${' not in path:
            return path
        for var in re.findall(r'\$\{([^}]+)\}', path):
            value = os.getenv(var, '')
            if value:
                path = path.replace(f'${{{var}}}', value)
            else:
                return './meeting_data'
        return path

    def _setup_logging(self):
        """Configure logging based on verbose mode."""
        logs_dir = Path(self.config['PATHS']['logs_folder'])
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / f"plaud_{datetime.date.today().strftime('%Y%m%d')}.log"

        for h in logging.root.handlers[:]:
            logging.root.removeHandler(h)

        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        console = logging.getLogger().handlers[-1]
        console.setLevel(logging.DEBUG if self.verbose else logging.INFO)

    def _log(self, phase: str, message: str):
        """Log progress message."""
        logging.info(f"[{phase}] {message}")

    def _format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS."""
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

    def process_file(self, file_path: str) -> bool:
        """Process a single audio file through the pipeline."""
        try:
            self._log("PIPELINE", f"Processing {Path(file_path).name}")

            # Validation
            self._log("VALIDATION", "Validating audio file...")
            if not self.audio_processor.validate_audio_file(file_path):
                logging.error(f"Invalid audio: {file_path}")
                return False

            duration = self.audio_processor.get_audio_duration(file_path)
            self._log("VALIDATION", f"Duration: {duration:.1f} minutes")

            # Transcription
            self._log("TRANSCRIPTION", "Starting transcription...")
            lang = self.transcription_language or self.config.get('SETTINGS', 'transcription_language', fallback=None)
            transcript = self.transcription.transcribe(file_path, lang)

            if transcript.get('error') or not transcript.get('text', '').strip():
                logging.error("Transcription failed or empty")
                return False

            self._log("TRANSCRIPTION", "Complete")

            # Speaker diarization
            if self.enable_speakers and self.speaker_service:
                self._log("SPEAKER_DIARIZATION", "Identifying speakers...")
                result = self.speaker_service.perform_diarization(file_path)

                if result.speakers:
                    self._log("SPEAKER_DIARIZATION", f"Found {len(result.speakers)} speakers")
                    naming = SpeakerNaming(transcript['segments'], result.segments)
                    merged = naming.merge_transcript_with_speakers()
                    quotes = naming.get_sample_quotes(merged)
                    names = naming.prompt_for_names(quotes, self.skip_interactive_naming)
                    transcript['segments'] = naming.apply_names(merged, names)
                    transcript['speaker_names'] = names

                    # Update speaker stats with names
                    named_stats = {}
                    for sid, stats in result.speaker_stats.items():
                        name = names.get(sid, sid)
                        if name in named_stats:
                            named_stats[name]['total_time'] += stats['total_time']
                            named_stats[name]['segments'] += stats['segments']
                        else:
                            named_stats[name] = stats.copy()
                    transcript['speaker_stats'] = named_stats

                    # Rebuild text with speakers
                    lines = []
                    for seg in transcript['segments']:
                        speaker = seg.get("speaker", "Unknown")
                        text = seg.get("text", "").strip()
                        if text:
                            lines.append(f"{speaker}: {text}")
                    transcript['text'] = '\n'.join(lines)

            # Format transcript
            if self.enable_speakers and transcript.get('speaker_names'):
                lines = []
                for seg in transcript['segments']:
                    t = self._format_time(seg.get("start", 0))
                    speaker = seg.get("speaker", "Unknown")
                    text = seg.get("text", "").strip()
                    if text:
                        lines.append(f"[{t}] {speaker}: {text}")
                formatted = '\n'.join(lines)
            else:
                formatted = self.transcription.format_transcript(transcript['segments'])

            date_str = datetime.date.today().strftime('%Y-%m-%d')

            # Transcribe-only mode: skip AI processing, output only transcript
            if self.transcribe_only:
                self._log("FILE_GENERATION", "Creating transcript output...")
                title = Path(file_path).stem.replace('_', ' ').replace('-', ' ').title()
                folder = self.obsidian.create_folder(title, date_str)

                transcript_data = {
                    'title': title,
                    'date': date_str,
                    'duration_formatted': f"{int(duration//60):02d}:{int(duration%60):02d}",
                    'formatted_transcript': formatted
                }
                files = {'transcript': self.obsidian.generate_transcript(transcript_data)}
                self.obsidian.write_files(folder, files)

                self._log("CLEANUP", "Archiving processed file...")
                self.audio_processor.move_to_processed(file_path)
                self._log("PIPELINE", f"Complete! Transcript: {folder}")
                return True

            # Full AI Processing
            self._log("AI_PROCESSING", "Generating content...")
            prompts = self.config['PROMPTS']

            self._log("AI_PROCESSING", "Generating title...")
            title = self.ai.generate_title(transcript['text'], prompts['title_generation_prompt'])

            detected_lang = transcript.get('language', 'unknown')
            if detected_lang == 'unknown' and lang:
                detected_lang = lang
            target_lang = get_language_name(self.summary_language or detected_lang)

            processed = {}
            if self.template:
                self._log("AI_PROCESSING", f"Using template: {self.template['name']}")
                for key, config in self.template['files'].items():
                    if config.get('enabled'):
                        self._log("AI_PROCESSING", f"Generating {key}...")
                        processed[key] = self.ai.process_with_template(
                            transcript['text'], config, self.context, target_lang
                        )
            else:
                self._log("AI_PROCESSING", f"Generating summary in {target_lang}...")
                processed['summary'] = self.ai.generate_summary(
                    transcript['text'], prompts['summary_prompt'], self.context, target_lang
                )
                if self.generate_action_items:
                    self._log("AI_PROCESSING", "Extracting action items...")
                    processed['action_items'] = self.ai.extract_action_items(
                        transcript['text'], prompts['action_items_prompt'], target_lang
                    )

            self._log("AI_PROCESSING", "Detecting participants...")
            participants = self.ai.detect_participants(
                transcript['text'], prompts['participant_detection_prompt']
            )

            # File Generation
            self._log("FILE_GENERATION", "Creating output...")
            folder = self.obsidian.create_folder(title, date_str)

            meeting_data = {
                'title': title,
                'date': date_str,
                'duration': int(duration),
                'participants': participants,
                'language': transcript.get('language', 'unknown'),
                'overview': f"Processed from {Path(file_path).name}",
                'processed_timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'original_filename': Path(file_path).name
            }

            transcript_data = {
                'title': title,
                'date': date_str,
                'duration_formatted': f"{int(duration//60):02d}:{int(duration%60):02d}",
                'formatted_transcript': formatted
            }

            # Build file contents
            files = {'transcript': self.obsidian.generate_transcript(transcript_data)}

            if self.template:
                files['meta'] = self.obsidian.generate_meta(meeting_data, self.template['files'])
                for key, config in self.template['files'].items():
                    if config.get('enabled') and key in processed:
                        content = processed[key]
                        if 'template' in config:
                            content = self.obsidian.generate_template_file(config, content, meeting_data)
                        files[key] = content
                self.obsidian.write_files(folder, files, self.template['files'])
            else:
                files['meta'] = self.obsidian.generate_meta(meeting_data)
                files['summary'] = self.obsidian.generate_summary({
                    'title': title,
                    'date': date_str,
                    'summary_content': processed.get('summary', ''),
                    'speaker_stats': transcript.get('speaker_stats', {})
                })
                if processed.get('action_items'):
                    files['action_items'] = self.obsidian.generate_action_items({
                        'title': title,
                        'date': date_str,
                        'action_items_content': processed['action_items']
                    })
                self.obsidian.write_files(folder, files)

            # Cleanup
            self._log("CLEANUP", "Archiving processed file...")
            self.audio_processor.move_to_processed(file_path)

            self._log("PIPELINE", f"Complete! Output: {folder}")
            return True

        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            return False

    def run_monitoring(self):
        """Monitor input folder for new files."""
        folder = self.audio_processor.input_folder
        logging.info(f"Monitoring: {folder}")

        try:
            while True:
                if folder.exists():
                    for f in self.audio_processor.detect_new_files(str(folder)):
                        logging.info(f"New file: {f}")
                        self.process_file(f)
                time.sleep(10)
        except KeyboardInterrupt:
            logging.info("Monitoring stopped")

    def process_single_file(self, file_path: str) -> bool:
        """Process a single file."""
        return self.process_file(file_path)
