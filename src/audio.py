"""Audio file processing, validation, and monitoring."""

import os
import re
import shutil
import datetime
import logging
from pathlib import Path
from typing import List

from pydub import AudioSegment


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
