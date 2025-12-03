"""AI processing for summaries, action items, and template-based analysis."""

import logging
from typing import Dict, List, Optional

from openai import OpenAI


class AIProcessor:
    """AI processing using OpenAI GPT-4o or YandexGPT."""

    def __init__(self, api_key: str, provider: str = "openai", yandex_folder_id: Optional[str] = None):
        self.provider = provider

        if provider == "yandex" and yandex_folder_id:
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://llm.api.cloud.yandex.net/v1"
            )
            self.model_name = f"gpt://{yandex_folder_id}/yandexgpt/latest"
        else:
            self.client = OpenAI(api_key=api_key)
            self.model_name = "gpt-4o"

    def process_with_template(self, transcript: str, file_config: Dict,
                              context: Optional[str] = None,
                              target_language: Optional[str] = None) -> str:
        """Process transcript using template configuration."""
        try:
            prompt = file_config.get('prompt', '')
            user_message = f"Please analyze this meeting transcript:\n\n{transcript}"

            if context:
                user_message = f"Context: {context}\n\n{prompt}\n\nTranscript:\n{transcript}"
            if target_language:
                user_message += f"\n\nIMPORTANT: Please respond in {target_language}."

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            return response.choices[0].message.content

        except Exception as e:
            logging.error(f"Template processing error: {e}")
            return f"Error: {e}"

    def generate_summary(self, transcript: str, prompt: str,
                         context: Optional[str] = None,
                         target_language: Optional[str] = None) -> str:
        """Generate meeting summary."""
        return self.process_with_template(
            transcript,
            {'prompt': prompt},
            context,
            target_language
        )

    def extract_action_items(self, transcript: str, prompt: str,
                             target_language: Optional[str] = None) -> str:
        """Extract action items from transcript."""
        try:
            user_message = f"Please extract action items from this transcript:\n\n{transcript}"
            if target_language:
                user_message += f"\n\nIMPORTANT: Write action items in {target_language}."

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=800,
                temperature=0.2
            )
            return response.choices[0].message.content

        except Exception as e:
            logging.error(f"Action items extraction error: {e}")
            return f"Error: {e}"

    def generate_title(self, transcript: str, prompt: str) -> str:
        """Generate meeting title from content."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Generate a title for:\n\n{transcript[:1000]}..."}
                ],
                max_tokens=50,
                temperature=0.3
            )
            return response.choices[0].message.content.strip().strip('"\'')

        except Exception as e:
            logging.error(f"Title generation error: {e}")
            return "Meeting Recording"

    def detect_participants(self, transcript: str, prompt: str) -> List[str]:
        """Identify meeting participants."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Identify participants:\n\n{transcript}"}
                ],
                max_tokens=200,
                temperature=0.2
            )
            text = response.choices[0].message.content.strip()
            return [p.strip() for p in text.split(',') if p.strip()]

        except Exception as e:
            logging.error(f"Participant detection error: {e}")
            return ["Unknown"]

    def translate(self, transcript: str, target_language: str = "Russian") -> str:
        """Translate transcript preserving timestamps and names."""
        try:
            prompt = f"""Translate this meeting transcript to {target_language}.

RULES:
1. Keep ALL timestamps exactly: [MM:SS]
2. Preserve proper names exactly as written
3. Translate only spoken content
4. Maintain format and structure"""

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Professional translator for meeting transcripts."},
                    {"role": "user", "content": f"{prompt}\n\n{transcript}"}
                ],
                max_tokens=4000,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logging.error(f"Translation error: {e}")
            return transcript


# Language code mapping
LANGUAGE_MAP = {
    'ru': 'Russian', 'russian': 'Russian', 'rus': 'Russian',
    'en': 'English', 'english': 'English', 'eng': 'English',
    'es': 'Spanish', 'spanish': 'Spanish',
    'fr': 'French', 'french': 'French',
    'de': 'German', 'german': 'German',
    'it': 'Italian', 'italian': 'Italian',
    'pt': 'Portuguese', 'portuguese': 'Portuguese'
}


def get_language_name(code: str) -> str:
    """Convert language code to full name."""
    if not code or code == 'unknown':
        return 'the same language as the transcript'
    return LANGUAGE_MAP.get(code.lower(), code.title())
