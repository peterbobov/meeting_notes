"""Obsidian markdown file generation and template management."""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional


class TemplateManager:
    """Manages meeting processing templates."""

    def __init__(self, templates_dir: str = "templates"):
        self.templates_dir = Path(templates_dir)

    def load(self, template_name: str) -> Dict:
        """Load a template configuration."""
        template_path = self.templates_dir / f"{template_name}.json"

        if not template_path.exists():
            raise FileNotFoundError(f"Template '{template_name}' not found")

        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template = json.load(f)
            self._validate(template)
            return template
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in template '{template_name}': {e}")

    def list_templates(self) -> List[str]:
        """List available templates."""
        if not self.templates_dir.exists():
            return []
        return sorted([f.stem for f in self.templates_dir.glob("*.json")])

    def _validate(self, template: Dict) -> None:
        """Validate template structure."""
        for field in ['name', 'description', 'files']:
            if field not in template:
                raise ValueError(f"Template missing field: {field}")

        if not isinstance(template['files'], dict):
            raise ValueError("Template 'files' must be a dictionary")

        for file_key, config in template['files'].items():
            if not isinstance(config, dict) or 'enabled' not in config:
                raise ValueError(f"Invalid config for '{file_key}'")


class ObsidianGenerator:
    """Generates structured markdown files for Obsidian."""

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def create_folder(self, title: str, date: str) -> Path:
        """Create folder for meeting output."""
        clean_title = re.sub(r'[^\w\s-]', '', title).strip()
        clean_title = re.sub(r'[-\s]+', '-', clean_title)
        folder_path = self.output_path / f"{date}_{clean_title}"
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path

    def generate_meta(self, data: Dict, template_files: Optional[Dict] = None) -> str:
        """Generate meeting_meta.md content."""
        nav_links = ["- [[2_meeting_transcript]] - Full transcript with timestamps"]

        if template_files:
            for file_key, config in template_files.items():
                if config.get('enabled', False) and file_key != 'transcript':
                    filename = config.get('filename', f'{file_key}.md').replace('.md', '')
                    nav_links.append(f"- [[{filename}]] - {config.get('description', file_key)}")
        else:
            nav_links.extend([
                "- [[3_meeting_summary]] - Key points and decisions",
                "- [[4_meeting_action_items]] - Tasks and follow-ups"
            ])

        return f"""# Meeting: {data['title']}

**Date:** {data['date']}
**Duration:** {data['duration']} minutes
**Participants:** {', '.join(data['participants'])}
**Language:** {data.get('language', 'unknown').title()}
**Status:** #meeting/processed

## Quick Navigation
{chr(10).join(nav_links)}

## Meeting Overview
{data.get('overview', 'Auto-generated meeting recording.')}

---
*Processed on: {data['processed_timestamp']}*
*Audio file: {data['original_filename']}*"""

    def generate_transcript(self, data: Dict) -> str:
        """Generate meeting_transcript.md content."""
        return f"""# Meeting Transcript - {data['title']}

**Date:** {data['date']}
**Duration:** {data['duration_formatted']}
**Tags:** #transcript #meeting

## Transcript

{data['formatted_transcript']}

---
**Related:** [[1_meeting_meta]] | [[3_meeting_summary]] | [[4_meeting_action_items]]
"""

    def generate_summary(self, data: Dict) -> str:
        """Generate meeting_summary.md content."""
        speaker_section = ""
        if data.get('speaker_stats'):
            speaker_section = "\n## Speaker Statistics\n\n"
            for speaker, stats in data['speaker_stats'].items():
                speaker_section += f"**{speaker}:**\n"
                speaker_section += f"- Speaking time: {stats['total_time']:.1f}s ({stats['percentage']:.1f}%)\n"
                speaker_section += f"- Segments: {stats['segments']}\n"
                speaker_section += f"- Average: {stats['avg_segment_duration']:.1f}s\n\n"

        return f"""# Meeting Summary - {data['title']}

**Date:** {data['date']}
**Tags:** #summary #meeting

{data['summary_content']}
{speaker_section}
---
**Related:** [[1_meeting_meta]] | [[2_meeting_transcript]] | [[4_meeting_action_items]]
"""

    def generate_action_items(self, data: Dict) -> str:
        """Generate meeting_action_items.md content."""
        return f"""# Action Items - {data['title']}

**Date:** {data['date']}
**Tags:** #actionitems #meeting #tasks

{data['action_items_content']}

---
**Related:** [[1_meeting_meta]] | [[2_meeting_transcript]] | [[3_meeting_summary]]
"""

    def generate_template_file(self, config: Dict, content: str, data: Dict) -> str:
        """Generate file from template configuration."""
        template = config.get('template', '{content}')
        return template.format(
            title=data.get('title', ''),
            date=data.get('date', ''),
            analysis_content=content,
            content=content
        )

    def write_files(self, folder: Path, contents: Dict, template_files: Optional[Dict] = None):
        """Write all markdown files to folder."""
        files = {
            '1_meeting_meta.md': contents.get('meta', ''),
            '2_meeting_transcript.md': contents.get('transcript', '')
        }

        if template_files:
            for file_key, config in template_files.items():
                if config.get('enabled', False) and file_key in contents:
                    filename = config.get('filename', f'{file_key}.md')
                    files[filename] = contents[file_key]
        else:
            if contents.get('summary'):
                files['3_meeting_summary.md'] = contents['summary']
            if contents.get('action_items'):
                files['4_meeting_action_items.md'] = contents['action_items']

        for filename, content in files.items():
            if content:
                file_path = folder / filename
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logging.info(f"Created: {file_path}")

        logging.info(f"Created meeting files in: {folder}")
