#!/usr/bin/env python3
"""
Plaud Processor - Audio to Obsidian Pipeline
Transform audio recordings into structured Obsidian notes with AI-powered
transcription, speaker diarization, and template-based summaries.
"""

import sys
import argparse
from pathlib import Path

from src.processor import PlaudProcessor
from src.obsidian import TemplateManager


def main():
    parser = argparse.ArgumentParser(
        description='Process audio recordings into Obsidian notes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --file meeting.mp3
  %(prog)s --file meeting.mp3 --speakers
  %(prog)s --file meeting.mp3 --template ai_committee --speakers
  %(prog)s --monitor --verbose
  %(prog)s --list-templates
        """
    )

    # Core options
    parser.add_argument('--file', help='Process single audio file')
    parser.add_argument('--monitor', action='store_true', help='Monitor folder for new files')
    parser.add_argument('--config', default='config.ini', help='Config file path')

    # Processing options
    parser.add_argument('--context', help='Context to guide AI summarization')
    parser.add_argument('--no-action-items', action='store_true', help='Skip action items generation')
    parser.add_argument('--template', help='Use a specific template (e.g., ai_committee, interview)')
    parser.add_argument('--list-templates', action='store_true', help='List available templates')
    parser.add_argument('--transcribe-only', action='store_true', help='Only transcribe, skip AI summarization')

    # Transcription options
    parser.add_argument('--language', help='Transcription language (e.g., en, ru, es)')
    parser.add_argument('--model', help='Whisper model (tiny, base, small, medium, large, large-v2, large-v3)')
    parser.add_argument('--summary-language', help='Language for AI output')

    # Speaker options
    parser.add_argument('--speakers', action='store_true', help='Enable speaker identification')
    parser.add_argument('--no-interactive', action='store_true', help='Skip interactive speaker naming')

    # AI provider
    parser.add_argument('--ai-provider', choices=['openai', 'yandex'], help='AI provider')

    # Output options
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    # List templates
    if args.list_templates:
        tm = TemplateManager()
        templates = tm.list_templates()
        print("Available Templates:")
        print("=" * 40)
        if not templates:
            print("No templates found in templates/ directory")
        else:
            for name in templates:
                try:
                    t = tm.load(name)
                    print(f"  {name:15} - {t.get('description', 'No description')}")
                except Exception as e:
                    print(f"  {name:15} - Error: {e}")
        print("\nUsage: uv run python main.py --template <name> --file <audio>")
        return 0

    # Validate arguments
    if not args.file and not args.monitor:
        parser.print_help()
        return 1

    print("Plaud Processor")
    print("=" * 40)

    try:
        processor = PlaudProcessor(
            config_path=args.config,
            context=args.context,
            generate_action_items=not args.no_action_items,
            summary_language=args.summary_language,
            transcription_language=args.language,
            whisper_model=args.model,
            verbose=args.verbose,
            enable_speakers=args.speakers,
            skip_interactive_naming=args.no_interactive,
            template_name=args.template,
            ai_provider=args.ai_provider,
            transcribe_only=args.transcribe_only
        )

        if args.file:
            if not Path(args.file).exists():
                print(f"File not found: {args.file}")
                return 1

            print(f"Processing: {args.file}")
            success = processor.process_single_file(args.file)
            if success:
                print("Done!")
                return 0
            else:
                print("Processing failed. Check logs.")
                return 1

        elif args.monitor:
            print("Starting continuous monitoring...")
            print("Press Ctrl+C to stop")
            processor.run_monitoring()
            return 0

    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nSetup instructions:")
        print("1. Create .env file with OPENAI_API_KEY")
        print("2. Copy config.ini.example to config.ini")
        print("3. For speakers: Set HUGGINGFACE_TOKEN and accept model license")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
