# Plaud Pin to Obsidian Processing Pipeline

## Project Overview
Create a Python script that processes audio recordings from a Plaud Pin device, transcribes them using OpenAI Whisper, generates summaries and action items using GPT-4o, and outputs structured Markdown files for Obsidian import.

## Requirements

### Core Functionality
1. **Audio Processing**: Monitor a designated folder for new audio files (MP3, WAV, M4A)
2. **Transcription**: Use OpenAI Whisper API to transcribe audio with timestamps
3. **AI Processing**: Use GPT-4o to generate:
   - Meeting summary
   - Action items with assignees and due dates
   - Key decisions and topics
4. **Obsidian Integration**: Create structured markdown files with proper linking

### Folder Structure Output
For each meeting, create a folder structure like:
```
meeting_data/
└── YYYY-MM-DD_Meeting-Name/
    ├── meeting_meta.md (main file with links to others)
    ├── meeting_transcript.md
    ├── meeting_summary.md
    └── meeting_action_items.md
```

### File Format Requirements

#### meeting_meta.md
```markdown
# Meeting: [Meeting Title]

**Date:** YYYY-MM-DD  
**Duration:** XX minutes  
**Participants:** [Auto-detected or manual input]  
**Status:** #meeting/processed

## Quick Navigation
- [[meeting_transcript]] - Full transcript with timestamps
- [[meeting_summary]] - Key points and decisions
- [[meeting_action_items]] - Tasks and follow-ups

## Meeting Overview
[Brief auto-generated description]

---
*Processed on: [timestamp]*
*Audio file: [original_filename]*
```

#### meeting_transcript.md
```markdown
# Meeting Transcript - [Title]

**Date:** YYYY-MM-DD  
**Duration:** XX:XX  
**Tags:** #transcript #meeting

## Transcript

[00:00] Speaker identification and timestamped content
[02:15] Continuing conversation...

---
**Related:** [[meeting_meta]] | [[meeting_summary]] | [[meeting_action_items]]
```

#### meeting_summary.md
```markdown
# Meeting Summary - [Title]

**Date:** YYYY-MM-DD  
**Tags:** #summary #meeting

## Key Points
- Main discussion topics
- Important decisions made
- Notable insights

## Decisions Made
1. Decision 1 with context
2. Decision 2 with reasoning

## Next Steps
- Overview of follow-up actions

---
**Related:** [[meeting_meta]] | [[meeting_transcript]] | [[meeting_action_items]]
```

#### meeting_action_items.md
```markdown
# Action Items - [Title]

**Date:** YYYY-MM-DD  
**Tags:** #actionitems #meeting #tasks

## Action Items

### High Priority
- [ ] **Action Item 1** - @assignee - Due: YYYY-MM-DD
  - Context and details
  
- [ ] **Action Item 2** - @assignee - Due: YYYY-MM-DD
  - Context and details

### Medium Priority
- [ ] **Action Item 3** - @assignee - Due: YYYY-MM-DD

### Follow-up Items
- [ ] **Follow-up task** - @assignee - Due: YYYY-MM-DD

---
**Related:** [[meeting_meta]] | [[meeting_transcript]] | [[meeting_summary]]
```

## Technical Specifications

### Dependencies
```python
# Core libraries
import os
import json
import datetime
from pathlib import Path
import configparser

# Audio processing
import librosa
from pydub import AudioSegment

# AI APIs
import openai
from openai import OpenAI

# File handling
import shutil
import hashlib

# Utilities
import re
import time
from typing import Dict, List, Optional
```

### Configuration
Create a `config.ini` file for settings:
```ini
[API]
openai_api_key = your_api_key_here

[PATHS]
input_folder = ./input_audio
output_folder = ./meeting_data
processed_folder = ./processed_audio

[SETTINGS]
auto_process = true
speaker_detection = true
max_file_size_mb = 500
supported_formats = mp3,wav,m4a,aac

[PROMPTS]
summary_prompt = You are a meeting assistant. Analyze this transcript and create a concise summary focusing on key decisions, main topics discussed, and important insights. Format as markdown.

action_items_prompt = Extract action items from this meeting transcript. For each action item, identify: 1) The specific task, 2) Who should do it (if mentioned), 3) Any deadlines or timeframes mentioned. Format as markdown with checkboxes.
```

### Core Functions Required

#### 1. Audio File Processing
```python
class AudioProcessor:
    def __init__(self, config):
        self.config = config
        
    def detect_new_files(self, folder_path: str) -> List[str]:
        """Monitor folder for new audio files"""
        pass
        
    def validate_audio_file(self, file_path: str) -> bool:
        """Check if file is valid audio and within size limits"""
        pass
        
    def get_audio_duration(self, file_path: str) -> float:
        """Get duration in minutes"""
        pass
```

#### 2. Transcription Service
```python
class TranscriptionService:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        
    def transcribe_audio(self, file_path: str) -> Dict:
        """
        Transcribe audio using Whisper API
        Returns: {
            'text': full_transcript,
            'segments': timestamped_segments,
            'duration': duration_seconds,
            'language': detected_language
        }
        """
        pass
        
    def format_transcript_with_timestamps(self, segments: List) -> str:
        """Format segments with timestamps for markdown"""
        pass
```

#### 3. AI Processing Service
```python
class AIProcessor:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        
    def generate_summary(self, transcript: str) -> str:
        """Generate meeting summary using GPT-4o"""
        pass
        
    def extract_action_items(self, transcript: str) -> str:
        """Extract and format action items using GPT-4o"""
        pass
        
    def generate_meeting_title(self, transcript: str) -> str:
        """Generate appropriate meeting title from content"""
        pass
        
    def detect_participants(self, transcript: str) -> List[str]:
        """Attempt to identify meeting participants"""
        pass
```

#### 4. Obsidian File Generator
```python
class ObsidianGenerator:
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        
    def create_meeting_folder(self, meeting_title: str, date: str) -> Path:
        """Create folder structure for meeting"""
        pass
        
    def generate_meta_file(self, meeting_data: Dict) -> str:
        """Generate meeting_meta.md content"""
        pass
        
    def generate_transcript_file(self, transcript_data: Dict) -> str:
        """Generate meeting_transcript.md content"""
        pass
        
    def generate_summary_file(self, summary_data: Dict) -> str:
        """Generate meeting_summary.md content"""
        pass
        
    def generate_action_items_file(self, action_items_data: Dict) -> str:
        """Generate meeting_action_items.md content"""
        pass
        
    def write_files(self, folder_path: Path, file_contents: Dict):
        """Write all markdown files to the meeting folder"""
        pass
```

#### 5. Main Processor Pipeline
```python
class PlaudProcessor:
    def __init__(self, config_path: str = "config.ini"):
        self.config = self.load_config(config_path)
        self.audio_processor = AudioProcessor(self.config)
        self.transcription_service = TranscriptionService(self.config['API']['openai_api_key'])
        self.ai_processor = AIProcessor(self.config['API']['openai_api_key'])
        self.obsidian_generator = ObsidianGenerator(self.config['PATHS']['output_folder'])
        
    def process_audio_file(self, file_path: str):
        """Main processing pipeline for a single audio file"""
        pass
        
    def run_continuous_monitoring(self):
        """Monitor input folder and process new files automatically"""
        pass
        
    def process_single_file(self, file_path: str):
        """Process a single file (for manual processing)"""
        pass
```

### Error Handling & Logging
- Comprehensive logging for debugging
- Graceful handling of API rate limits
- File corruption detection
- Resume capability for interrupted processing
- Cost tracking for API usage

### CLI Interface
```python
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process Plaud Pin recordings for Obsidian')
    parser.add_argument('--file', help='Process single audio file')
    parser.add_argument('--monitor', action='store_true', help='Monitor folder for new files')
    parser.add_argument('--config', default='config.ini', help='Config file path')
    
    args = parser.parse_args()
    
    processor = PlaudProcessor(args.config)
    
    if args.file:
        processor.process_single_file(args.file)
    elif args.monitor:
        processor.run_continuous_monitoring()
    else:
        print("Please specify --file or --monitor mode")
```

## Implementation Notes

### Cost Optimization
- Check file duration before processing to estimate costs
- Implement batching for multiple short recordings
- Add option to use different models based on file length
- Cache processing results to avoid reprocessing

### Obsidian Integration Features
- Use proper Wiki-link formatting for cross-references
- Add relevant tags for organization
- Include metadata for Obsidian's properties
- Support for custom templates

### Advanced Features (Future)
- Speaker diarization for multi-person meetings
- Integration with calendar apps for meeting context
- Auto-detection of meeting types (standup, review, planning)
- Email integration for sharing summaries
- Slack/Teams integration for action items

## File Structure
```
plaud_processor/
├── main.py (main script)
├── config.ini (configuration)
├── requirements.txt (dependencies)
├── README.md (documentation)
├── input_audio/ (watched folder)
├── meeting_data/ (output folder)
├── processed_audio/ (archive)
└── logs/ (processing logs)
```

## Testing Strategy
1. Test with various audio formats and qualities
2. Test with different meeting lengths
3. Validate markdown formatting in Obsidian
4. Test error handling with corrupted files
5. Performance testing with large files

## Security Considerations
- Secure API key storage
- Local audio file encryption option
- Cleanup of temporary files
- Audit trail for processed files

---

# Implementation Status - COMPLETED ✅

## What We've Built

### ✅ Complete Implementation
**Full Python Pipeline** - All core classes implemented and working:
- `AudioProcessor` - File validation, monitoring, duration calculation
- `TranscriptionService` - OpenAI Whisper API integration with timestamps  
- `AIProcessor` - GPT-4o for summaries, action items, titles, participants
- `ObsidianGenerator` - Structured markdown file creation with cross-links
- `PlaudProcessor` - Main pipeline orchestrating the entire process

### ✅ Project Setup & Git Ready
- **Git repository** initialized with proper `.gitignore`
- **Virtual environment** (.venv) configured 
- **Environment variables** (.env) for secure API key storage
- **Configuration system** (config.ini) for all settings
- **Dependencies** (requirements.txt) with all required packages
- **Documentation** (README.md) with setup and usage instructions

### ✅ Enhanced Features Added
- **Configurable input folder** via environment variables (`PLAUD_RECORDINGS_PATH`)
- **Direct Obsidian integration** via `OBSIDIAN_VAULT_PATH` environment variable
- **Numbered file organization** for proper Obsidian ordering (1_, 2_, 3_, 4_)
- **Language-aware AI processing** - summaries/action items in original transcript language
- **Cross-linked markdown files** with proper internal navigation
- **AirDrop workflow** optimized for Mac users
- **Comprehensive logging** with daily log files
- **Error handling** with graceful API failure recovery
- **CLI interface** with single file and continuous monitoring modes
- **Environment variable substitution** in configuration files
- **Task management system** with `pending_tasks.md`

### ✅ File Transfer Solution
**AirDrop → Custom Folder → Auto-processing → Direct Obsidian**
1. Set `PLAUD_RECORDINGS_PATH` and `OBSIDIAN_VAULT_PATH` in `.env` file
2. Record on Plaud Pin
3. AirDrop to Mac (save to your configured folder)  
4. Run `python main.py --monitor`
5. Automatic processing directly into your Obsidian vault

### ✅ Security & Best Practices
- **API keys** stored only in `.env` (never in config or code)
- **Sensitive files excluded** from git (.gitignore)
- **Modular architecture** with clear separation of concerns
- **Type hints** and comprehensive docstrings
- **Configuration flexibility** without hardcoded values

## Ready to Use
The processor is fully functional and ready for production use:

```bash
# Setup
cp .env.example .env          # Add OPENAI_API_KEY, PLAUD_RECORDINGS_PATH, OBSIDIAN_VAULT_PATH
cp config.ini.example config.ini
source .venv/bin/activate
pip install -r requirements.txt

# Usage  
python main.py --file audio.mp3    # Single file
python main.py --monitor           # Continuous monitoring
```

## Output Structure
Each processed meeting creates:
```
meeting_data/2024-06-21_Project-Planning/
├── meeting_meta.md         # Overview with navigation links
├── meeting_transcript.md   # Full transcript with timestamps  
├── meeting_summary.md      # AI-generated key points & decisions
└── meeting_action_items.md # Extracted action items by priority
```

**Status**: Implementation complete and production-ready! 🚀

---

## 🆕 LOCAL TRANSCRIPTION IMPLEMENTATION - June 2025

### Local Whisper Support
- **New Module**: `local_transcription.py` - Optimized local Whisper implementation
- **M2 Mac Acceleration**: Metal Performance Shaders (MPS) support for superior performance
- **Audio Optimizations**: Silence removal, normalization, and format conversion
- **Memory Management**: Efficient processing with automatic cleanup
- **Model Selection**: Support for all Whisper models (tiny to large-v3)

### Whisper.cpp Backend
- **New Module**: `whisper_cpp_backend.py` - Native whisper-cpp binary integration
- **Apple Silicon Optimization**: Optimized for M2 Mac performance
- **Model Management**: Automatic download and management of GGML models
- **Audio Preprocessing**: 16kHz mono conversion for optimal performance
- **Multi-threading**: 8-thread processing for speed

### Hybrid Transcription Service
- **Smart Fallback**: Local-first with API fallback
- **Cost Optimization**: $0 transcription costs when using local models
- **Configuration Options**: Choose between local and API transcription
- **Processing Info**: Detailed metrics on transcription method and optimizations

### Configuration Updates
```ini
[SETTINGS]
# Enable local transcription (default: true)
use_local_transcription = true

# Choose Whisper model (default: medium)
local_whisper_model = medium
# Options: tiny, base, small, medium, large, large-v2, large-v3
```

### Performance Benefits
- **Cost Savings**: $0.006/minute → $0 for transcription
- **Privacy**: Audio never leaves your Mac during transcription
- **Speed**: M2 acceleration + audio optimizations
- **Offline Capability**: Works without internet for transcription

### New Files Added
```
plaud_processor/
├── local_transcription.py          # Local Whisper implementation
├── whisper_cpp_backend.py          # whisper.cpp backend
├── LOCAL_TRANSCRIPTION_GUIDE.md    # Detailed setup and usage guide
└── whisper_models/                 # Directory for GGML models
    └── ggml-medium.bin             # Downloaded model files
```

---

## 🆕 NEW FEATURES ADDED - June 2025

### Context Parameter for Custom Summarization
- **Parameter**: `--context`
- **Description**: Provides context about the meeting to guide the AI summarization
- **Use Case**: Perfect for interviews, specific meeting types, or when you need tailored summaries
- **Example**: 
  ```bash
  python main.py --file interview.mp3 --context "this is an interview I was giving, summarize based on the topics covered with key bullet points for each. No need to shorten it too much, the interview was short enough"
  ```

### Action Items Control
- **Flag**: `--no-action-items`
- **Description**: Disables action items generation and file creation
- **Use Case**: Interviews, presentations, or meetings where action items aren't relevant
- **Default**: Action items are generated unless this flag is used
- **Example**:
  ```bash
  python main.py --file interview.mp3 --no-action-items
  ```

### Combined Usage Examples

#### Interview Processing (no action items needed)
```bash
python main.py --file interview.mp3 --context "this is an interview I was giving, summarize based on the topics covered with key bullet points for each. No need to shorten it too much, the interview was short enough" --no-action-items
```

#### Sprint Planning Meeting
```bash
python main.py --file sprint-planning.mp3 --context "This is a sprint planning meeting, focus on task assignments and timeline decisions"
```

#### Weekly Standup Monitoring
```bash
python main.py --monitor --context "Weekly team standup, highlight blockers and progress updates"
```

### Implementation Details
- **Context integration**: Context is passed to GPT-4o to customize the summarization prompt
- **Conditional file generation**: When `--no-action-items` is used:
  - No `4_meeting_action_items.md` file is created
  - Navigation links in `1_meeting_meta.md` exclude action items reference
  - Processing is faster with fewer API calls
- **Backward compatibility**: All existing functionality works unchanged

---

---

## 🆕 TRANSCRIPTION QUALITY IMPROVEMENTS - June 2025

### Russian Name Recognition Issue Solved
**Problem Identified**: Russian transcription was incorrectly transcribing proper names
- Russian models mapped unfamiliar names to common Russian alternatives
- Large-v3 model showed same issue as medium model
- English transcription preserved names more accurately

**Solution Implemented**: 
- **Medium + English model combination** provides best name accuracy for Russian speech
- Added `--model` and `--language` CLI parameters for flexible transcription
- Optimal workflow: `python main.py --file audio.mp3 --model medium --language en`

### Model Selection Enhancement
```bash
# New CLI parameters available:
python main.py --file audio.mp3 --model medium --language en
python main.py --monitor --model large-v3 --language ru
```

### Recommended Workflow for Russian Content
1. **For accuracy**: Use `--model medium --language en` 
2. **Manual translation**: Create Russian version preserving correct names
3. **Best of both**: Accurate English transcript + proper Russian translation

---

## 🆕 MAJOR STABILITY & UX IMPROVEMENTS - July 2025

### Critical Bug Fixes - Data Loss Prevention 🚨
**Problem Solved**: UTF-8 encoding errors causing complete loss of transcription after 18+ minutes of work

**Solutions Implemented**:
- **Robust Encoding Handling**: Multi-encoding fallback (utf-8 → cp1251 → latin-1 → utf-16)
- **Partial Transcription Recovery**: Real-time capture of transcript lines with automatic fallback
- **Transcript Validation**: Comprehensive content validation before AI processing
- **Graceful Degradation**: Recovery methods that parse timestamps and text from stdout

### Enhanced User Experience - Verbose Logging & Progress Bars
**New `--verbose` / `-v` Flag**:
```bash
# Clean, minimal output with progress bars (default)
python main.py --file audio.mp3

# Detailed debugging output
python main.py --file audio.mp3 --verbose
python main.py --file audio.mp3 -v
```

**Smart Progress Bars** (like tqdm):
- **In-place updating progress bar** for whisper.cpp transcription
- **Real-time time estimation** based on audio duration
- **Clean integration** with logging system
- **Automatic verbose mode detection** (progress bars in clean mode, detailed logs in verbose mode)

### Progress Bar Features
```
Default Mode:
[TRANSCRIPTION] Starting audio transcription...
Transcribing audio: ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 30% (15:20/53:17, 12:45 remaining)
[TRANSCRIPTION] Transcription completed successfully

Verbose Mode:
[WHISPER_CPP] [00:12:34.100 --> 00:12:34.100] Alright.
[WHISPER_CPP] [00:12:35.200 --> 00:12:35.200] Got you.
[WHISPER_CPP] Model loaded successfully in 2.34 seconds
```

### Advanced Recovery System
- **Never lose transcription work** due to encoding issues
- **Automatic partial result recovery** from stdout parsing
- **Multi-level fallback system** for maximum reliability
- **Comprehensive error handling** with graceful degradation

---

## 🆕 SPEAKER DIARIZATION IMPLEMENTATION - July 2025

### Advanced Speaker Identification System
**New Feature**: Complete speaker diarization with interactive naming workflow

**Implementation Details**:
- **New Module**: `speaker_diarization.py` - Full speaker identification pipeline
- **Interactive Workflow**: `InteractiveSpeakerNaming` class for user-friendly speaker labeling
- **M2 Mac Optimization**: Metal Performance Shaders (MPS) acceleration for pyannote-audio
- **Local Processing**: 100% on-device processing for privacy and security
- **Parallel Processing**: Runs alongside transcription for efficient workflow

### Key Features
- **Voice Activity Detection**: Identifies speech vs silence segments
- **Speaker Clustering**: Groups audio segments by speaker identity
- **Interactive Naming**: Sample quotes with timestamps for easy speaker identification
- **Speaker Statistics**: Speaking time, participation metrics, turn-taking analysis
- **Enhanced Transcripts**: Speaker-labeled transcripts with accurate attribution
- **AI Integration**: Speaker-aware summaries and action items generation

### CLI Parameters
```bash
# Enable speaker identification with interactive naming
python main.py --file meeting.mp3 --speakers

# Skip interactive naming (keep default Speaker A, B, C names)
python main.py --file meeting.mp3 --speakers --no-interactive

# Combined with other features
python main.py --file meeting.mp3 --speakers --context "team standup" --verbose
```

### Interactive Naming Workflow
```
Found 3 speakers in this meeting.
Please provide names for each speaker (press Enter to keep default name):

Speaker A said at [02:15]: "I think we should focus on the quarterly targets..."
Please enter name for Speaker A: John Smith

Speaker B said at [03:42]: "That's a good point, but we also need to consider..."
Please enter name for Speaker B: Sarah Johnson

Speaker C said at [05:18]: "I agree with both perspectives..."
Please enter name for Speaker C: Mike Chen
```

### Requirements & Setup
- **Dependencies**: `pyannote-audio>=3.1.0` (installed via requirements.txt)
- **HuggingFace Token**: Required for pyannote speaker-diarization model
- **Model License**: Must accept pyannote model user agreement
- **Environment Variable**: `HUGGINGFACE_TOKEN` in .env file

### Technical Implementation
- **Backend**: pyannote-audio 3.1 with speaker-diarization-3.1 model
- **Audio Optimization**: Mono conversion, 16kHz resampling, normalization
- **Confidence Filtering**: Configurable thresholds for segment quality
- **Memory Management**: Efficient processing with automatic cleanup
- **Error Handling**: Graceful fallback when diarization fails

### Performance Benefits
- **Enhanced Accuracy**: Speaker-aware AI processing for better summaries
- **Meeting Analytics**: Participation metrics and speaking time analysis
- **Improved Transcripts**: Clear speaker attribution for multi-person meetings
- **Privacy First**: All voice processing happens locally on your device

### New Files Added
```
plaud_processor/
├── speaker_diarization.py           # Core speaker identification service
├── test_speakers.py                 # Testing utility for speaker diarization
├── SPEAKER_DIARIZATION_GUIDE.md     # Comprehensive usage guide
└── requirements.txt                 # Updated with pyannote-audio
```

---

## 🚀 PROJECT STATUS: PRODUCTION READY

### Current State (July 2025)
The Plaud Processor is **fully functional and production-ready** with all major features implemented:

**✅ Core Features Complete:**
- **Audio Processing**: Automatic file detection and validation
- **Transcription**: Local Whisper + OpenAI API with fallback
- **AI Processing**: GPT-4o summaries and action items
- **Obsidian Integration**: Structured markdown with cross-links
- **Multi-language Support**: Russian/English optimization
- **Error Recovery**: Robust encoding handling and partial result preservation

**✅ Advanced Features:**
- **Context-aware Processing**: Custom summarization with `--context` parameter
- **Flexible Output**: Optional action items with `--no-action-items` flag
- **Model Selection**: `--model` and `--language` parameters for optimal transcription
- **Verbose Logging**: Clean progress bars vs detailed debugging with `--verbose`
- **Local Transcription**: Cost-free processing with M2 Mac acceleration
- **Speaker Diarization**: Interactive speaker identification with `--speakers` flag

**✅ Production Quality:**
- **Data Loss Prevention**: UTF-8 encoding fixes and recovery mechanisms
- **Progress Tracking**: Real-time progress bars and comprehensive logging
- **Configuration System**: Environment variables and INI file management
- **Documentation**: Complete setup guides and usage instructions

### Quick Start Commands
```bash
# Basic usage
python main.py --file audio.mp3

# With context and no action items (interviews)
python main.py --file interview.mp3 --context "interview summary" --no-action-items

# Continuous monitoring with verbose output
python main.py --monitor --verbose

# Russian transcription with English model for name accuracy
python main.py --file russian_meeting.mp3 --model medium --language en

# Speaker identification with interactive naming
python main.py --file meeting.mp3 --speakers

# Full featured meeting processing
python main.py --file meeting.mp3 --speakers --context "team standup" --verbose
```

---

## ⚠️ IMPORTANT: Task Management Protocol

**Before implementing any new features or changes:**

1. **Check `pending_tasks.md` first** for current issues and planned work
2. **Add new tasks to `pending_tasks.md`** before starting implementation  
3. **Remove tasks from `pending_tasks.md`** only when fully completed
4. **Update this CLAUDE.md** with significant changes

This ensures proper task tracking and prevents overlooking important issues.