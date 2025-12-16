# Audio-to-Obsidian: AI-Powered Meeting Processor

Transform any audio recording into structured, searchable Obsidian notes with AI-powered transcription, intelligent summaries, and actionable insights.

## 🎯 Why This Project?

**The Problem**: Plaud Pin charges premium prices for basic transcription services that can be achieved much cheaper and with better quality using OpenAI's APIs directly.

**The Solution**: This tool processes any audio recording (Plaud Pin, iPhone recordings, dedicated recorders, etc.) at a fraction of the cost while providing superior features:

- **Cost Comparison**: ~$0.85 for a 60-minute meeting vs. Plaud's subscription fees
- **Better Quality**: OpenAI Whisper provides more accurate transcription
- **Enhanced Features**: Automatic summaries, action items, and Obsidian integration
- **Universal Compatibility**: Works with any audio source, not just Plaud devices

## ✨ Features

- 🎙️ **Universal Audio Support**: MP3, WAV, M4A, AAC from any recording device
- 🤖 **AI-Powered Processing**: OpenAI Whisper + GPT-4o for superior results
- 📝 **Intelligent Summaries**: Auto-generated meeting summaries with key decisions
- ✅ **Smart Action Items**: Extracted tasks with priorities and assignees
- 👥 **Speaker Diarization**: Interactive speaker identification and naming
- 🌍 **Multi-language**: Maintains original language (Russian, English, etc.)
- 📁 **Obsidian Integration**: Direct vault integration with cross-linked files
- 📊 **Organized Output**: Numbered files for perfect Obsidian organization
- 🔄 **Automation**: Continuous monitoring and processing
- 💰 **Cost-Effective**: ~$0.006/minute vs. expensive subscription services

## 🚀 Complete Workflow Guide

### Step 1: Installation & Setup

```bash
# Clone the repository
git clone https://github.com/peterbobov/meeting_notes.git
cd meeting_notes

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies with uv (automatically creates virtual environment)
uv sync

# Install FFmpeg (required for audio processing)
# macOS:
brew install ffmpeg
# Ubuntu/Debian:
sudo apt install ffmpeg
# Windows: Download from https://ffmpeg.org/
```

### Step 2: Configuration

```bash
# Copy example files
cp .env.example .env
cp config.ini.example config.ini

# Edit .env with your settings (ALL PATHS CONFIGURED HERE)
OPENAI_API_KEY=your_api_key_here                           # Get from OpenAI
PLAUD_RECORDINGS_PATH=/your_path_to_recordings_here        # Your audio files folder
OBSIDIAN_VAULT_PATH=/your_path_to_obsidian_vault/Meetings  # Direct Obsidian integration
HUGGINGFACE_TOKEN=your_huggingface_token_here             # Optional: For speaker diarization
```

### Step 3: Recording & Processing Workflow

#### Option A: Automatic Processing (Recommended)
```bash
# Start continuous monitoring
uv run python main.py --monitor

# In another terminal/process:
# 1. Record audio (Plaud Pin, iPhone, any recorder)
# 2. Save/AirDrop to your PLAUD_RECORDINGS_PATH folder
# 3. Files are automatically processed and appear in Obsidian
```

#### Option B: Manual Processing
```bash
# Process a specific file (speaker diarization enabled by default)
uv run python main.py --file path/to/your/recording.mp3

# Process with custom config
uv run python main.py --file recording.mp3 --config custom_config.ini

# Skip interactive speaker naming (use SPEAKER_00, SPEAKER_01, etc.)
uv run python main.py --file meeting.mp3 --no-interactive

# Disable speaker identification (for self-recordings)
uv run python main.py --file meeting.mp3 --no-speakers
```

### Step 4: Obsidian Integration

Your processed meetings automatically appear in Obsidian with:
- `1_meeting_meta.md` - Overview with navigation links
- `2_meeting_transcript.md` - Full transcript with timestamps  
- `3_meeting_summary.md` - AI-generated summary
- `4_meeting_action_items.md` - Extracted action items

Click any `[[link]]` to navigate between files!

## 📁 Output Structure

Each processed meeting creates a perfectly organized folder:

```
your-obsidian-vault/Meetings/
└── 2024-06-21_Meeting-Title/
    ├── 1_meeting_meta.md         # 📋 Overview with navigation & metadata
    ├── 2_meeting_transcript.md   # 📝 Full transcript with timestamps
    ├── 3_meeting_summary.md      # 🎯 AI-generated summary & decisions
    └── 4_meeting_action_items.md # ✅ Extracted tasks by priority
```

**File Contents:**
- **Meta**: Date, duration, participants, language, cross-links
- **Transcript**: Timestamped transcript in original language (with speaker labels if --speakers used)
- **Summary**: Key points, decisions, next steps (same language as transcript)
- **Action Items**: Categorized by priority with assignees and deadlines

## Obsidian Integration

### Direct Vault Integration (Recommended)

Set `OBSIDIAN_VAULT_PATH` in your `.env` file to automatically save processed meetings directly to your Obsidian vault:

```bash
OBSIDIAN_VAULT_PATH=/path/to/your/obsidian/vault/Meetings
```

### Manual Import

If not using direct integration, copy meeting folders from `meeting_data/` to your Obsidian vault manually.

### Features in Obsidian

- **Cross-linked files**: Click `[[meeting_transcript]]` to navigate between files
- **Searchable content**: Full-text search across all transcripts  
- **Taggable**: Files include `#meeting`, `#transcript`, `#summary`, `#actionitems` tags
- **Graph view**: Visual connections between related meetings

## Configuration

### Environment Variables (`.env`)
Configure all paths in your `.env` file:
- **PLAUD_RECORDINGS_PATH**: Where your audio files are stored
- **OBSIDIAN_VAULT_PATH**: Direct integration with your Obsidian vault
- **OPENAI_API_KEY**: Your OpenAI API key

### Settings (`config.ini`)
Customize processing behavior:
- **Settings**: File size limits, supported formats, timeouts
- **Prompts**: AI processing instructions for summaries and action items

## Folder Structure

```
plaud_processor/
├── main.py                 # Main application
├── config.ini             # Configuration
├── .env                   # API keys
├── requirements.txt       # Dependencies
├── input_audio/          # Drop audio files here
├── meeting_data/         # Processed outputs
├── processed_audio/      # Archived files
└── logs/                # Processing logs
```

## 💰 Cost Comparison

### This Tool (OpenAI APIs)
- **Whisper Transcription**: $0.006 per minute
- **GPT-4o Processing**: ~$0.01-0.02 per meeting
- **60-minute meeting**: ~$0.38 total

### Plaud Pin Subscription
- **Monthly**: $79/month ($948/year)
- **Annual**: $699/year  
- **Per meeting cost**: $2-5+ depending on usage

### 📊 Savings Example
- **100 hours of meetings/year**:
  - **This tool**: ~$38/year
  - **Plaud subscription**: $699-948/year
  - **Your savings**: $660-910/year (95%+ cost reduction)

## Troubleshooting

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| `ffmpeg not found` | Install FFmpeg: `brew install ffmpeg` (macOS) |
| `Missing API Key` | Set `OPENAI_API_KEY` in `.env` file |
| Files not in Obsidian | Check `OBSIDIAN_VAULT_PATH` in `.env` |
| Processing fails | Check `logs/` folder for detailed error info |
| Audio format issues | Use supported formats: MP3, WAV, M4A, AAC |

## 🛠️ Advanced Usage

### Speaker Diarization
Speaker identification is **enabled by default** for multi-person meetings:
```bash
# Default: interactive speaker naming
uv run python main.py --file meeting.mp3

# Skip interactive naming (use SPEAKER_00, SPEAKER_01, etc.)
uv run python main.py --file meeting.mp3 --no-interactive

# Disable for self-recordings
uv run python main.py --file meeting.mp3 --no-speakers

# Combined with other features
uv run python main.py --file meeting.mp3 --context "team standup" --verbose
```

**Requirements:**
- `pyannote-audio` is included in dependencies (installed via `uv sync`)
- Set `HUGGINGFACE_TOKEN` in .env file
- Accept pyannote model license at https://huggingface.co/pyannote/speaker-diarization-3.1

### Resume from Transcript
Transcripts are automatically cached to `logs/transcripts/` after successful transcription. If processing fails later (e.g., during AI or file generation), resume without re-transcribing:
```bash
uv run python main.py --from-transcript logs/transcripts/audio_file_transcript.json --template default
```

### Custom Prompts
Edit `config.ini` to customize AI processing:
- **summary_prompt**: Change summary format and focus
- **action_items_prompt**: Modify action item extraction
- **title_generation_prompt**: Customize meeting titles

### Custom Templates
Create JSON files in `templates/` to customize output. Example:
```json
{
  "name": "Interview",
  "description": "Interview processing",
  "files": {
    "summary": {
      "enabled": true,
      "filename": "3_interview_summary.md",
      "prompt": "Summarize this interview by topic...",
      "template": "# Summary - {title}\n\n**Date:** {date}\n\n{content}"
    }
  }
}
```

**Template placeholders** - only these are supported in the `template` field:
- `{title}` - Generated meeting title
- `{date}` - Processing date (YYYY-MM-DD)
- `{content}` - AI-generated content from the prompt
- `{analysis_content}` - Same as `{content}` (alias)

### Batch Processing
```bash
# Process all MP3 files in a directory
for file in /your_path_to_recordings_here/*.mp3; do
    uv run python main.py --file "$file"
done
```

### Development

#### Adding New Features
1. **Audio Formats**: Update `supported_formats` in config
2. **AI Prompts**: Modify prompts in `config.ini`  
3. **Output Templates**: Edit `ObsidianGenerator` class methods

#### Contributing
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

### Testing

#### Step-by-Step Testing Guide

1. **Verify Setup**
   ```bash
   # Check dependencies are installed
   uv pip list | grep -E "(openai|librosa|pydub|python-dotenv)"

   # Verify config files exist
   ls -la .env config.ini
   ```

2. **Test with Sample Audio**
   ```bash
   # Create test folder structure
   mkdir -p input_audio test_outputs

   # Record a short test audio (or use any MP3/WAV file)
   # Place in input_audio/ folder

   # Test single file processing
   uv run python main.py --file input_audio/your_test_file.mp3
   ```

3. **Verify Output Structure**
   ```bash
   # Check generated files
   ls -la meeting_data/
   
   # Expected structure:
   # meeting_data/YYYY-MM-DD_Generated-Title/
   #   ├── meeting_meta.md
   #   ├── meeting_transcript.md  
   #   ├── meeting_summary.md
   #   └── meeting_action_items.md
   ```

4. **Test Continuous Monitoring**
   ```bash
   # Start monitoring (in one terminal)
   uv run python main.py --monitor

   # Add new file (in another terminal)
   cp test_file.mp3 input_audio/new_recording.mp3

   # Should auto-process within seconds
   ```

5. **Validate Obsidian Integration**
   - Import `meeting_data` folder into Obsidian vault
   - Check cross-links work (click [[meeting_transcript]] links)
   - Verify markdown formatting displays correctly

#### Testing Checklist

- [ ] Environment variables loaded correctly
- [ ] Audio file validation works
- [ ] Whisper transcription completes
- [ ] GPT-4o generates summaries
- [ ] Action items extracted properly
- [ ] Markdown files created with proper formatting
- [ ] Cross-links between files work
- [ ] Processed files moved to archive
- [ ] Logs written to logs/ folder
- [ ] Error handling works with corrupted files

#### Common Test Scenarios

```bash
# Test different audio formats
uv run python main.py --file test.mp3
uv run python main.py --file test.wav
uv run python main.py --file test.m4a

# Test speaker diarization (enabled by default)
uv run python main.py --file meeting.mp3
uv run python main.py --file meeting.mp3 --no-interactive
uv run python main.py --file meeting.mp3 --no-speakers

# Test error handling
uv run python main.py --file nonexistent.mp3  # Should handle gracefully
uv run python main.py --file empty.mp3        # Should detect and skip

# Test configuration
uv run python main.py --config test_config.ini --file test.mp3
```

## 🌟 Why Choose This Over Plaud's Service?

### ✅ Advantages
- **95%+ cost savings** - Pay only for what you use
- **Universal compatibility** - Any audio device works
- **Better AI** - Latest OpenAI models vs. basic transcription
- **Speaker identification** - Advanced diarization with interactive naming
- **Full control** - Your data, your processing, your pace
- **Obsidian integration** - Seamless knowledge management
- **Multi-language support** - Maintains original language
- **Open source** - Customize and extend as needed

### 📈 Perfect For
- **Frequent meeting attendees** - Researchers, consultants, managers
- **Obsidian users** - Seamless knowledge base integration  
- **Cost-conscious users** - Avoid expensive subscriptions
- **Privacy-focused users** - Process recordings locally
- **Multi-language users** - Russian, English, and other languages

---

## 📄 License

MIT License - Feel free to modify and distribute.

## 🤝 Contributing

Contributions welcome! Please see the contributing guidelines and submit pull requests for any improvements.

## ⭐ Show Your Support

If this project saves you money and time, please give it a star on GitHub!