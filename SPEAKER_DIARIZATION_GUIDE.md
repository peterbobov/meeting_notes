# Speaker Diarization Guide

## Overview

The Plaud Processor now includes speaker diarization capabilities, allowing you to identify and label different speakers in your meeting recordings. This feature adds speaker names to transcripts, generates speaker statistics, and enables more accurate AI processing with speaker context.

## Features

### ✅ Core Functionality
- **Voice Activity Detection**: Identifies speech segments vs silence
- **Speaker Clustering**: Groups voice segments by speaker identity
- **Interactive Naming**: User-friendly workflow to assign names to speakers
- **Speaker Statistics**: Speaking time, participation metrics, and turn-taking analysis
- **Enhanced Transcripts**: Timestamped transcripts with speaker labels
- **AI Integration**: Speaker-aware summaries and action items

### ✅ Technical Implementation
- **Local Processing**: 100% on-device using pyannote-audio
- **M2 Mac Optimized**: Metal Performance Shaders (MPS) acceleration
- **Security**: Voice biometric data never leaves your device
- **Efficient**: Parallel processing with existing transcription pipeline

### 🆕 Quality Improvements (Latest Update)
- **Timestamp Alignment Tolerance**: ±0.5s tolerance for audio processing drift compensation
- **Purity Scoring Algorithm**: Detects and rejects multi-speaker contaminated segments
- **Isolation Scoring**: Analyzes temporal separation between different speakers
- **Quote Validation Pass**: Final quality assurance with conversation marker detection
- **Comprehensive Diagnostic Logging**: Detailed debug output for troubleshooting
- **Chronological Prioritization**: Early meeting segments preferred for cleaner identification

## Installation & Setup

### 1. Install Dependencies

```bash
# Install speaker diarization dependencies
pip install pyannote-audio>=3.1.0
```

### 2. HuggingFace Token Setup

Speaker diarization requires a HuggingFace token and model license acceptance:

1. **Create HuggingFace Account**: Sign up at https://huggingface.co/
2. **Get Access Token**: 
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with "Read" permissions
3. **Accept Model License**:
   - Visit https://huggingface.co/pyannote/speaker-diarization-3.1
   - Accept the user agreement
4. **Add Token to Environment**:
   ```bash
   # Add to your .env file
   HUGGINGFACE_TOKEN=your_token_here
   ```

### 3. Test Installation

```bash
# Test speaker diarization setup
python test_speakers.py

# Test with an audio file
python test_speakers.py /path/to/meeting.mp3
```

## Usage

### Basic Speaker Identification

```bash
# Enable speaker identification
python main.py --file meeting.mp3 --speakers
```

This will:
1. Transcribe the audio using Whisper
2. Perform speaker diarization in parallel
3. Prompt you to name each speaker with sample quotes
4. Generate speaker-aware transcripts and summaries

### Interactive Naming Workflow

When speakers are detected, you'll see:

```
Found 3 speakers in this meeting.
Please provide names for each speaker (press Enter to keep default name):

Speaker A said: "I think we should focus on the quarterly targets..."
Please enter name for Speaker A: John Smith

Speaker B said: "That's a good point, but we also need to consider..."
Please enter name for Speaker B: Sarah Johnson

Speaker C said: "I agree with both perspectives..."
Please enter name for Speaker C: Mike Chen
```

### Skip Interactive Naming

```bash
# Keep default speaker names (Speaker A, B, C)
python main.py --file meeting.mp3 --speakers --no-interactive
```

### Combined with Other Features

```bash
# Speaker identification with custom context
python main.py --file meeting.mp3 --speakers --context "Sprint planning meeting"

# Speaker identification without action items
python main.py --file meeting.mp3 --speakers --no-action-items

# Continuous monitoring with speaker identification
python main.py --monitor --speakers
```

## Output Format

### Enhanced Transcript

```markdown
# Meeting Transcript - Project Planning

**Date:** 2024-06-21  
**Duration:** 23:15  
**Tags:** #transcript #meeting

## Transcript

[00:00] John Smith: Good morning everyone, let's start with the quarterly updates.
[00:15] Sarah Johnson: Thanks John. I wanted to discuss the new feature requirements.
[00:42] Mike Chen: I have some concerns about the timeline we discussed.
[01:10] John Smith: That's a valid point Mike. Let's adjust the schedule accordingly.
```

### Speaker Statistics

```markdown
# Meeting Summary - Project Planning

**Date:** 2024-06-21  
**Tags:** #summary #meeting

## Key Points
- Quarterly targets discussed and approved
- Feature requirements need refinement
- Timeline adjustments agreed upon

## Speaker Statistics

**John Smith:**
- Speaking time: 420.5s (45.2%)
- Segments: 23
- Average segment: 18.3s

**Sarah Johnson:**
- Speaking time: 315.8s (34.1%)
- Segments: 18
- Average segment: 17.5s

**Mike Chen:**
- Speaking time: 193.2s (20.7%)
- Segments: 12
- Average segment: 16.1s
```

## Technical Details

### Processing Pipeline

1. **Audio Preprocessing**: Convert to 16kHz mono for optimal diarization
2. **Parallel Processing**: Transcription and diarization run simultaneously
3. **Speaker Clustering**: Group voice segments by speaker identity
4. **Transcript Merging**: Combine speech recognition with speaker labels
5. **Interactive Naming**: User assigns names to detected speakers
6. **Output Generation**: Create speaker-aware markdown files

### Performance Characteristics

- **Accuracy**: 85-95% speaker identification accuracy (depends on audio quality)
- **Speed**: ~30-50% additional processing time for diarization
- **Memory**: ~1-2GB additional RAM usage during processing
- **Minimum Duration**: Works best with segments >1 second
- **Optimal Conditions**: Clear audio, distinct speakers, minimal background noise

### Configuration Options

```python
# Speaker diarization service configuration
SpeakerDiarizationService(
    confidence_threshold=0.7,     # Minimum confidence for speaker segments
    min_segment_duration=1.0,     # Minimum duration for speaker segments (seconds)
    device="mps"                  # Use M2 Mac acceleration
)
```

## Troubleshooting

### Common Issues

#### 1. HuggingFace Token Error
```
Error: Failed to load pipeline: Unauthorized
```
**Solution**: 
- Check HUGGINGFACE_TOKEN in .env file
- Ensure token has correct permissions
- Accept model license agreement

#### 2. No Speakers Detected
```
No speakers detected, continuing without speaker info
```
**Possible causes**:
- Audio quality too poor
- Single speaker recording
- Background noise interference
- Very short audio segments

**Solutions**:
- Check audio quality and volume
- Adjust `min_segment_duration` setting
- Use noise reduction preprocessing

#### 3. Memory Issues
```
RuntimeError: CUDA out of memory / MPS out of memory
```
**Solution**:
- Reduce audio file size
- Process shorter segments
- Lower confidence threshold
- Restart Python process

#### 4. Poor Speaker Separation
```
Speakers incorrectly merged or split
```
**Solutions**:
- Improve audio quality
- Ensure speakers have distinct voices
- Check microphone placement
- Adjust confidence threshold

#### 5. Multi-Speaker Quote Contamination (Fixed in Latest Update)
```
Quote snippets contain quotes from multiple speakers
```
**Now Automatically Handled**:
- Purity scoring algorithm detects contaminated segments
- Isolation scoring analyzes speaker separation
- Quote validation pass removes problematic quotes
- Enhanced logging shows rejection reasons

#### 6. Timestamp Drift Issues (Fixed in Latest Update)  
```
Speaker marked as speaking from 00:00 but started later
```
**Now Automatically Handled**:
- ±0.5s alignment tolerance compensates for processing drift
- Timestamp validation detects suspicious 00:00 segments
- Diagnostic logging tracks alignment quality
- Fallback mechanisms for drift recovery

### Performance Optimization

#### For Best Results:
- **Audio Quality**: Use clear recordings with distinct speakers
- **Microphone Setup**: Individual microphones when possible
- **Background Noise**: Minimize ambient sound
- **Speaker Consistency**: Avoid voice changes (illness, fatigue)

#### Speed Optimization:
- **Parallel Processing**: Diarization runs alongside transcription
- **M2 Acceleration**: Automatically uses Metal Performance Shaders
- **Memory Management**: Automatic cleanup and garbage collection

### Enhanced Debugging (Latest Update)

The latest version includes comprehensive diagnostic logging to help troubleshoot speaker identification issues:

#### Enable Verbose Logging
```bash
# Get detailed debug output for speaker diarization
python main.py --file meeting.mp3 --speakers --verbose
```

#### What the Enhanced Logging Shows:
- **Timestamp Alignment Statistics**: Perfect vs tolerance-based matches
- **Purity Analysis**: Why segments are rejected for contamination
- **Isolation Scoring**: How well speakers are separated temporally
- **Quote Selection Process**: Scoring breakdown for sample quotes
- **Validation Results**: Quality checks on selected quotes

#### Sample Debug Output:
```
[SPEAKER_DIARIZATION] Timestamp Alignment Quality:
[SPEAKER_DIARIZATION] - Perfect matches: 156 (78.4%)
[SPEAKER_DIARIZATION] - Tolerance matches: 32 (16.1%)
[SPEAKER_DIARIZATION] - No matches: 11 (5.5%)

[SPEAKER_DIARIZATION] ⚠️  REJECTED impure segment for SPEAKER_01 (purity: 0.45) at 125.3s: 'да, согласен, но нужно учесть...'

[SPEAKER_DIARIZATION] 🔍 Purity analysis for SPEAKER_02 at 89.7s:
  - Duration: 12.4s
  - Isolation score: 0.70
  - Overlap penalty: 0.85
  - Final purity: 0.82

[SPEAKER_DIARIZATION] ✅ Passed validation for SPEAKER_01 (score: 1.00)
```

## Integration with Existing Workflow

### Obsidian Integration

Speaker-identified transcripts integrate seamlessly with existing Obsidian workflows:

- **Cross-references**: Links between meta, transcript, summary, and action items
- **Tags**: Automatic tagging with speaker information
- **Search**: Find meetings by speaker participation
- **Templates**: Consistent formatting across all generated files

### AI Processing Enhancement

Speaker identification improves AI processing:

- **Context-Aware Summaries**: GPT-4o understands who said what
- **Speaker-Specific Action Items**: Automatically assign tasks to speakers
- **Participation Analysis**: Identify discussion patterns and engagement
- **Meeting Roles**: Recognize facilitators, contributors, and decision makers

## Future Enhancements

### Planned Features
- **Speaker Profile Learning**: Remember speakers across meetings
- **Voice Biometric Matching**: Automatic speaker recognition
- **Advanced Statistics**: Turn-taking analysis, interruption detection
- **Export Options**: Speaker-specific transcript exports
- **Integration**: Calendar app integration for speaker context

### Advanced Configuration
- **Custom Models**: Support for domain-specific diarization models
- **Quality Metrics**: Confidence scoring and quality assessment
- **Batch Processing**: Multiple file processing with speaker consistency
- **API Integration**: External speaker databases and recognition services

## Security & Privacy

### Data Protection
- **Local Processing**: All voice analysis happens on-device
- **No Cloud Upload**: Voice biometric data never leaves your Mac
- **Temporary Files**: Automatic cleanup of processing artifacts
- **User Control**: Full control over speaker naming and data retention

### Privacy Considerations
- **Speaker Consent**: Ensure participants consent to voice analysis
- **Data Retention**: Consider local storage policies for voice data
- **Access Control**: Secure storage of generated speaker profiles
- **Anonymization**: Option to use generic speaker labels

## Support & Resources

### Documentation
- **Main Documentation**: [README.md](README.md)
- **Technical Specs**: [CLAUDE.md](CLAUDE.md)
- **Installation Guide**: [Installation instructions in README](README.md#installation)

### Testing
- **Test Suite**: `python test_speakers.py`
- **Sample Audio**: Provide test recordings for validation
- **Performance Benchmarks**: Monitor processing time and accuracy

### Community
- **Issues**: Report problems via GitHub issues
- **Discussions**: Feature requests and usage questions
- **Contributions**: Code improvements and bug fixes welcome

---

*Generated with Claude Code - Speaker Diarization v1.0*