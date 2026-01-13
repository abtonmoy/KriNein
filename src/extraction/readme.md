# Ad Content Extraction Module

A sophisticated video advertisement analysis system that uses vision-language models (VLMs) to extract structured information from video ads, including brand details, promotional content, sentiment analysis, and engagement metrics.

## Overview

This module processes video advertisements by:

1. Sampling keyframes from videos at strategic temporal positions
2. Sending frames to vision-language models (Claude, GPT-4V, or Gemini)
3. Extracting structured ad metadata including brand info, CTAs, sentiment, and topics
4. Supporting adaptive schema selection based on detected ad type
5. Incorporating audio transcription context when available

## Features

- **Multi-Model Support**: Works with Anthropic Claude, OpenAI GPT-4V, and Google Gemini
- **Temporal Awareness**: Analyzes narrative progression across frames with timestamps
- **Adaptive Schemas**: Automatically detects ad type and adjusts extraction schema
- **Audio Context Integration**: Incorporates speech transcription and audio mood
- **Rich Taxonomy**: 38 topic categories and 30 sentiment types from video ad research
- **Engagement Metrics**: Predicts humor, excitement, and overall effectiveness

## Architecture

```
src/extraction/
├── llm_client.py      # LLM API clients and extraction orchestration
├── prompts.py         # Prompt engineering and frame preparation
└── schema.py          # Schema definitions and taxonomies
```

## Core Components

### 1. LLM Clients (`llm_client.py`)

Abstraction layer for multiple vision-language model providers:

```python
from extraction.llm_client import AdExtractor

extractor = AdExtractor(
    provider="anthropic",  # or "openai", "gemini", "mock"
    model="claude-sonnet-4-20250514",
    schema_mode="adaptive",  # or "fixed", "flexible"
    temporal_context=True
)

result = extractor.extract(
    frames=[(0.0, frame1), (5.0, frame2), (10.0, frame3)],
    video_duration=15.0,
    audio_context={"transcription": [...], "mood": "upbeat"}
)
```

**Supported Providers:**

- `anthropic`: Claude Sonnet 4 (default, recommended)
- `openai`: GPT-4V/GPT-4o
- `gemini`: Gemini 3.0 Flash
- `mock`: Testing without API calls

### 2. Schema System (`schema.py`)

Three schema modes for different use cases:

**Fixed Schema** - Comprehensive baseline extraction:

```python
{
    "brand": {...},
    "product": {...},
    "promotion": {...},
    "call_to_action": {...},
    "message": {...},
    "visual_elements": {...},
    "content_rating": {...},
    "topic": {...},
    "sentiment": {...},
    "engagement_metrics": {...}
}
```

**Adaptive Schema** - Detects ad type and adds specialized fields:

- `product_demo`: Features, demo steps
- `testimonial`: Speaker info, quotes, credibility
- `brand_awareness`: Emotional appeal, storytelling
- `tutorial`: Skills taught, instructional steps
- `entertainment`: Humor type, celebrity, viral elements

**Flexible Schema** - Streamlined for quick analysis with narrative focus

### 3. Prompt Engineering (`prompts.py`)

Advanced prompt construction with:

- Temporal context (timestamps, deltas, position labels)
- Narrative analysis instructions
- Audio transcription integration
- Topic/sentiment taxonomy references

```python
from extraction.prompts import build_temporal_prompt, prepare_frames_for_prompt

prepared_frames = prepare_frames_for_prompt(
    frames=[(0.0, frame1), (5.0, frame2)],
    video_duration=10.0,
    include_position_labels=True  # OPENING, MIDDLE, CLOSING
)

prompt = build_temporal_prompt(
    frames=prepared_frames,
    video_duration=10.0,
    schema=schema,
    audio_context=audio_data
)
```

## Taxonomy Systems

### Topic Classification (38 Categories)

Derived from video ad research datasets:

1. **Food & Beverage** (1-8): Restaurants, snacks, beverages
2. **Automotive** (9): Cars, parts, insurance, repair
3. **Technology** (10-11): Electronics, telecom services
4. **Services** (12-16): Financial, education, software, other
5. **Personal Care** (17-20): Beauty, healthcare, clothing, baby
6. **Home & Leisure** (21-29): Games, cleaning, appliances, travel, shopping
7. **Social Issues** (30-38): Environment, rights, safety, charity

### Sentiment Classification (30 Categories)

Maps to emotional responses the ad aims to evoke:

- **Energy**: Active, Eager, Inspired, Confident
- **Joy**: Cheerful, Amused, Loving, Grateful
- **Calm**: Peaceful, Conscious, Educated
- **Negative**: Afraid, Alarmed, Angry, Disturbed, Sad
- **Identity**: Fashionable, Feminine, Manly, Youthful

## Extracted Fields

### Brand Information

- `brand_name_text`: Company/brand name as displayed
- `logo_visible`: Boolean logo presence
- `brand_text_contrast`: Visual prominence (low/medium/high)
- `industry`: Business category

### Promotional Content

- `promo_present`: Boolean promotional offer detection
- `promo_text`: Core offer only (e.g., "50% off", not full description)
- `promo_deadline`: Time constraints
- `price_value`: Specific pricing

### Call to Action

- `cta_present`: Boolean CTA detection
- `cta_type`: Specific action (e.g., "Sign up button", "Download app")

### Content Analysis

- `primary_message`: Main value proposition
- `text_density`: Amount of on-screen text (low/medium/high)
- `text_overlays`: List of visible text
- `dominant_colors`: Color palette
- `is_nsfw`: Content safety rating

### Engagement Metrics

- `is_funny`: 0.0-1.0 humor rating
- `is_exciting`: 0.0-1.0 excitement rating
- `effectiveness_score`: 1-5 predicted effectiveness

## Usage Examples

### Basic Extraction

```python
from extraction.llm_client import create_extractor

# From config dict
config = {
    "extraction": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
        "schema": {"mode": "adaptive"},
        "temporal_context": {
            "enabled": True,
            "include_timestamps": True,
            "include_time_deltas": True
        }
    }
}

extractor = create_extractor(config)

# Extract from video frames
result = extractor.extract(
    frames=sampled_frames,
    video_duration=30.0
)

print(f"Brand: {result['brand']['brand_name_text']}")
print(f"Topic: {result['topic']['topic_name']}")
print(f"Sentiment: {result['sentiment']['primary_sentiment_name']}")
print(f"Effectiveness: {result['engagement_metrics']['effectiveness_score']}/5")
```

### With Audio Context

```python
audio_context = {
    "transcription": [
        {"start": 0.0, "end": 2.5, "text": "Introducing the all-new..."},
        {"start": 2.5, "end": 5.0, "text": "50% off for a limited time"}
    ],
    "mood": "upbeat",
    "key_phrases": [
        {"text": "limited time offer", "timestamp": 3.0}
    ]
}

result = extractor.extract(
    frames=frames,
    video_duration=15.0,
    audio_context=audio_context
)
```

### Ad Type Detection Only

```python
ad_type = extractor.detect_ad_type(prepared_frames)
# Returns: "product_demo", "testimonial", "brand_awareness", etc.
```

## Configuration Options

```python
AdExtractor(
    provider="anthropic",           # LLM provider
    model="claude-sonnet-4-20250514",  # Model name
    max_tokens=2000,                # Response length limit
    temperature=0.0,                # Sampling temperature (0=deterministic)
    schema_mode="adaptive",         # Schema selection strategy
    temporal_context=True,          # Enable temporal analysis
    include_timestamps=True,        # Show frame timestamps
    include_time_deltas=True,       # Show gaps between frames
    include_position_labels=True,   # Add OPENING/MIDDLE/CLOSING labels
    include_narrative_instructions=True  # Add narrative analysis guidance
)
```

## Performance Considerations

### Frame Sampling Strategy

- **Opening** (0-15%): Brand intro, hook
- **Middle** (40-60%): Core message, product demo
- **Closing** (85-100%): CTA, logo reinforcement

Typical sampling: 3-8 frames for 15-60 second ads

### Token Optimization

- Frames resized to max 512px dimension
- JPEG quality 85% for API efficiency
- Audio transcription limited to first 10 segments
- Prompt engineering minimizes redundancy

### Cost Estimates (per ad)

- **Claude Sonnet 4**: ~$0.10-0.30 for 5 frames
- **GPT-4o**: ~$0.15-0.40 for 5 frames
- **Gemini Flash**: ~$0.03-0.08 for 5 frames

## Error Handling

```python
try:
    result = extractor.extract(frames, video_duration)

    if "error" in result:
        print(f"Extraction failed: {result['error']}")
        if "raw_response" in result:
            print(f"Raw response: {result['raw_response']}")
    else:
        # Process successful extraction
        pass

except Exception as e:
    logger.error(f"Extraction exception: {e}")
```

## Integration with Video Pipeline

```python
# Typical workflow
from video_processing import extract_keyframes
from audio_processing import transcribe_audio
from extraction.llm_client import create_extractor

# 1. Extract keyframes
keyframes = extract_keyframes(video_path, num_frames=5)

# 2. Transcribe audio (optional)
audio_context = transcribe_audio(video_path)

# 3. Extract metadata
extractor = create_extractor(config)
metadata = extractor.extract(
    frames=keyframes,
    video_duration=get_video_duration(video_path),
    audio_context=audio_context
)

# 4. Store results
save_metadata(metadata, output_path)
```

## Research Background

The topic and sentiment taxonomies are based on:

- Video Advertisement Dataset analysis
- 38 industry-standard topic categories
- 30 validated emotional response categories
- Engagement metrics from ad effectiveness research

## Future Enhancements

- [ ] Multi-language support for international ads
- [ ] Object detection integration for product recognition
- [ ] Scene segmentation for detailed narrative analysis
- [ ] A/B testing framework for prompt optimization
- [ ] Batch processing for large-scale analysis
- [ ] Fine-tuned models for specific ad verticals

## Dependencies

```
anthropic>=0.18.0
openai>=1.0.0
google-generativeai>=0.3.0
opencv-python>=4.8.0
pillow>=10.0.0
numpy>=1.24.0
```
