# Extraction Module

The `extraction` module handles LLM-based content extraction from video frames, including prompt building, schema definition, and multi-provider LLM client support.

## Overview

This module orchestrates the final stage of video analysis: extracting structured information from selected frames using Large Language Models (LLMs).

## Components

### AdExtractor

Main extractor class supporting multiple LLM providers.

```python
from video_analyzer.extraction import AdExtractor

extractor = AdExtractor(
    provider="anthropic",     # "anthropic", "openai", "google"
    model="claude-sonnet-4-6",# Model name
    temperature=0.0,          # Deterministic extraction
    max_tokens=4096
)

result = extractor.extract(
    frames=selected_frames,
    audio_context=transcription,
    scene_boundaries=scenes
)
```

**Supported providers:**
- **Anthropic**: Claude models (Sonnet, Opus)
- **OpenAI**: GPT-4 Vision, GPT-4o
- **Google**: Gemini 1.5/2.0 (native video support)
- **Mock**: For testing without API calls

### Prompts Module

Builds structured prompts for LLM extraction.

```python
from video_analyzer.extraction.prompts import (
    build_single_pass_prompt,
    build_segmented_prompt,
    prepare_frames_for_prompt
)

# Single-pass: type detection + extraction in one call
prompt = build_single_pass_prompt(
    frames=frames,
    duration=30.0,
    schema_fields=["brand", "product", "promotion"],
    audio_context=transcription
)

# Segmented: Group frames by scene for temporal awareness
prompt = build_segmented_prompt(
    frames=frames,
    duration=30.0,
    schema_fields=["brand", "product"],
    scenes=[(0.0, 5.0), (5.0, 15.0), (15.0, 30.0)]
)
```

### Schema Module

Defines output schemas and validates ad types.

```python
from video_analyzer.extraction.schema import (
    get_schema,
    get_valid_ad_types,
    FrameForPrompt
)

# Get base schema
schema = get_schema(ad_type="product")

# Valid ad types
ad_types = get_valid_ad_types()
# ['product', 'emotional', 'demo', 'testimonial', ...]

# Frame container
frame = FrameForPrompt(
    timestamp=2.5,
    base64_image="base64_encoded_image",
    position_label="OPENING"  # Optional
)
```

## Features

### Single-Pass Extraction

Combines ad type detection and content extraction in one LLM call (~50% cost reduction).

### Retry with Backoff

Automatic retry on transient errors with exponential backoff.

```python
from video_analyzer.extraction.llm_client import _retry_with_backoff

result = _retry_with_backoff(
    func=extraction_call,
    max_retries=3,
    base_delay=1.0
)
```

### Robust JSON Parsing

Handles non-deterministic LLM output formats:

- Markdown code blocks: ```json {...} ```
- Surrounded text: "Here is the result: {...}"
- Trailing commas: {"a": 1, "b": 2,}
- Array wrapper: [{...}] vs {...}

### Confidence Scoring

Appends confidence score to extraction results.

```python
result = extractor.extract(frames)
print(result["_metadata"]["confidence"])  # 0.0 - 1.0
```

## Configuration

```yaml
extraction:
  llm_provider: anthropic
  model: claude-sonnet-4-6
  temperature: 0.0
  max_tokens: 4096
  retry:
    max_retries: 3
    base_delay: 1.0
```

## Schema Fields

**Base schema fields:**
- `brand`: Brand name and description
- `product`: Product/service being advertised
- `promotion`: Promotional offers
- `call_to_action`: CTA text and type
- `visual_elements`: Key visual components
- `content_rating`: Age appropriateness
- `message`: Core advertising message
- `target_audience`: Intended audience
- `persuasion_techniques`: Marketing techniques used

**Type-specific extensions:**
- `emotional_appeal`: For emotional ads
- `demo_details`: For demonstration ads
- `testimonial_info`: For testimonial ads

## Dependencies

- **Anthropic SDK**: Claude API
- **OpenAI SDK**: GPT API
- **Google Generative AI**: Gemini API
- **Pillow**: Image processing
- **python-dotenv**: Environment variable loading

## Environment Variables

```bash
# .env file
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

## Output Format

```json
{
  "ad_type": "product",
  "brand": {
    "name": "Brand Name",
    "description": "..."
  },
  "product": {
    "name": "Product Name",
    "category": "Electronics"
  },
  "promotion": {
    "text": "50% off",
    "type": "discount"
  },
  "_metadata": {
    "confidence": 0.92,
    "processing_time": 2.3
  }
}
```
