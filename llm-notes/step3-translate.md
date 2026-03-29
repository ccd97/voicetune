# Step 3: Translate

## Purpose

Add English translations to non-English diarized transcripts. Uses Claude via Bedrock gateway for batch translation. Designed for multilingual calls (Hindi, Marathi) where the diarization output is in the original language.

## Module

`voicetune/translate/` — run via `python -m voicetune.translate`

## CLI Args


| Flag           | Default               | Description                        |
| -------------- | --------------------- | ---------------------------------- |
| `--input-dir`  | `./output/diarized`   | Directory with diarized JSON files |
| `--output-dir` | `./output/translated` | Where to write translated JSON     |


## What It Does

1. **Read** each `*_diarized.json` from the diarized output
2. **Check each turn** for whether it needs translation:
  - Skip if language code is English (`en-US`, `en-GB`, etc.)
  - Skip if text is predominantly Latin script (>70% Latin chars) — catches Indian English misdetected as Hindi
3. **Batch translate** non-English turns using Claude via Bedrock (batches of 20 turns per API call)
4. **Write** output with `text_en` field added to every turn (original text for English, translation for others)

## Input/Output

**Input:** `output/diarized/{call_id}_diarized.json`

**Output:**

```json
// output/translated/{call_id}_translated.json
{
  "call_id": "call_recording",
  "mode": "aws",
  "language": "hi-IN",
  "turns": [
    {
      "speaker": "spk_0", "start": 0.0, "end": 3.2,
      "text": "नमस्ते, मैं आपकी कैसे मदद कर सकता हूं?",
      "language": "hi-IN",
      "text_en": "Hello, how can I help you?"
    }
  ]
}
```

## Translation Details

- **Model:** Configurable via `TRANSLATE_MODEL` env var, defaults to `us.anthropic.claude-haiku-4-5-20251001-v1:0`
- **Prompt strategy:** Sends numbered dialogue turns in a single prompt, asks for conversational English
- **Special handling:** Transliterated English in Devanagari is converted back to proper English rather than literally translated
- **Batch size:** 20 turns per API call, `max_tokens=4096`

## Environment Variables


| Var                          | Description                     |
| ---------------------------- | ------------------------------- |
| `ANTHROPIC_BEDROCK_BASE_URL` | Bedrock gateway URL             |
| `ANTHROPIC_AUTH_TOKEN`       | Auth bearer token               |
| `TRANSLATE_MODEL`            | Claude model ID (optional)      |
| `NODE_EXTRA_CA_CERTS`        | Custom CA certs path (optional) |


## Dependencies

`httpx`, `python-dotenv`

## Key Implementation Details

- `is_latin_text()` uses Unicode codepoint check (`ord(c) < 0x0250`) to detect script
- Response parsing is line-based: splits on newlines, strips numbering prefix
- Each turn gets a `language` field (source) and `text_en` field (English translation or original)
- Uses httpx for HTTP (not boto3) — communicates with Bedrock via a gateway proxy

