# Step 4: Translate

## Purpose

Add English translations to non-English diarized transcripts. Supports two backends: Claude via Bedrock gateway, or a local GGUF model via llama-cpp-python. Designed for multilingual calls (Hindi, Marathi) where the diarization output is in the original language.

## Module

`voicetune/stages/translate/` — run via `python -m voicetune.stages.translate`

## CLI Args


| Flag           | Default               | Description                            |
| -------------- | --------------------- | -------------------------------------- |
| `--input-dir`  | `./output/diarized`   | Directory with diarized JSON files     |
| `--output-dir` | `./output/translated` | Where to write translated JSON         |
| `--backend`    | `llamacpp`            | Translation backend: `bedrock` or `llamacpp` |


## What It Does

1. **Read** each `*_diarized.json` from the diarized output
2. **Check each turn** for whether it needs translation:
  - Skip if language code is English (`en-US`, `en-GB`, etc.)
  - Skip if text is predominantly Latin script (>70% Latin chars) — catches Indian English misdetected as Hindi
3. **Batch translate** non-English turns (batches of 20 turns per API call) using the selected backend
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

- **Prompt strategy:** Sends numbered dialogue turns in a single prompt, asks for conversational English. Same prompt template (`translate.j2`) used by both backends.
- **Special handling:** Transliterated English in Devanagari is converted back to proper English rather than literally translated
- **Batch size:** 20 turns per API call, `max_tokens=4096`

### Bedrock backend

- **Model:** Configurable via `TRANSLATE_MODEL` env var, defaults to `us.anthropic.claude-haiku-4-5-20251001-v1:0`
- Uses httpx for HTTP (not boto3) — communicates with Bedrock via a gateway proxy

### llama.cpp backend

- Uses llama-cpp-python with `create_chat_completion` (text-only, no mmproj needed)
- Model cached across files within a single run via module-level `_get_llm()`
- Runs with `temperature=0` for deterministic output

## Environment Variables


| Var                          | Backend   | Description                     |
| ---------------------------- | --------- | ------------------------------- |
| `ANTHROPIC_BEDROCK_BASE_URL` | bedrock   | Bedrock gateway URL             |
| `ANTHROPIC_AUTH_TOKEN`       | bedrock   | Auth bearer token               |
| `TRANSLATE_MODEL`            | bedrock   | Claude model ID (optional)      |
| `NODE_EXTRA_CA_CERTS`        | bedrock   | Custom CA certs path (optional) |
| `LLAMACPP_MODEL_PATH`        | llamacpp  | Path to GGUF model file         |


## Dependencies

`httpx`, `python-dotenv`, `llama-cpp-python` (for llamacpp backend)

## Key Implementation Details

- `is_latin_text()` uses Unicode codepoint check (`ord(c) < 0x0250`) to detect script
- Response parsing is line-based: splits on newlines, strips numbering prefix (`_parse_translations`)
- Each turn gets a `language` field (source) and `text_en` field (English translation or original)
- `process_file` accepts a `backend` parameter; builds a `translate_fn` closure that dispatches to the right backend

