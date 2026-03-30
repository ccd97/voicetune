# Step 8: Pair Format (Training Pair Generation)

## Purpose
Convert labeled dialogues into (input, output) training pairs for speech-to-speech model fine-tuning. Each pair has: context + prompt from "other" speaker as input, and "me" response as output.

## Module
`voicetune/pairformat/` — run via `python -m voicetune.pairformat`

## CLI Args
| Flag | Default | Description |
|------|---------|-------------|
| `--segmented-dir` | `./output/segmented` | Directory with segmented call output |
| `--output-dir` | `./output/pairs` | Where to write training pairs |
| `--context-turns` | `4` | Max previous turns to include as context |
| `--min-duration` | `0.5` | Skip "me" responses shorter than this (seconds) |

## What It Does
1. **Scan** each `dialogue.json` for turns where `speaker_label == "me"` responds
2. **For each "me" response:**
   - Find the preceding "other" turn (the prompt)
   - Skip if no prompt exists (me speaks first)
   - Gather up to N context turns before the prompt
3. **Generate audio files:**
   - `{pair_id}_input.wav` — context turns + prompt concatenated with 0.3s silence gaps
   - `{pair_id}_prompt.wav` — just the prompt turn
   - `{pair_id}_response.wav` — the "me" response (training target)
4. **Write** a pairs.json manifest with text, speaker info, and audio paths

## Input/Output

**Input:** `output/segmented/{call_id}/dialogue.json` (with `speaker_label` from label step)

**Output:**
```
output/pairs/{call_id}/
  pairs.json             # manifest with all pairs
  audio/
    {call_id}_pair_001_input.wav      # context + prompt concatenated
    {call_id}_pair_001_prompt.wav     # prompt only
    {call_id}_pair_001_response.wav   # "me" response (target)
    ...
```

**pairs.json structure:**
```json
{
  "call_id": "call_recording",
  "language": "en-US",
  "num_pairs": 24,
  "context_turns": 4,
  "pairs": [
    {
      "pair_id": "call_recording_pair_001",
      "call_id": "call_recording",
      "input": {
        "audio_path": "audio/..._input.wav",
        "prompt_audio_path": "audio/..._prompt.wav",
        "prompt_text": "What's your account number?",
        "prompt_speaker": "other",
        "prompt_duration": 2.5,
        "context": [
          {"speaker": "me", "text": "..."},
          {"speaker": "other", "text": "..."}
        ]
      },
      "output": {
        "audio_path": "audio/..._response.wav",
        "text": "Sure, it's 1234...",
        "speaker": "me",
        "duration": 3.1
      }
    }
  ]
}
```

## Key Implementation Details
- Only generates pairs when "me" responds to "other" — skips "me" speaking first
- Context audio is concatenated with 0.3s silence gaps (`CONTEXT_GAP_SECONDS`)
- Audio is 16-bit PCM WAV at 16kHz mono (clipped to [-1, 1])
- Context text preserves speaker_label ("me"/"other") or falls back to raw speaker ID
- The pair format is flexible — input audio includes full context for models that consume history, prompt is separate for models that don't

## Dependencies
`soundfile`, `numpy`
