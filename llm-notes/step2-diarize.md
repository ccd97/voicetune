# Step 2: Diarize (Speaker Diarization + Transcription)

## Purpose
Identify who spoke when and transcribe each segment. Combines steps 2 and 3 from the initial plan (diarization + transcription) into a single module.

## Module
`voicetune/diarize/` — run via `python -m voicetune.diarize --mode {aws|whisperx|mlx|llamacpp}`

## CLI Args
| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | (required) | Backend: `aws`, `whisperx`, `mlx`, or `llamacpp` |
| `--input-dir` | `./output/preprocessed` | Preprocessed WAV directory |
| `--output-dir` | `./output/diarized` | Where to write diarized JSON |
| `--num-speakers` | auto-detect | Expected speaker count |
| `--language` | auto-detect | Force language code (e.g. `en-US`, `hi-IN`) |

## Four Backends

### AWS Transcribe (`aws.py`)
- **Batch mode:** all files are uploaded and submitted as concurrent TranscriptionJobs up front, then polled together in a single loop (15s intervals). Results are collected and S3 cleaned up as each job finishes. Per-file failures don't block the rest.
- `diarize()` still works for single-file use; `diarize_batch()` is used by the CLI for multi-file runs
- Auto-detects language between `en-US`, `hi-IN`, `mr-IN` (overridable)
- Speaker labels via `ShowSpeakerLabels` setting (max 3 by default)
- Parses word-level speaker map and groups into turns
- **Requires:** `AWS_S3_BUCKET`, `AWS_REGION` env vars, boto3

### WhisperX (`whisperx.py`)
- Local GPU (CUDA) or CPU transcription with Whisper large-v3
- On Apple Silicon (MPS available), transcription is automatically swapped to `mlx-whisper` (`mlx-community/whisper-large-v3-turbo`); alignment and diarization still run through `whisperx` on CPU
- Aligns word timestamps with `whisperx.align()`
- Speaker diarization via pyannote through WhisperX's `DiarizationPipeline`
- Assigns word-level speaker labels, then merges into turns
- **Requires:** `HF_TOKEN` env var, torch, whisperx, pyannote model access (plus `mlx-whisper` on Apple Silicon)

### MLX (`mlx.py`)
- Apple Silicon native — uses mlx-whisper (Metal GPU) for transcription
- Uses `mlx-community/whisper-large-v3-turbo` model
- pyannote for diarization (uses MPS backend on Apple Silicon)
- Word-level speaker assignment via midpoint overlap with pyannote timeline
- Majority vote for segment-level speaker from word speakers
- **Requires:** `HF_TOKEN` env var, mlx-whisper, pyannote.audio, torch

### llama.cpp (`llamacpp.py`)
- Single-model transcription + diarization using Gemma 4's audio conformer encoder via llama.cpp
- No separate diarization step — the model identifies speakers natively from the audio
- Audio WAV is base64-encoded and passed through the mtmd multimodal API
- Model is cached across files within a single run (GGUF loading is expensive)
- **Does not produce timestamps** — all turns have `start: 0.0, end: 0.0`
- Prompts model for JSON output with speaker labels; falls back to regex parsing
- **Requires:** `LLAMACPP_MODEL_PATH` env var (GGUF model), `LLAMACPP_MMPROJ_PATH` env var (mmproj), llama-cpp-python (built with `LLAVA_BUILD=ON`)

## Input/Output

**Input:** `output/preprocessed/*/full_normalized.wav`

**Output:**
```json
// output/diarized/{call_id}_diarized.json
{
  "call_id": "call_recording",
  "mode": "aws|whisperx|mlx|llamacpp",
  "language": "en-US",
  "turns": [
    {"speaker": "spk_0", "start": 0.0, "end": 3.2, "text": "Hi, how can I help?"},
    {"speaker": "spk_1", "start": 3.5, "end": 8.1, "text": "I'm calling about my bill..."}
  ]
}
```

## Shared Utilities (`utils.py`)
- `get_call_id()` — derives call ID from path (parent dir name if file is `full_normalized.wav`)
- `find_preprocessed_wavs()` — finds `*/full_normalized.wav` or falls back to `*.wav`
- `save_result()` — writes JSON output
- `join_words()` — attaches punctuation to preceding word

## Key Implementation Details
- All backends produce the same output format (call_id, mode, language, turns[])
- Backends are lazy-imported to avoid loading heavy deps unnecessarily
- Turn merging (consecutive same-speaker) happens later in the segment step, not here
- The `full_normalized.wav` mono mixdown is always used, even if stereo channels exist
