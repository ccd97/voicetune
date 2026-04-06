# Step 1: Preprocess

## Purpose
Convert raw call recordings into normalized 16kHz 16-bit mono WAV files suitable for downstream diarization and transcription.

## Module
`voicetune/stages/preprocess/` — run via `python -m voicetune.stages.preprocess`

## CLI Args
| Flag | Default | Description |
|------|---------|-------------|
| `--run-dir` | `./output` | Base output directory |
| `--input-dir` | `./input` | Directory with raw audio files |
| `--output-dir` | `<run-dir>/preprocessed` | Where to write processed output |

## What It Does
1. Decode any supported format (.m4a, .mp3, .wav, .flac, .ogg, .aac, .wma, .opus, .3gp) to float32 via PyAV.
2. Resample to 16 kHz using soxr (HQ).
3. Trim leading/trailing silence via frame-level RMS (threshold −40 dB).
4. Normalize loudness to −23 LUFS (EBU R128 / ITU-R BS.1770 via pyloudnorm). Peak-normalize instead for clips < 0.4 s.
5. If stereo, write `full_normalized.wav` (mono mixdown) plus one `channel_N.wav` per channel.
6. Estimate SNR (top 10% vs bottom 10% frame RMS) and log it.
7. Write `metadata.json` with original/output format, duration, SNR, timestamps.

## Input/Output

**Input:** `input/*.m4a` (or any supported format)

**Output per file:**
```
output/preprocessed/{stem}/
  full_normalized.wav    # mono 16kHz 16-bit PCM
  channel_0.wav          # (only if stereo)
  channel_1.wav          # (only if stereo)
  metadata.json          # processing metadata
```

## Key Constants
Defined in `voicetune/common/audio.py`:
- `DEFAULT_TARGET_SR = 16000`
- `DEFAULT_TARGET_LUFS = -23.0`
- `DEFAULT_SILENCE_THRESHOLD_DB = -40.0`

## Dependencies
`av`, `numpy`, `pyloudnorm`, `soxr` (+ `soundfile` via `voicetune.common.write_wav`)

## Key Implementation Details
- Decoding uses PyAV (not ffmpeg CLI) — all container formats handled natively.
- Audio is float32 internally, clipped to [−1, 1] on WAV write.
