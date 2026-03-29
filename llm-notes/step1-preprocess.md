# Step 1: Preprocess

## Purpose
Convert raw call recordings into normalized 16kHz 16-bit mono WAV files suitable for downstream diarization and transcription.

## Module
`voicetune/preprocess/` — run via `python -m voicetune.preprocess`

## CLI Args
| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir` | `./input` | Directory with raw audio files |
| `--output-dir` | `./output/preprocessed` | Where to write processed output |

## What It Does
1. **Decode** any supported format (.m4a, .mp3, .wav, .flac, .ogg, .aac, .wma, .opus, .3gp) to float32 numpy array using PyAV
2. **Resample** to 16kHz using soxr (HQ quality)
3. **Trim silence** at start/end using frame-level RMS energy (threshold: -40 dB)
4. **Normalize loudness** to -23 LUFS using EBU R128 / ITU-R BS.1770 (pyloudnorm). Falls back to peak normalization for clips < 0.4s
5. **Split channels** if stereo — writes `full_normalized.wav` (mono mixdown) + individual `channel_N.wav` files
6. **Estimate SNR** (top 10% vs bottom 10% frame RMS) and log it
7. **Write metadata.json** with original/output format details, duration, SNR, timestamps

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
- `TARGET_SR = 16000`
- `TARGET_LUFS = -23.0`
- `SILENCE_THRESHOLD_DB = -40.0`

## Dependencies
`av`, `numpy`, `pyloudnorm`, `soxr` (+ `soundfile` via `voicetune.common.write_wav`)

## Key Implementation Details
- Uses PyAV (not ffmpeg CLI) for decoding — handles all container formats natively
- Audio is always float32 internally, clipped to [-1, 1] on WAV write
- The `full_normalized.wav` is what the diarize step picks up (via `find_preprocessed_wavs()`)
