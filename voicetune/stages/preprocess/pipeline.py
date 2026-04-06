"""Core audio pre-processing functions.

Converts raw call recordings into normalized 16kHz 16-bit WAV files,
with optional channel separation for stereo inputs.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

from voicetune.common import (
    DEFAULT_SILENCE_THRESHOLD_DB,
    DEFAULT_TARGET_LUFS,
    DEFAULT_TARGET_SR,
    decode_audio,
    estimate_snr,
    normalize_loudness,
    resample,
    split_channels,
    to_mono,
    trim_silence,
    write_json,
    write_wav,
)

log = logging.getLogger(__name__)


def process_file(input_path: Path, output_dir: Path) -> None:
    """Run the full pre-processing pipeline on a single audio file."""
    stem = input_path.stem
    file_output_dir = output_dir / stem
    file_output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Processing: {input_path.name}")

    audio, orig_sr, n_channels = decode_audio(input_path)
    original_duration = len(audio) / orig_sr

    audio = resample(audio, orig_sr, DEFAULT_TARGET_SR)
    audio = trim_silence(audio, DEFAULT_TARGET_SR)
    audio = normalize_loudness(audio, DEFAULT_TARGET_SR)

    output_files = []

    if n_channels > 1:
        mono_mix = to_mono(audio)
        full_path = file_output_dir / "full_normalized.wav"
        write_wav(full_path, mono_mix, DEFAULT_TARGET_SR)
        output_files.append(full_path.name)

        for i, ch in enumerate(split_channels(audio)):
            ch_path = file_output_dir / f"channel_{i}.wav"
            write_wav(ch_path, ch, DEFAULT_TARGET_SR)
            output_files.append(ch_path.name)
    else:
        audio = to_mono(audio)
        full_path = file_output_dir / "full_normalized.wav"
        write_wav(full_path, audio, DEFAULT_TARGET_SR)
        output_files.append(full_path.name)

    processed_duration = len(audio) / DEFAULT_TARGET_SR
    snr = estimate_snr(audio, DEFAULT_TARGET_SR)
    log.info(f"  SNR: {snr:.1f} dB")

    metadata = {
        "original_file": input_path.name,
        "original_format": input_path.suffix.lstrip("."),
        "original_sample_rate": orig_sr,
        "original_channels": n_channels,
        "original_duration_seconds": round(original_duration, 2),
        "output_sample_rate": DEFAULT_TARGET_SR,
        "output_bit_depth": 16,
        "output_duration_seconds": round(processed_duration, 2),
        "target_lufs": DEFAULT_TARGET_LUFS,
        "silence_threshold_db": DEFAULT_SILENCE_THRESHOLD_DB,
        "snr_db": round(snr, 1),
        "output_files": output_files,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(file_output_dir / "metadata.json", metadata, ensure_ascii=True)

    log.info(
        f"Done: {input_path.name} -> {file_output_dir.name}/ "
        f"({original_duration:.1f}s -> {processed_duration:.1f}s, "
        f"{n_channels}ch, {orig_sr}Hz -> {DEFAULT_TARGET_SR}Hz)"
    )
