"""Core audio pre-processing functions.

Converts raw call recordings into normalized 16kHz 16-bit WAV files,
with optional channel separation for stereo inputs.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import av
import numpy as np
import pyloudnorm as pyln
import soxr

from voicetune.common import write_wav

SUPPORTED_EXTENSIONS = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".aac", ".wma", ".opus", ".3gp"}
TARGET_SR = 16000
TARGET_LUFS = -23.0
SILENCE_THRESHOLD_DB = -40.0

log = logging.getLogger(__name__)


def decode_audio(path: Path) -> tuple[np.ndarray, int, int]:
    """Decode audio file to float32 numpy array using PyAV.

    Returns (audio, sample_rate, n_channels) where audio has shape (n_samples, n_channels).
    """
    container = av.open(str(path))
    stream = container.streams.audio[0]
    orig_sr = stream.rate
    n_channels = stream.channels

    resampler = av.AudioResampler(format="fltp", layout=stream.layout, rate=orig_sr)

    frames = []
    for frame in container.decode(audio=0):
        resampled = resampler.resample(frame)
        for r in resampled:
            frames.append(r.to_ndarray())

    for r in resampler.resample(None):
        frames.append(r.to_ndarray())

    container.close()

    if not frames:
        raise ValueError(f"No audio frames decoded from {path}")

    audio = np.concatenate(frames, axis=1).T.astype(np.float32)

    log.debug(f"Decoded {path.name}: {orig_sr}Hz, {n_channels}ch, {len(audio)/orig_sr:.1f}s")
    return audio, orig_sr, n_channels


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate using soxr."""
    if orig_sr == target_sr:
        return audio
    resampled = soxr.resample(audio, orig_sr, target_sr, quality="HQ")
    log.debug(f"Resampled {orig_sr}Hz -> {target_sr}Hz")
    return resampled.astype(np.float32)


def trim_silence(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = SILENCE_THRESHOLD_DB,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Trim leading and trailing silence based on frame-level RMS energy."""
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio

    n_frames = 1 + (len(mono) - frame_length) // hop_length
    if n_frames <= 0:
        return audio

    rms = np.array([
        np.sqrt(np.mean(mono[i * hop_length : i * hop_length + frame_length] ** 2))
        for i in range(n_frames)
    ])

    rms_db = 20 * np.log10(rms + 1e-10)

    above = np.where(rms_db > threshold_db)[0]
    if len(above) == 0:
        log.warning("Entire signal is below silence threshold — returning original")
        return audio

    start_sample = above[0] * hop_length
    end_sample = min(above[-1] * hop_length + frame_length, len(audio))

    trimmed = audio[start_sample:end_sample]
    log.debug(f"Trimmed {(len(audio) - len(trimmed)) / sr:.2f}s of silence")
    return trimmed


def normalize_loudness(
    audio: np.ndarray, sr: int, target_lufs: float = TARGET_LUFS
) -> np.ndarray:
    """Normalize audio to target LUFS using EBU R128 / ITU-R BS.1770."""
    meter = pyln.Meter(sr, block_size=0.4)

    duration = len(audio) / sr
    if duration < 0.4:
        log.warning(f"Audio too short ({duration:.2f}s) for loudness measurement — using peak normalization")
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio * (0.9 / peak)
        return audio

    loudness = meter.integrated_loudness(audio)

    if loudness == float("-inf"):
        log.warning("Measured loudness is -inf (silence) — skipping normalization")
        return audio

    normalized = pyln.normalize.loudness(audio, loudness, target_lufs)
    log.debug(f"Normalized loudness: {loudness:.1f} LUFS -> {target_lufs:.1f} LUFS")
    return normalized.astype(np.float32)


def split_channels(audio: np.ndarray) -> list[np.ndarray]:
    """Split multi-channel audio into a list of mono arrays."""
    if audio.ndim == 1:
        return [audio]
    return [audio[:, i] for i in range(audio.shape[1])]


def estimate_snr(audio: np.ndarray, sr: int, frame_length: int = 2048, hop_length: int = 512) -> float:
    """Estimate SNR by comparing speech frames to noise floor.

    Uses frame-level RMS: top 10% frames as signal, bottom 10% as noise.
    Returns SNR in dB. Higher = cleaner audio.
    """
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    n_frames = 1 + (len(mono) - frame_length) // hop_length
    if n_frames <= 0:
        return 0.0

    rms = np.array([
        np.sqrt(np.mean(mono[i * hop_length : i * hop_length + frame_length] ** 2))
        for i in range(n_frames)
    ])

    rms_sorted = np.sort(rms)
    n10 = max(1, n_frames // 10)
    noise_rms = np.mean(rms_sorted[:n10]) + 1e-10
    signal_rms = np.mean(rms_sorted[-n10:]) + 1e-10

    return float(20 * np.log10(signal_rms / noise_rms))


def process_file(input_path: Path, output_dir: Path) -> None:
    """Run the full pre-processing pipeline on a single audio file."""
    stem = input_path.stem
    file_output_dir = output_dir / stem
    file_output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Processing: {input_path.name}")

    audio, orig_sr, n_channels = decode_audio(input_path)
    original_duration = len(audio) / orig_sr

    audio = resample(audio, orig_sr, TARGET_SR)
    audio = trim_silence(audio, TARGET_SR)
    audio = normalize_loudness(audio, TARGET_SR)

    output_files = []

    if n_channels > 1:
        mono_mix = audio.mean(axis=1) if audio.ndim == 2 else audio
        full_path = file_output_dir / "full_normalized.wav"
        write_wav(full_path, mono_mix, TARGET_SR)
        output_files.append(full_path.name)

        for i, ch in enumerate(split_channels(audio)):
            ch_path = file_output_dir / f"channel_{i}.wav"
            write_wav(ch_path, ch, TARGET_SR)
            output_files.append(ch_path.name)
    else:
        if audio.ndim == 2:
            audio = audio[:, 0]
        full_path = file_output_dir / "full_normalized.wav"
        write_wav(full_path, audio, TARGET_SR)
        output_files.append(full_path.name)

    processed_duration = len(audio) / TARGET_SR
    snr = estimate_snr(audio if audio.ndim == 1 else audio.mean(axis=1), TARGET_SR)
    log.info(f"  SNR: {snr:.1f} dB")

    metadata = {
        "original_file": input_path.name,
        "original_format": input_path.suffix.lstrip("."),
        "original_sample_rate": orig_sr,
        "original_channels": n_channels,
        "original_duration_seconds": round(original_duration, 2),
        "output_sample_rate": TARGET_SR,
        "output_bit_depth": 16,
        "output_duration_seconds": round(processed_duration, 2),
        "target_lufs": TARGET_LUFS,
        "silence_threshold_db": SILENCE_THRESHOLD_DB,
        "snr_db": round(snr, 1),
        "output_files": output_files,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(file_output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(
        f"Done: {input_path.name} -> {file_output_dir.name}/ "
        f"({original_duration:.1f}s -> {processed_duration:.1f}s, "
        f"{n_channels}ch, {orig_sr}Hz -> {TARGET_SR}Hz)"
    )
