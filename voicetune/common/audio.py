"""Audio I/O, transforms, and analysis primitives shared across the pipeline."""

import base64
import io
import logging
from pathlib import Path

import av
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import soxr

log = logging.getLogger(__name__)

DEFAULT_TARGET_SR = 16000
DEFAULT_TARGET_LUFS = -23.0
DEFAULT_SILENCE_THRESHOLD_DB = -40.0

SUPPORTED_EXTENSIONS = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".aac", ".wma", ".opus", ".3gp"}

LOW_ENERGY_CODE = "low_energy"
MOSTLY_SILENCE_CODE = "mostly_silence"
TAIL_DECAY_CODE = "tail_decay"

SILENCE_RMS_DB = -40.0
LOW_ENERGY_RMS_DB = -30.0
MAX_TAIL_DECAY_DB = 5.0
DECAY_FRAME_MS = 30
SPEECH_FLOOR_DB = -30.0


# ------------------------------------------------------------------ decoding / I/O


def decode_audio(path: Path) -> tuple[np.ndarray, int, int]:
    """Decode audio via PyAV; returns (audio[(n_samples, n_channels)], sample_rate, n_channels)."""
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


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Downmix (samples, channels) audio to 1D mono; pass through if already 1D."""
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=1)


def read_mono_wav(path: Path, dtype: str = "float32") -> tuple[np.ndarray, int]:
    """Read a WAV as mono; downmix in-place if multi-channel."""
    audio, sr = sf.read(str(path), dtype=dtype)
    return to_mono(audio), sr


def read_audio_channels_first(path: Path, dtype: str = "float32") -> tuple[np.ndarray, int]:
    """Read audio into a `(channels, samples)` array (pyannote-compatible layout)."""
    audio, sr = sf.read(str(path), dtype=dtype)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    else:
        audio = audio.T
    return audio, sr


def audio_duration(path: Path) -> float:
    """Return the duration of an audio file in seconds (no full read)."""
    info = sf.info(str(path))
    return info.frames / info.samplerate


def audio_to_data_uri(audio_path: Path) -> str:
    """Encode a WAV file's raw bytes as a base64 `data:audio/wav;base64,...` URI."""
    raw = audio_path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


def write_wav(path: Path, audio: np.ndarray, sr: int = DEFAULT_TARGET_SR) -> None:
    """Write audio as 16-bit PCM WAV, clipping to [-1, 1]."""
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(str(path), audio, sr, subtype="PCM_16")


# ------------------------------------------------------------------ transforms


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
    threshold_db: float = DEFAULT_SILENCE_THRESHOLD_DB,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Trim leading and trailing silence based on frame-level RMS energy."""
    mono = to_mono(audio)

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
    audio: np.ndarray, sr: int, target_lufs: float = DEFAULT_TARGET_LUFS
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


def cut_turn_audio(audio: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    """Extract a segment of audio between start and end times."""
    start_sample = max(0, int(start * sr))
    end_sample = min(len(audio), int(end * sr))
    return audio[start_sample:end_sample]


def clip_turn_audio(wav_path: Path, start: float, end: float) -> str:
    """Extract [start, end] seconds from a WAV file; return base64-encoded 16-bit WAV bytes."""
    info = sf.info(str(wav_path))
    sr = info.samplerate
    audio, _ = sf.read(str(wav_path), start=int(start * sr), stop=int(end * sr))

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return base64.standard_b64encode(buf.getvalue()).decode()


# ------------------------------------------------------------------ analysis


def frame_rms_db(audio: np.ndarray, sr: int) -> np.ndarray:
    """Per-frame RMS in dB using non-overlapping `DECAY_FRAME_MS` frames."""
    frame = max(1, int(sr * DECAY_FRAME_MS / 1000))
    n = (len(audio) // frame) * frame
    if n < frame:
        return np.array([])
    frames = audio[:n].reshape(-1, frame)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def audio_issue(audio: np.ndarray, sr: int) -> str | None:
    """Return a drop-reason code if the clip is unusable for training, else None."""
    if len(audio) == 0:
        return MOSTLY_SILENCE_CODE
    rms_db = 20 * np.log10(float(np.sqrt(np.mean(audio ** 2) + 1e-10)) + 1e-12)
    if rms_db < SILENCE_RMS_DB:
        return MOSTLY_SILENCE_CODE
    if rms_db < LOW_ENERGY_RMS_DB:
        return LOW_ENERGY_CODE

    frames_db = frame_rms_db(audio, sr)
    if len(frames_db) < 10:
        return MOSTLY_SILENCE_CODE
    speech = frames_db > SPEECH_FLOOR_DB
    if speech.sum() < 10:
        return MOSTLY_SILENCE_CODE
    t1 = len(frames_db) // 3
    t2 = 2 * len(frames_db) // 3
    first = frames_db[:t1][speech[:t1]]
    last = frames_db[t2:][speech[t2:]]
    if first.size == 0 or last.size == 0:
        return MOSTLY_SILENCE_CODE
    if float(first.mean() - last.mean()) > MAX_TAIL_DECAY_DB:
        return TAIL_DECAY_CODE
    return None


def estimate_snr(audio: np.ndarray, sr: int, frame_length: int = 2048, hop_length: int = 512) -> float:
    """Frame-RMS SNR in dB: top 10% of frames as signal, bottom 10% as noise."""
    mono = to_mono(audio)
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
