"""Output validation and encoding (HANDOFF section 6.5).

Formats (per request; owner decision 2026-09-26): MP3 192 kbit/s CBR (default), WAV PCM16,
or FLAC 16-bit lossless. All keep the native 44.1 kHz stereo and apply no loudness change.
FLAC is written by libsndfile (bundled in the pinned `soundfile` wheel; libsndfile LGPL-2.1,
libFLAC BSD-3-Clause) and decodes bit-exactly back to the PCM16 that WAV would carry. MP3 uses LAME through the `lameenc` binding (LAME 3.100, LGPL): it
ships self-contained manylinux and macOS wheels, needs no system ffmpeg, and runs in
process. WAV uses the standard library. Every file is decoded back with libsndfile
(`soundfile`) before upload to prove it is playable and the right length.

Quality flags are warnings, not rejections: quiet ambience is a legitimate output.
"""
from __future__ import annotations

import hashlib
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from backend import CHANNELS, SAMPLE_RATE_HZ, WorkerFailure

DURATION_TOLERANCE_S = 0.1
MP3_BITRATE_KBPS = 192
# LAME -q 3 (LAME's own default). Bitrate and size are unchanged; -q 2 measured 2.7x slower
# (11.6 vs 4.2 ms per audio second on Apple silicon, 2026-09-26), and encode time is billed
# GPU-worker time. Audible difference is unmeasured; include it in qualification listening.
MP3_QUALITY = 3
# libsndfile maps compression_level 0..1 onto libFLAC levels 0..8; 0.625 targets level 5,
# libFLAC's own default (mapping per libsndfile source; not independently measured).
FLAC_COMPRESSION_LEVEL = 0.625
ENCODER_VERSION = ("lameenc==1.8.4 (LAME 3.100, LGPL-3.0-or-later); wav=python stdlib; "
                   "flac=soundfile==0.14.0 libsndfile 1.2.2 (LGPL-2.1) + libFLAC (BSD-3-Clause)")

NEAR_SILENT_RMS_DBFS = -60.0
CLIPPING_THRESHOLD = 0.999
CLIPPING_FRACTION = 0.001
STATIC_FLATNESS = 0.45  # a white-noise periodogram measures ~0.56; music is far lower

FORMATS = {
    "mp3": {"content_type": "audio/mpeg", "encoding": "mp3_cbr_192k", "suffix": ".mp3"},
    "wav": {"content_type": "audio/wav", "encoding": "wav_pcm16", "suffix": ".wav"},
    "flac": {"content_type": "audio/flac", "encoding": "flac_16", "suffix": ".flac"},
}


@dataclass
class EncodedAudio:
    path: Path
    content_type: str
    encoding: str
    bytes: int
    sha256: str
    duration_seconds: float
    sample_rate_hz: int = SAMPLE_RATE_HZ
    channels: int = CHANNELS
    quality_flags: list = field(default_factory=list)


def prepare_samples(samples: np.ndarray, sample_rate_hz: int, requested_seconds: int) -> np.ndarray:
    """Validate raw backend output and trim excess. Never pads a short generation."""
    if not isinstance(samples, np.ndarray) or samples.ndim != 2:
        raise WorkerFailure("invalid_output", "generated audio has the wrong shape", retryable=False)
    if samples.shape[0] != CHANNELS:
        raise WorkerFailure("invalid_output", "generated audio is not stereo", retryable=False)
    if sample_rate_hz != SAMPLE_RATE_HZ:
        raise WorkerFailure("invalid_output", "generated audio has the wrong sample rate", retryable=False)
    if not np.isfinite(samples).all():
        raise WorkerFailure("invalid_output", "generated audio contains non-finite samples", retryable=False)
    wanted = int(round(requested_seconds * SAMPLE_RATE_HZ))
    have = samples.shape[1]
    if have < wanted - int(DURATION_TOLERANCE_S * SAMPLE_RATE_HZ):
        raise WorkerFailure("invalid_output", "generated audio is shorter than requested", retryable=False)
    return np.ascontiguousarray(samples[:, :wanted], dtype=np.float32)


def quality_flags(samples: np.ndarray) -> list[str]:
    flags: list[str] = []
    if samples.size == 0:
        return ["near_silent"]
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    if rms <= 0.0 or 20.0 * np.log10(rms) < NEAR_SILENT_RMS_DBFS:
        flags.append("near_silent")
    if float(np.mean(np.abs(samples) >= CLIPPING_THRESHOLD)) > CLIPPING_FRACTION:
        flags.append("clipping")
    if "near_silent" not in flags and spectral_flatness(samples) > STATIC_FLATNESS:
        flags.append("static_suspect")
    return flags


def spectral_flatness(samples: np.ndarray, frame: int = 4096, max_frames: int = 64) -> float:
    """Mean spectral flatness (geometric / arithmetic mean power) of the mono mix."""
    mono = samples.mean(axis=0, dtype=np.float64)
    if mono.size < frame:
        return 0.0
    starts = np.linspace(0, mono.size - frame, num=min(max_frames, mono.size // frame)).astype(int)
    window = np.hanning(frame)
    values = []
    for start in starts:
        power = np.abs(np.fft.rfft(mono[start:start + frame] * window)) ** 2 + 1e-20
        values.append(np.exp(np.mean(np.log(power))) / np.mean(power))
    return float(np.mean(values))


def to_pcm16(samples: np.ndarray) -> np.ndarray:
    """(2, n) float32 -> (n, 2) int16 interleaved, upstream 32767 scale."""
    return np.round(np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).T.copy()


def _write_wav(pcm: np.ndarray, path: Path) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE_HZ)
        wav.writeframes(pcm.astype("<i2").tobytes())


def _write_mp3(pcm: np.ndarray, path: Path) -> None:
    import lameenc

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(MP3_BITRATE_KBPS)
    encoder.set_in_sample_rate(SAMPLE_RATE_HZ)
    encoder.set_channels(CHANNELS)
    encoder.set_quality(MP3_QUALITY)
    with path.open("wb") as fh:
        fh.write(encoder.encode(pcm.astype("<i2").tobytes()))
        fh.write(encoder.flush())


def _write_flac(pcm: np.ndarray, path: Path) -> None:
    import soundfile

    soundfile.write(str(path), pcm, SAMPLE_RATE_HZ, format="FLAC", subtype="PCM_16",
                    compression_level=FLAC_COMPRESSION_LEVEL)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_decodable(path: Path, requested_seconds: int) -> float:
    """Decode the encoded file and return its duration; raise if unplayable or wrong."""
    import soundfile

    try:
        info = soundfile.info(str(path))
        data, rate = soundfile.read(str(path), dtype="int16", always_2d=True)
    except Exception:
        raise WorkerFailure("invalid_output", "encoded audio could not be decoded", retryable=False) from None
    if rate != SAMPLE_RATE_HZ or data.shape[1] != CHANNELS or info.channels != CHANNELS:
        raise WorkerFailure("invalid_output", "encoded audio has the wrong format", retryable=False)
    duration = data.shape[0] / float(rate)
    if abs(duration - requested_seconds) > DURATION_TOLERANCE_S:
        raise WorkerFailure("invalid_output", "encoded audio has the wrong duration", retryable=False)
    return duration


def encode(samples: np.ndarray, sample_rate_hz: int, requested_seconds: int, output_format: str,
           directory: Path, max_bytes: int) -> EncodedAudio:
    if output_format not in FORMATS:
        raise WorkerFailure("validation", "unsupported output format", retryable=False)
    spec = FORMATS[output_format]
    trimmed = prepare_samples(samples, sample_rate_hz, requested_seconds)
    flags = quality_flags(trimmed)
    pcm = to_pcm16(trimmed)
    path = Path(directory) / ("audio" + spec["suffix"])
    if output_format == "mp3":
        _write_mp3(pcm, path)
    elif output_format == "flac":
        _write_flac(pcm, path)
    else:
        _write_wav(pcm, path)
    size = path.stat().st_size
    if size <= 0:
        raise WorkerFailure("invalid_output", "encoded audio is empty", retryable=False)
    if size > max_bytes:
        raise WorkerFailure("invalid_output", "encoded audio exceeds the upload size limit", retryable=False)
    duration = verify_decodable(path, requested_seconds)
    return EncodedAudio(path=path, content_type=spec["content_type"], encoding=spec["encoding"],
                        bytes=size, sha256=_sha256(path), duration_seconds=round(duration, 6),
                        quality_flags=flags)
