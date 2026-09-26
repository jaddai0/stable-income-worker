from __future__ import annotations

import numpy as np
import pytest
import soundfile

from backend import SAMPLE_RATE_HZ, GenerationRequest, WorkerFailure
from backends.mock_backend import MockBackend
from encode import encode, prepare_samples, quality_flags


def audio(mode="ok", seconds=5, seed=1):
    backend = MockBackend(mode=mode)
    backend.load()
    return backend.generate(GenerationRequest("tone", seconds, seed, "wav")).samples


@pytest.mark.parametrize("fmt,seconds", [("mp3", 5), ("wav", 5), ("flac", 5), ("mp3", 30), ("wav", 30), ("flac", 30)])
def test_encoded_file_is_playable_and_exact(tmp_path, fmt, seconds):
    enc = encode(audio(seconds=seconds), SAMPLE_RATE_HZ, seconds, fmt, tmp_path, 100_000_000)
    info = soundfile.info(str(enc.path))
    assert info.samplerate == 44100 and info.channels == 2
    assert abs(enc.duration_seconds - seconds) <= 0.1
    assert enc.bytes == enc.path.stat().st_size and len(enc.sha256) == 64
    assert enc.quality_flags == []
    if fmt == "wav":
        data, _ = soundfile.read(str(enc.path), dtype="int16")
        assert data.shape == (seconds * 44100, 2)


def test_mp3_is_192k_cbr(tmp_path):
    enc = encode(audio(seconds=30), SAMPLE_RATE_HZ, 30, "mp3", tmp_path, 100_000_000)
    kbps = enc.bytes * 8 / 30 / 1000
    assert 185 <= kbps <= 200, kbps


def test_excess_is_trimmed(tmp_path):
    enc = encode(audio("long"), SAMPLE_RATE_HZ, 5, "wav", tmp_path, 100_000_000)
    data, _ = soundfile.read(str(enc.path), dtype="int16")
    assert data.shape[0] == 5 * 44100


def test_short_output_is_rejected_never_padded():
    with pytest.raises(WorkerFailure) as info:
        prepare_samples(audio("short"), SAMPLE_RATE_HZ, 5)
    assert info.value.code == "invalid_output"


def test_120s_returned_for_180s_request_is_rejected():
    samples = np.zeros((2, 120 * 44100), dtype=np.float32)
    with pytest.raises(WorkerFailure) as info:
        prepare_samples(samples, SAMPLE_RATE_HZ, 180)
    assert info.value.code == "invalid_output"


def test_within_tolerance_short_is_accepted():
    samples = np.zeros((2, 5 * 44100 - 2000), dtype=np.float32)  # 45 ms short
    assert prepare_samples(samples, SAMPLE_RATE_HZ, 5).shape[1] == 5 * 44100 - 2000


@pytest.mark.parametrize("mode", ["nan", "mono"])
def test_invalid_samples_rejected(mode):
    with pytest.raises(WorkerFailure) as info:
        prepare_samples(audio(mode), SAMPLE_RATE_HZ, 5)
    assert info.value.code == "invalid_output"


def test_wrong_sample_rate_rejected():
    with pytest.raises(WorkerFailure):
        prepare_samples(audio(), 48000, 5)


def test_quiet_audio_is_flagged_not_rejected(tmp_path):
    enc = encode(audio("silent"), SAMPLE_RATE_HZ, 5, "wav", tmp_path, 100_000_000)
    assert enc.quality_flags == ["near_silent"]


def test_static_is_flagged():
    assert "static_suspect" in quality_flags(audio("static"))
    assert "static_suspect" not in quality_flags(audio("ok"))


def test_clipping_is_flagged():
    samples = np.clip(audio() * 10, -1, 1)
    assert "clipping" in quality_flags(samples)


def test_size_limit_enforced(tmp_path):
    with pytest.raises(WorkerFailure) as info:
        encode(audio(), SAMPLE_RATE_HZ, 5, "wav", tmp_path, 1000)
    assert info.value.code == "invalid_output"


def test_different_seeds_differ():
    assert not np.array_equal(audio(seed=1), audio(seed=2))


def test_unknown_format_rejected(tmp_path):
    with pytest.raises(WorkerFailure) as info:
        encode(audio(), SAMPLE_RATE_HZ, 5, "ogg", tmp_path, 100_000_000)
    assert info.value.code == "validation"


def test_flac_round_trip_is_bit_exact_against_pcm16(tmp_path):
    from encode import to_pcm16

    samples = audio(seconds=30)
    enc = encode(samples, SAMPLE_RATE_HZ, 30, "flac", tmp_path, 100_000_000)
    assert enc.content_type == "audio/flac" and enc.encoding == "flac_16"
    assert enc.path.suffix == ".flac"
    info = soundfile.info(str(enc.path))
    assert info.format == "FLAC" and info.subtype == "PCM_16"
    data, _ = soundfile.read(str(enc.path), dtype="int16")
    assert np.array_equal(data, to_pcm16(prepare_samples(samples, SAMPLE_RATE_HZ, 30)))
    assert enc.bytes < 30 * 44100 * 4  # lossless but compressed below raw PCM


def test_flac_of_incompressible_audio_stays_within_the_gateway_bound(tmp_path):
    """Worst case for FLAC is noise: size must stay under the gateway's per-format bound
    (gateway/src/artifact-delivery.ts maxOutputBytesFor: ceil(pcm * 1.01) + 65536)."""
    rng = np.random.default_rng(7)
    noise = rng.uniform(-1, 1, size=(2, 10 * 44100)).astype(np.float32)
    enc = encode(noise, SAMPLE_RATE_HZ, 10, "flac", tmp_path, 100_000_000)
    pcm = 10 * 44100 * 4
    assert enc.bytes <= int(np.ceil(pcm * 1.01)) + 65536
    data, _ = soundfile.read(str(enc.path), dtype="int16")
    assert data.shape == (10 * 44100, 2)


def test_flac_rejected_when_over_the_upload_limit(tmp_path):
    with pytest.raises(WorkerFailure) as exc:
        encode(audio(seconds=5), SAMPLE_RATE_HZ, 5, "flac", tmp_path, 1000)
    assert exc.value.code == "invalid_output"
