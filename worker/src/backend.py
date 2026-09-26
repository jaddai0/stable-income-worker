"""Project-owned backend interface (HANDOFF section 6.1).

Hides the upstream Stable Audio 3 call signatures from the rest of the worker. Every
backend returns float32 audio shaped (channels, samples) in [-1, 1] at 44.1 kHz, and
raises WorkerFailure with a contract error code instead of leaking upstream exceptions,
whose text can contain the prompt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

import numpy as np

MODEL_ID = "stabilityai/stable-audio-3-medium"
PINNED_MODEL_REVISION = "27b5a21b791b1b033d193a9e1e3ce78493f102f9"
PINNED_SOURCE_COMMIT = "779434a908193105335fd8d833418603625b2859"
SAMPLE_RATE_HZ = 44100
CHANNELS = 2
MAX_PROMPT_TOKENS = 256  # upstream T5Gemma conditioner max_length; excess is silently truncated
MIN_DURATION_S = 5
MAX_DURATION_S = 380

# WorkerError.code values that describe transient infrastructure trouble. Everything else
# is deterministic and must not be retried (HANDOFF section 9, "Claims and retries").
RETRYABLE_CODES = frozenset({"upload_failed", "gateway_unreachable", "timeout", "internal"})
ERROR_CODES = frozenset({
    "validation", "license", "missing_artifact", "runtime_mismatch", "unsupported_engine",
    "deterministic_runtime", "oom", "invalid_output", "upload_failed", "gateway_unreachable",
    "timeout", "internal",
})


class WorkerFailure(Exception):
    """A failure that maps to a contract WorkerError.

    `message` must be a fixed, content-free template. Never interpolate prompts, URLs,
    tokens or upstream exception text into it.
    """

    def __init__(self, code: str, message: str, retryable: Optional[bool] = None):
        if code not in ERROR_CODES:
            raise ValueError(f"unknown worker error code {code!r}")
        super().__init__(message)
        self.code = code
        self.message = message[:500]
        self.retryable = (code in RETRYABLE_CODES) if retryable is None else retryable

    def to_contract(self) -> dict:
        return {"code": self.code, "retryable": self.retryable, "message": self.message}


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    duration_seconds: int
    seed: int
    output_format: str  # "mp3" | "wav" | "flac"


@dataclass(frozen=True)
class BackendIdentity:
    backend: str  # "pytorch" | "tensorrt" | "mock"
    model_id: str
    model_revision: str
    runtime_version: str
    engine_sha256: Optional[str] = None
    gpu_name: Optional[str] = None
    compute_capability: Optional[str] = None
    cuda_version: Optional[str] = None
    torch_version: Optional[str] = None
    tensorrt_version: Optional[str] = None


@dataclass
class AudioResult:
    samples: np.ndarray  # float32, shape (channels, n_samples), values in [-1, 1]
    sample_rate_hz: int
    channels: int
    seed: int
    inference_s: float
    decode_s: Optional[float] = None  # None when decode is fused into inference
    peak_vram_bytes: Optional[int] = None
    extra: dict = field(default_factory=dict)  # content-free diagnostics only

    @property
    def duration_seconds(self) -> float:
        return self.samples.shape[-1] / float(self.sample_rate_hz)


class AudioBackend(Protocol):
    name: str

    def load(self) -> BackendIdentity: ...

    def count_prompt_tokens(self, prompt: str) -> int:
        """Token count exactly as this backend's own tokenizer call will see it."""
        ...

    def generate(self, request: GenerationRequest) -> AudioResult: ...

    def close(self) -> None: ...


def float_audio_from_int16(pcm: np.ndarray) -> np.ndarray:
    """(n, 2) int16 interleaved -> (2, n) float32, using upstream's 32767 scale."""
    if pcm.ndim != 2 or pcm.shape[1] != CHANNELS:
        raise WorkerFailure("invalid_output", "backend returned audio with the wrong shape")
    return (pcm.astype(np.float32) / 32767.0).T.copy()


BackendFactory = Callable[[], AudioBackend]
