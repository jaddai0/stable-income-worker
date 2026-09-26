"""Deterministic CPU mock backend for local tests and the no-GPU service canary.

Produces seed- and prompt-dependent tones so duplicate/varied-output tests are meaningful.
Never used for real traffic: the handler only selects it when SA3_BACKEND=mock.
"""
from __future__ import annotations

import hashlib
import time
from typing import Optional

import numpy as np

from backend import (CHANNELS, MODEL_ID, SAMPLE_RATE_HZ, AudioResult, BackendIdentity,
                     GenerationRequest, WorkerFailure)


class MockBackend:
    name = "mock"
    load_count = 0  # class-wide: proves a process loads the model once

    def __init__(self, *, mode: str = "ok", fail_code: Optional[str] = None,
                 token_counter=None, latency_s: float = 0.0):
        self.mode = mode
        self.fail_code = fail_code
        self.latency_s = latency_s
        self._count = token_counter or (lambda prompt: len(prompt.split()))
        self.loaded = False
        self.generate_calls = 0

    def load(self) -> BackendIdentity:
        MockBackend.load_count += 1
        self.loaded = True
        return BackendIdentity(backend="mock", model_id=MODEL_ID, model_revision="mock",
                               runtime_version="mock")

    def count_prompt_tokens(self, prompt: str) -> int:
        return self._count(prompt)

    def generate(self, request: GenerationRequest) -> AudioResult:
        if not self.loaded:
            raise WorkerFailure("internal", "backend used before load")
        self.generate_calls += 1
        if self.fail_code:
            raise WorkerFailure(self.fail_code, "mock backend failure")
        start = time.perf_counter()
        if self.latency_s:
            time.sleep(self.latency_s)
        n = int(round(request.duration_seconds * SAMPLE_RATE_HZ))
        if self.mode == "short":
            n -= SAMPLE_RATE_HZ  # one second short: must be rejected, never padded
        elif self.mode == "long":
            n += SAMPLE_RATE_HZ // 2  # excess must be trimmed
        key = hashlib.sha256(f"{request.seed}|{request.prompt}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(key[:8], "little"))
        t = np.arange(n, dtype=np.float64) / SAMPLE_RATE_HZ
        if self.mode == "silent":
            audio = np.zeros((CHANNELS, n), dtype=np.float32)
        elif self.mode == "static":
            audio = rng.uniform(-0.5, 0.5, size=(CHANNELS, n)).astype(np.float32)
        else:
            freqs = rng.uniform(110.0, 880.0, size=3)
            mono = sum(0.2 * np.sin(2 * np.pi * f * t) for f in freqs)
            audio = np.stack([mono, np.roll(mono, 17)]).astype(np.float32)
        if self.mode == "nan":
            audio[0, n // 2] = np.nan
        if self.mode == "mono":
            audio = audio[:1]
        return AudioResult(samples=audio, sample_rate_hz=SAMPLE_RATE_HZ, channels=audio.shape[0],
                           seed=request.seed, inference_s=time.perf_counter() - start)

    def close(self) -> None:
        self.loaded = False
