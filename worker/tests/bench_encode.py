"""Manual timing of the encode stage (not collected by pytest: file name has no test_ prefix).

Run: uv run python -B tests/bench_encode.py
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backend import SAMPLE_RATE_HZ, GenerationRequest  # noqa: E402
from backends.mock_backend import MockBackend  # noqa: E402
from encode import encode  # noqa: E402

backend = MockBackend()
backend.load()
for seconds in (5, 30, 60, 120, 180, 240, 320, 380):
    samples = backend.generate(GenerationRequest("tone", seconds, 1, "mp3")).samples
    for fmt in ("mp3", "wav"):
        with tempfile.TemporaryDirectory() as tmp:
            start = time.perf_counter()
            enc = encode(samples, SAMPLE_RATE_HZ, seconds, fmt, Path(tmp), 10**9)
            elapsed = time.perf_counter() - start
        print(f"{seconds:>4}s {fmt}: {elapsed:6.3f}s total (encode+verify+hash), {enc.bytes/1e6:6.2f} MB")
