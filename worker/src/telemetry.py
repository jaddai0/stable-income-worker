"""Stage timing, memory readings and content-free structured logs.

Logs are JSON lines on stdout with an allow-listed key set, so a prompt, URL, token or
exception string cannot reach RunPod's log store by accident (HANDOFF section 13).
"""
from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from typing import Iterator, Optional

MEASUREMENT_KEYS = (
    "process_init_s", "model_load_s", "claim_s", "inference_s", "decode_s", "encode_s",
    "upload_s", "handler_total_s", "peak_vram_bytes", "process_rss_bytes",
)

_LOG_KEYS = frozenset({
    "event", "job_id", "attempt", "attempt_id", "fencing_token", "status", "code", "retryable",
    "backend", "duration_seconds", "output_format", "bytes", "stage", "seconds", "http_status",
    "try", "worker_process_id", "jobs_served_by_process", "quality_flags", "graph_build_ms",
    "token_count", "accepted",
})


def log_event(event: str, **fields) -> None:
    record = {"event": event}
    for key, value in fields.items():
        if key in _LOG_KEYS:
            record[key] = value
    sys.stdout.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    sys.stdout.flush()


class Stopwatch:
    def __init__(self) -> None:
        self.stages: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = round(time.perf_counter() - start, 6)


def process_rss_bytes() -> Optional[int]:
    """Current resident set size (Linux /proc), falling back to peak RSS elsewhere."""
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except Exception:
        return None


def reset_peak_vram() -> None:
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_vram_bytes() -> Optional[int]:
    """Peak bytes allocated by PyTorch's caching allocator since the last reset.

    Only PyTorch allocations are visible here; TensorRT engine memory allocated outside
    the torch allocator is not counted, so the TensorRT backend reports NVML-measured
    process memory instead (see tensorrt_backend.py).
    """
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    return None


def empty_measurements(cold_process: bool) -> dict:
    measurements = {key: None for key in MEASUREMENT_KEYS}
    measurements["cold_process"] = cold_process
    return measurements


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else default
