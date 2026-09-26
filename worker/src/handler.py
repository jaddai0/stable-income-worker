"""RunPod Serverless entry point (HANDOFF sections 3, 6, 9, 10, 13).

Lifecycle
- Process start: build the backend once, with a bounded initialisation timeout. If init
  fails or times out the process exits non-zero instead of accepting jobs it cannot serve;
  the control-plane watchdog detects repeated failed starts.
- Per job: validate the content-free input -> claim (gets prompt, seed and a presigned
  upload) -> check the token limit -> generate -> validate and encode -> upload -> complete.
- The handler never raises into the RunPod SDK (which would store a traceback in the job
  output) and returns only the metadata manifest from contracts/worker-result.schema.json.
"""
from __future__ import annotations

import os

# Must precede any import that can pull in huggingface_hub/transformers: the hub reads
# these into module constants at import time and ignores later changes (measured on the
# 4090: set late, the T5Gemma tokenizer tried to fetch from the gated repo and got 401).
for _name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY",
              "HF_HUB_DISABLE_PROGRESS_BARS"):
    os.environ[_name] = "1"

import json
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from backend import (MAX_DURATION_S, MIN_DURATION_S, PINNED_MODEL_REVISION, AudioBackend,
                     BackendIdentity, GenerationRequest, WorkerFailure)
from encode import encode
from execution_claim import ClaimGranted, ClaimRejected, GatewayClient
from prompt_tokens import check_prompt
from telemetry import Stopwatch, empty_measurements, env_float, log_event, process_rss_bytes
from upload import upload_file

WORKER_VERSION = "0.1.0"
MAX_MANIFEST_BYTES = 32 * 1024
DEFAULT_INIT_TIMEOUT_S = 180.0  # policy limits.application_initialization_timeout_seconds
CONTRACTS_DIR = Path(os.environ.get("SA3_CONTRACTS_DIR", Path(__file__).resolve().parents[2] / "contracts"))


def runtime_version() -> str:
    digest = os.environ.get("SA3_IMAGE_DIGEST", "unknown")
    return f"stable-income-worker/{WORKER_VERSION}+{digest}"


@dataclass
class WorkerState:
    backend: AudioBackend
    identity: BackendIdentity
    process_init_s: float
    model_load_s: float
    worker_process_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    jobs_served: int = 0
    cold: bool = True
    lock: threading.Lock = field(default_factory=threading.Lock)

    def identity_dict(self) -> dict:
        ident = {
            "backend": self.identity.backend,
            "model_id": self.identity.model_id,
            "model_revision": self.identity.model_revision,
            "runtime_version": self.identity.runtime_version,
            "engine_sha256": self.identity.engine_sha256,
            "gpu_name": self.identity.gpu_name,
            "compute_capability": self.identity.compute_capability,
            "cuda_version": self.identity.cuda_version,
            "torch_version": self.identity.torch_version,
            "tensorrt_version": self.identity.tensorrt_version,
            "worker_process_id": self.worker_process_id,
            "jobs_served_by_process": self.jobs_served,
        }
        return ident


def check_deployed_revision() -> None:
    """infra sets SA3_MODEL_REVISION from the deployed policy; refuse to serve if the image
    pins a different model revision than the one the endpoint was configured for."""
    configured = os.environ.get("SA3_MODEL_REVISION")
    if configured and configured != PINNED_MODEL_REVISION:
        raise WorkerFailure("runtime_mismatch", "endpoint model revision differs from the image pin")


def make_backend(name: str) -> AudioBackend:
    if name == "pytorch":
        from backends.pytorch_backend import PyTorchBackend
        return PyTorchBackend(runtime_version=runtime_version())
    if name == "tensorrt":
        from backends.tensorrt_backend import TensorRTBackend
        return TensorRTBackend(runtime_version=runtime_version())
    if name == "mock":
        from backends.mock_backend import MockBackend
        return MockBackend()
    raise WorkerFailure("runtime_mismatch", "unknown backend selected")


def init_worker(backend: Optional[AudioBackend] = None, timeout_s: Optional[float] = None,
                process_started: Optional[float] = None) -> WorkerState:
    """Load the backend once. Raises WorkerFailure on failure or timeout."""
    started = process_started if process_started is not None else time.perf_counter()
    timeout_s = timeout_s if timeout_s is not None else env_float("SA3_INIT_TIMEOUT_S", DEFAULT_INIT_TIMEOUT_S)
    check_deployed_revision()
    backend = backend or make_backend(os.environ.get("SA3_BACKEND", "pytorch"))
    outcome: dict = {}

    def target() -> None:
        try:
            load_start = time.perf_counter()
            outcome["identity"] = backend.load()
            outcome["load_s"] = time.perf_counter() - load_start
        except WorkerFailure as exc:
            outcome["error"] = exc
        except Exception as exc:  # never surface upstream text
            outcome["error"] = WorkerFailure("internal", f"backend load raised {type(exc).__name__}")

    thread = threading.Thread(target=target, name="backend-load", daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise WorkerFailure("timeout", "backend initialisation exceeded its time limit")
    if "error" in outcome:
        raise outcome["error"]
    return WorkerState(backend=backend, identity=outcome["identity"],
                       process_init_s=round(time.perf_counter() - started, 6),
                       model_load_s=round(outcome["load_s"], 6))


# --- input validation ------------------------------------------------------------------

def _load_schema(name: str) -> Optional[dict]:
    path = CONTRACTS_DIR / name
    try:
        return json.loads(path.read_text())
    except OSError:
        return None


_INPUT_SCHEMA = None


def validate_input(job_input) -> dict:
    """Validate against contracts/worker-input.schema.json (hand checks if schema absent)."""
    global _INPUT_SCHEMA
    if not isinstance(job_input, dict):
        raise WorkerFailure("validation", "job input is not an object", retryable=False)
    if _INPUT_SCHEMA is None:
        _INPUT_SCHEMA = _load_schema("worker-input.schema.json") or {}
    if _INPUT_SCHEMA:
        import jsonschema

        try:
            jsonschema.validate(job_input, _INPUT_SCHEMA)
        except jsonschema.ValidationError:
            raise WorkerFailure("validation", "job input does not match the worker input contract",
                                retryable=False) from None
    else:
        allowed = {"schema_version", "job_id", "attempt", "capability", "synthetic"}
        if (set(job_input) - allowed or job_input.get("schema_version") != 1
                or not isinstance(job_input.get("job_id"), str)
                or not isinstance(job_input.get("attempt"), int)
                or not isinstance(job_input.get("capability"), str)):
            raise WorkerFailure("validation", "job input does not match the worker input contract",
                                retryable=False)
    return job_input


def _generation_request(generation: dict) -> GenerationRequest:
    try:
        request = GenerationRequest(prompt=str(generation["prompt"]),
                                    duration_seconds=int(generation["duration_seconds"]),
                                    seed=int(generation["seed"]),
                                    output_format=str(generation["output_format"]))
    except (KeyError, TypeError, ValueError):
        raise WorkerFailure("validation", "claim returned malformed generation parameters", retryable=False) from None
    if not MIN_DURATION_S <= request.duration_seconds <= MAX_DURATION_S:
        raise WorkerFailure("validation", "duration is outside the supported range", retryable=False)
    if not 0 <= request.seed <= 4294967295:
        raise WorkerFailure("validation", "seed is outside the supported range", retryable=False)
    if request.output_format not in ("mp3", "wav", "flac"):
        raise WorkerFailure("validation", "unsupported output format", retryable=False)
    return request


# --- manifest ---------------------------------------------------------------------------

def _manifest(job_id: str, attempt: int, status: str, *, claim: Optional[ClaimGranted] = None,
              output: Optional[dict] = None, error: Optional[dict] = None,
              worker: Optional[dict] = None, measurements: Optional[dict] = None) -> dict:
    manifest = {
        "schema_version": 1,
        "job_id": job_id,
        "attempt": attempt,
        "attempt_id": claim.attempt_id if claim else None,
        "fencing_token": claim.fencing_token if claim else None,
        "status": status,
    }
    if output is not None:
        manifest["output"] = output
    if error is not None:
        manifest["error"] = error
    if worker is not None:
        manifest["worker"] = worker
    if measurements is not None:
        manifest["measurements"] = measurements
    if len(json.dumps(manifest).encode()) > MAX_MANIFEST_BYTES:  # defensive; normal size is ~1-2 KiB
        manifest.pop("worker", None)
        if "error" in manifest:
            manifest["error"]["message"] = manifest["error"].get("message", "")[:100]
    return manifest


# --- job processing ---------------------------------------------------------------------

ClientFactory = Callable[[str], GatewayClient]


def process_job(job: dict, state: WorkerState, client_factory: Optional[ClientFactory] = None,
                upload_fn: Callable = upload_file, workdir: Optional[str] = None) -> dict:
    """Run one RunPod job. Never raises; always returns a contract manifest."""
    with state.lock:
        return _process_job(job, state, client_factory or GatewayClient.from_env, upload_fn, workdir)


def _process_job(job, state, client_factory, upload_fn, workdir) -> dict:
    handler_start = time.perf_counter()
    watch = Stopwatch()
    cold = state.cold
    state.cold = False
    state.jobs_served += 1
    measurements = empty_measurements(cold)
    if cold:
        measurements["process_init_s"] = state.process_init_s
        measurements["model_load_s"] = state.model_load_s

    def finish(status: str, **kwargs) -> dict:
        for name, seconds in watch.stages.items():
            measurements[name] = seconds
        measurements["handler_total_s"] = round(time.perf_counter() - handler_start, 6)
        measurements["process_rss_bytes"] = process_rss_bytes()
        manifest = _manifest(job_id, attempt, status, worker=state.identity_dict(),
                             measurements=measurements, **kwargs)
        log_event("job_finished", job_id=job_id, attempt=attempt, status=status,
                  code=(kwargs.get("error") or {}).get("code"),
                  worker_process_id=state.worker_process_id,
                  jobs_served_by_process=state.jobs_served)
        return manifest

    raw_input = (job or {}).get("input") if isinstance(job, dict) else None
    job_id = raw_input.get("job_id") if isinstance(raw_input, dict) and isinstance(raw_input.get("job_id"), str) else "unknown"
    attempt = raw_input.get("attempt") if isinstance(raw_input, dict) and isinstance(raw_input.get("attempt"), int) else 0
    try:
        job_input = validate_input(raw_input)
    except WorkerFailure as failure:
        return finish("failed", error=failure.to_contract())

    runpod_job_id = str(job.get("id") or "unknown")[:128]
    try:
        client = client_factory(job_input["capability"])
        with watch.stage("claim_s"):
            claim = client.claim(job_id, attempt, runpod_job_id, state.identity_dict())
    except WorkerFailure as failure:
        return finish("failed", error=failure.to_contract())
    except Exception as exc:
        return finish("failed", error=WorkerFailure("internal", f"claim raised {type(exc).__name__}").to_contract())

    if isinstance(claim, ClaimRejected):
        log_event("claim_rejected", job_id=job_id, attempt=attempt, http_status=claim.http_status)
        return finish("claim_rejected", error=claim.error)

    log_event("claim_granted", job_id=job_id, attempt=attempt, attempt_id=claim.attempt_id,
              fencing_token=claim.fencing_token)

    def fail_after_claim(failure: WorkerFailure) -> dict:
        body = {"attempt_id": claim.attempt_id, "fencing_token": claim.fencing_token,
                "outcome": "failed", "error": failure.to_contract(),
                "worker": state.identity_dict(), "measurements": _snapshot(measurements, watch)}
        try:
            client.complete(job_id, body)
        except WorkerFailure:
            log_event("completion_unconfirmed", job_id=job_id, attempt=attempt, status="failed")
        return finish("failed", claim=claim, error=failure.to_contract())

    try:
        request = _generation_request(claim.generation)
        token_count = check_prompt(request.prompt, state.backend.count_prompt_tokens)
        log_event("prompt_checked", job_id=job_id, token_count=token_count,
                  duration_seconds=request.duration_seconds, output_format=request.output_format)
        with watch.stage("inference_s"):
            result = state.backend.generate(request)
        measurements["peak_vram_bytes"] = result.peak_vram_bytes
        if result.decode_s is not None:
            measurements["decode_s"] = round(result.decode_s, 6)
        if result.extra.get("graph_build_ms") is not None:
            log_event("graph_build", job_id=job_id, graph_build_ms=result.extra["graph_build_ms"])
        with tempfile.TemporaryDirectory(prefix="sa3-", dir=workdir) as tmp:
            with watch.stage("encode_s"):
                encoded = encode(result.samples, result.sample_rate_hz, request.duration_seconds,
                                 request.output_format, Path(tmp), int(claim.upload["max_bytes"]))
            del result
            with watch.stage("upload_s"):
                upload_fn(encoded.path, claim.upload, encoded.content_type)
    except WorkerFailure as failure:
        return fail_after_claim(failure)
    except MemoryError:
        return fail_after_claim(WorkerFailure("oom", "worker ran out of host memory", retryable=False))
    except Exception as exc:
        return fail_after_claim(WorkerFailure("internal", f"job raised {type(exc).__name__}"))

    output = {
        "object_key": str(claim.upload["object_key"]),
        "content_type": encoded.content_type,
        "encoding": encoded.encoding,
        "bytes": encoded.bytes,
        "sha256": encoded.sha256,
        "duration_seconds": encoded.duration_seconds,
        "sample_rate_hz": encoded.sample_rate_hz,
        "channels": encoded.channels,
        "quality_flags": encoded.quality_flags,
    }
    body = {"attempt_id": claim.attempt_id, "fencing_token": claim.fencing_token,
            "outcome": "succeeded", "output": output, "worker": state.identity_dict(),
            "measurements": _snapshot(measurements, watch)}
    try:
        answer = client.complete(job_id, body)
    except WorkerFailure:
        # Audio is uploaded; the gateway reconciles from this manifest (never bills twice).
        return finish("completion_unconfirmed", claim=claim, output=output)
    log_event("completion_answered", job_id=job_id, attempt=attempt, accepted=answer["accepted"])
    return finish("succeeded", claim=claim, output=output)


def _snapshot(measurements: dict, watch: Stopwatch) -> dict:
    snap = dict(measurements)
    for name, seconds in watch.stages.items():
        snap[name] = seconds
    return snap


# --- process entry ----------------------------------------------------------------------

STATE: Optional[WorkerState] = None


def handler(job: dict) -> dict:
    return process_job(job, STATE)


def main() -> None:
    global STATE
    started = time.perf_counter()
    try:
        STATE = init_worker(process_started=started)
    except WorkerFailure as failure:
        log_event("init_failed", code=failure.code, retryable=failure.retryable)
        sys.stdout.flush()
        os._exit(3)  # a stuck load thread must not keep the worker alive
    log_event("init_done", backend=STATE.identity.backend, seconds=STATE.process_init_s,
              worker_process_id=STATE.worker_process_id)
    import runpod

    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
