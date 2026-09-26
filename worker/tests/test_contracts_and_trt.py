from __future__ import annotations

import hashlib
import json

import pytest

import handler
from backend import WorkerFailure
from backends.tensorrt_backend import REQUIRED_TENSORRT_VERSION, verify_engine_manifest
from conftest import job_input

REQUIRED = ["t5gemma/t5gemma_fp16.trt", "sa3-m/dit_fp16.trt", "same-l/dec_fp16_chunkable_limiter.trt"]


def test_example_worker_input_matches_contract(validate):
    validate(job_input(), "worker-input.schema.json")
    validate(job_input(attempt=2, synthetic=True), "worker-input.schema.json")


@pytest.mark.parametrize("bad", [
    job_input(prompt="hello"),
    job_input(attempt=3),
    job_input(job_id="not-a-job"),
    {k: v for k, v in job_input().items() if k != "capability"},
    job_input(callback_url="https://evil.invalid"),
])
def test_handler_input_validation_matches_contract(bad, validate):
    with pytest.raises(WorkerFailure):
        handler.validate_input(bad)
    with pytest.raises(AssertionError):
        validate(bad, "worker-input.schema.json")


def make_engines(root, arch="sm_89", files=REQUIRED, tamper=None, trt=REQUIRED_TENSORRT_VERSION):
    manifest = {"arch": arch, "tensorrt_version": trt, "precision": "fp16",
                "source_model_revision": "27b5a21b791b1b033d193a9e1e3ce78493f102f9", "files": {}}
    for rel in files:
        path = root / arch / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        data = rel.encode() * 10
        path.write_bytes(data)
        manifest["files"][rel] = hashlib.sha256(data).hexdigest()
    if tamper:
        (root / arch / tamper).write_bytes(b"tampered")
    (root / "engines.lock.json").write_text(json.dumps(manifest))
    return manifest


def test_engine_manifest_ok(tmp_path):
    make_engines(tmp_path)
    manifest = verify_engine_manifest(tmp_path, "sm_89", REQUIRED, REQUIRED_TENSORRT_VERSION)
    assert len(manifest["_manifest_sha256"]) == 64


@pytest.mark.parametrize("kwargs,runtime_trt,code", [
    ({"tamper": "sa3-m/dit_fp16.trt"}, REQUIRED_TENSORRT_VERSION, "missing_artifact"),
    ({"arch": "sm_90"}, REQUIRED_TENSORRT_VERSION, "unsupported_engine"),
    ({}, "10.16.0.0", "unsupported_engine"),
    ({"files": REQUIRED[:2]}, REQUIRED_TENSORRT_VERSION, "missing_artifact"),
])
def test_engine_manifest_rejects(tmp_path, kwargs, runtime_trt, code):
    make_engines(tmp_path, **kwargs)
    with pytest.raises(WorkerFailure) as info:
        verify_engine_manifest(tmp_path, "sm_89", REQUIRED, runtime_trt)
    assert info.value.code == code


def test_missing_manifest(tmp_path):
    with pytest.raises(WorkerFailure) as info:
        verify_engine_manifest(tmp_path, "sm_89", REQUIRED, REQUIRED_TENSORRT_VERSION)
    assert info.value.code == "missing_artifact"


def test_unknown_backend_is_rejected():
    with pytest.raises(WorkerFailure) as info:
        handler.make_backend("small")
    assert info.value.code == "runtime_mismatch"


def test_worker_error_codes_match_contract():
    from backend import ERROR_CODES
    from conftest import CONTRACTS

    schema = json.loads((CONTRACTS / "internal-api.schema.json").read_text())
    assert set(schema["$defs"]["WorkerError"]["properties"]["code"]["enum"]) == set(ERROR_CODES)
    measurement_keys = set(schema["$defs"]["Measurements"]["properties"])
    from telemetry import MEASUREMENT_KEYS
    assert set(MEASUREMENT_KEYS) | {"cold_process"} == measurement_keys


def test_tensorrt_backend_uses_eager_path_not_per_length_graphs():
    """Graph mode rebuilds a CUDA graph for every new length (measured 4.5 s at 380 s on a
    4090); the service must call the eager path, which is bit-identical and has no build."""
    import numpy as np

    from backend import GenerationRequest
    from backends.tensorrt_backend import TensorRTBackend

    calls = []

    class FakeInference:
        def generate(self, *a, **k):  # pragma: no cover - must not be called
            raise AssertionError("graph path used")

        def generate_eager(self, prompt, *, seconds, steps, seed, cfg):
            calls.append((prompt, seconds, steps, seed, cfg))
            return np.zeros((int(seconds * 44100), 2), dtype=np.int16), {}

    backend = TensorRTBackend()
    backend._inference = FakeInference()
    result = backend.generate(GenerationRequest("rain", 30, 4294967295, "mp3"))
    assert calls == [("rain", 30.0, 8, 4294967295, 1.0)]
    assert result.seed == 4294967295


def test_real_engine_lock_from_the_4090_pod_passes_manifest_check(tmp_path):
    """The engines.lock.json produced on pod 89exl83ei78ozh (and uploaded to the private
    engine repo) must satisfy the worker's manifest check for upstream's required files."""
    import json
    import shutil
    from pathlib import Path

    # Copy of benchmarks/results/2026-09-26-tensorrt-4090/engines.lock.json, kept inside worker/
    # so the test also runs in the public worker repo.
    lock = Path(__file__).resolve().parent / "fixtures" / "sm89_engines.lock.json"
    shutil.copy(lock, tmp_path / "engines.lock.json")
    required = ["t5gemma/t5gemma_fp16.trt", "sa3-m/dit_fp16.trt", "same-l/dec_fp16_chunkable_limiter.trt"]
    for rel in required:
        (tmp_path / "sm_89" / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "sm_89" / rel).write_bytes(b"x")
    manifest = verify_engine_manifest(tmp_path, "sm_89", required, REQUIRED_TENSORRT_VERSION, verify_hashes=False)
    assert manifest["source_model_revision"] == "27b5a21b791b1b033d193a9e1e3ce78493f102f9"
    assert json.loads(lock.read_text())["tensorrt_version"] == REQUIRED_TENSORRT_VERSION


def test_worker_refuses_endpoint_revision_mismatch(monkeypatch):
    import pytest

    from backend import WorkerFailure
    from handler import check_deployed_revision

    monkeypatch.setenv("SA3_MODEL_REVISION", "0" * 40)
    with pytest.raises(WorkerFailure) as exc:
        check_deployed_revision()
    assert exc.value.code == "runtime_mismatch"
    monkeypatch.setenv("SA3_MODEL_REVISION", "27b5a21b791b1b033d193a9e1e3ce78493f102f9")
    check_deployed_revision()
    monkeypatch.delenv("SA3_MODEL_REVISION")
    check_deployed_revision()
