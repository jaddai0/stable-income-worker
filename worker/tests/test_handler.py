from __future__ import annotations

import json
import threading
import time

import pytest

import handler
from backend import BackendIdentity, WorkerFailure
from backends.mock_backend import MockBackend
from conftest import CANARIES, SECRET_CAPABILITY, SECRET_PROMPT, claim_payload, runpod_job
from execution_claim import ClaimGranted, ClaimRejected


class FakeClient:
    """Scripted stand-in for GatewayClient that records every call."""

    def __init__(self, claim=None, complete=None, claim_error=None, complete_error=None):
        self.claim_result = claim if claim is not None else _granted()
        self.complete_result = complete or {"job_id": "gen_abcdefghijklmnop1234", "accepted": True,
                                            "state": "SUCCEEDED"}
        self.claim_error = claim_error
        self.complete_error = complete_error
        self.claims = []
        self.completions = []

    def claim(self, job_id, attempt, runpod_job_id, worker):
        self.claims.append({"job_id": job_id, "attempt": attempt, "runpod_job_id": runpod_job_id,
                            "worker": worker})
        if self.claim_error:
            raise self.claim_error
        return self.claim_result

    def complete(self, job_id, body):
        self.completions.append(json.loads(json.dumps(body)))
        if self.complete_error:
            raise self.complete_error
        return self.complete_result


def _granted(**kwargs) -> ClaimGranted:
    p = claim_payload(**kwargs)
    return ClaimGranted(p["job_id"], p["attempt"], p["attempt_id"], p["fencing_token"],
                        p["claim_expires_at"], p["generation"], p["upload"])


class RecordingUpload:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def __call__(self, path, upload, content_type):
        self.calls.append({"bytes": path.read_bytes(), "upload": upload, "content_type": content_type})
        if self.error:
            raise self.error


@pytest.fixture
def state():
    return handler.init_worker(MockBackend(), timeout_s=10)


def run(state, client, upload=None, job=None):
    upload = upload or RecordingUpload()
    manifest = handler.process_job(job or runpod_job(), state, client_factory=lambda cap: client,
                                   upload_fn=upload)
    return manifest, upload


def test_success_mp3_uploads_once_and_completes(state, validate):
    client = FakeClient()
    manifest, upload = run(state, client)
    assert manifest["status"] == "succeeded"
    assert len(upload.calls) == 1 and len(client.completions) == 1
    body = client.completions[0]
    assert body["outcome"] == "succeeded" and body["fencing_token"] == 7
    import hashlib
    assert body["output"]["sha256"] == hashlib.sha256(upload.calls[0]["bytes"]).hexdigest()
    assert body["output"]["bytes"] == len(upload.calls[0]["bytes"])
    assert manifest["output"] == body["output"]
    assert manifest["output"]["content_type"] == "audio/mpeg"
    validate(manifest, "worker-result.schema.json")
    validate(body, "internal-api.schema.json", "/$defs/CompleteRequest")
    claim_body = {k: client.claims[0][k] for k in ("attempt", "runpod_job_id", "worker")}
    validate(claim_body, "internal-api.schema.json", "/$defs/ClaimRequest")


def test_success_wav(state, validate):
    client = FakeClient(claim=_granted(fmt="wav", duration=7))
    manifest, upload = run(state, client)
    assert manifest["status"] == "succeeded"
    assert manifest["output"]["encoding"] == "wav_pcm16"
    assert abs(manifest["output"]["duration_seconds"] - 7) <= 0.1
    validate(manifest, "worker-result.schema.json")


def test_success_flac(state, validate):
    client = FakeClient(claim=_granted(fmt="flac", duration=7))
    manifest, upload = run(state, client)
    assert manifest["status"] == "succeeded"
    assert manifest["output"]["encoding"] == "flac_16"
    assert manifest["output"]["content_type"] == "audio/flac"
    assert manifest["output"]["object_key"].endswith(".flac")
    validate(manifest, "worker-result.schema.json")
    validate(client.completions[0], "internal-api.schema.json", "/$defs/CompleteRequest")


@pytest.mark.parametrize("status", [409, 410])
def test_claim_rejected_generates_nothing(status, validate):
    backend = MockBackend()
    state = handler.init_worker(backend, timeout_s=10)
    client = FakeClient(claim=ClaimRejected(status, "held"))
    manifest, upload = run(state, client)
    assert manifest["status"] == "claim_rejected"
    assert backend.generate_calls == 0 and upload.calls == [] and client.completions == []
    assert "error" not in manifest
    validate(manifest, "worker-result.schema.json")


def test_claim_refused_credential_is_non_retryable(validate):
    backend = MockBackend()
    state = handler.init_worker(backend, timeout_s=10)
    client = FakeClient(claim=ClaimRejected(403, "refused", {"code": "validation", "retryable": False,
                                                            "message": "gateway refused the execution claim (HTTP 403)"}))
    manifest, _ = run(state, client)
    assert manifest["status"] == "claim_rejected"
    assert manifest["error"]["retryable"] is False and backend.generate_calls == 0
    validate(manifest, "worker-result.schema.json")


def test_gateway_unreachable_at_claim_generates_nothing(validate):
    backend = MockBackend()
    state = handler.init_worker(backend, timeout_s=10)
    client = FakeClient(claim_error=WorkerFailure("gateway_unreachable", "gateway claim failed"))
    manifest, upload = run(state, client)
    assert manifest["status"] == "failed"
    assert manifest["error"] == {"code": "gateway_unreachable", "retryable": True, "message": "gateway claim failed"}
    assert backend.generate_calls == 0 and upload.calls == []
    validate(manifest, "worker-result.schema.json")


def test_token_limit_rejects_before_inference(validate):
    backend = MockBackend(token_counter=lambda prompt: 257)
    state = handler.init_worker(backend, timeout_s=10)
    client = FakeClient()
    manifest, upload = run(state, client)
    assert manifest["status"] == "failed"
    assert manifest["error"]["code"] == "validation" and manifest["error"]["retryable"] is False
    assert backend.generate_calls == 0 and upload.calls == []
    assert client.completions[0]["outcome"] == "failed"
    validate(client.completions[0], "internal-api.schema.json", "/$defs/CompleteRequest")
    validate(manifest, "worker-result.schema.json")


def test_token_limit_boundary_256_is_allowed():
    backend = MockBackend(token_counter=lambda prompt: 256)
    state = handler.init_worker(backend, timeout_s=10)
    manifest, _ = run(state, FakeClient())
    assert manifest["status"] == "succeeded"


@pytest.mark.parametrize("code,retryable", [("oom", False), ("deterministic_runtime", False), ("internal", True)])
def test_backend_failure_reports_failed_completion(code, retryable, validate):
    state = handler.init_worker(MockBackend(fail_code=code), timeout_s=10)
    client = FakeClient()
    manifest, upload = run(state, client)
    assert manifest["status"] == "failed" and upload.calls == []
    assert manifest["error"]["code"] == code
    assert manifest["error"]["retryable"] is retryable
    assert client.completions[0]["error"]["code"] == code
    validate(manifest, "worker-result.schema.json")


def test_generation_shorter_than_requested_is_rejected_not_padded(validate):
    # Regression for the measured upstream default sample_size (~120 s): a 180 s request that
    # comes back 120 s long must fail, never be padded or reported as success.
    class ShortBackend(MockBackend):
        def generate(self, request):
            result = super().generate(request)
            result.samples = result.samples[:, : 120 * 44100]
            return result

    state = handler.init_worker(ShortBackend(), timeout_s=10)
    client = FakeClient(claim=_granted(duration=180, fmt="wav"))
    manifest, upload = run(state, client)
    assert manifest["status"] == "failed"
    assert manifest["error"]["code"] == "invalid_output" and upload.calls == []
    validate(manifest, "worker-result.schema.json")


def test_upload_failure_reports_failed(validate):
    state = handler.init_worker(MockBackend(), timeout_s=10)
    client = FakeClient()
    upload = RecordingUpload(error=WorkerFailure("upload_failed", "upload failed after 3 tries (HTTP 503)"))
    manifest, _ = run(state, client, upload=upload)
    assert manifest["status"] == "failed"
    assert manifest["error"]["code"] == "upload_failed" and manifest["error"]["retryable"] is True
    assert client.completions[0]["outcome"] == "failed"
    validate(manifest, "worker-result.schema.json")


def test_completion_unreachable_after_upload_is_unconfirmed(validate):
    state = handler.init_worker(MockBackend(), timeout_s=10)
    client = FakeClient(complete_error=WorkerFailure("gateway_unreachable", "gateway complete failed"))
    manifest, upload = run(state, client)
    assert manifest["status"] == "completion_unconfirmed"
    assert len(upload.calls) == 1 and manifest["output"]["bytes"] > 0
    assert manifest["attempt_id"] == "att_0123456789ab" and manifest["fencing_token"] == 7
    validate(manifest, "worker-result.schema.json")


def test_stale_completion_is_not_retried(validate):
    state = handler.init_worker(MockBackend(), timeout_s=10)
    client = FakeClient(complete={"job_id": "gen_abcdefghijklmnop1234", "accepted": False, "state": "SUCCEEDED"})
    manifest, upload = run(state, client)
    assert len(client.completions) == 1 and len(upload.calls) == 1
    validate(manifest, "worker-result.schema.json")


def test_model_loaded_once_across_repeated_jobs():
    before = MockBackend.load_count
    state = handler.init_worker(MockBackend(), timeout_s=10)
    manifests = [run(state, FakeClient())[0] for _ in range(3)]
    assert MockBackend.load_count == before + 1
    assert [m["worker"]["jobs_served_by_process"] for m in manifests] == [1, 2, 3]
    assert len({m["worker"]["worker_process_id"] for m in manifests}) == 1
    assert [m["measurements"]["cold_process"] for m in manifests] == [True, False, False]
    assert manifests[0]["measurements"]["model_load_s"] is not None
    assert manifests[1]["measurements"]["model_load_s"] is None


def test_invalid_input_never_claims(validate):
    state = handler.init_worker(MockBackend(), timeout_s=10)
    client = FakeClient()
    job = runpod_job()
    job["input"]["prompt"] = SECRET_PROMPT  # prompts are never allowed in RunPod input
    manifest, _ = run(state, client, job=job)
    assert manifest["status"] == "failed" and manifest["error"]["code"] == "validation"
    assert client.claims == []
    validate(manifest, "worker-result.schema.json")


@pytest.mark.parametrize("job", [None, {}, {"input": "x"}, {"input": {"job_id": 5}}, {"id": 1, "input": []}])
def test_garbage_jobs_never_raise(job):
    state = handler.init_worker(MockBackend(), timeout_s=10)
    manifest = handler.process_job(job, state, client_factory=lambda cap: FakeClient(),
                                   upload_fn=RecordingUpload())
    assert manifest["status"] == "failed"


def test_nothing_secret_reaches_manifest_or_logs(capsys):
    class LeakyBackend(MockBackend):
        def generate(self, request):
            raise RuntimeError(f"upstream exploded on {request.prompt}")

    for backend, client in [
        (MockBackend(), FakeClient()),
        (LeakyBackend(), FakeClient()),
        (MockBackend(), FakeClient(complete_error=WorkerFailure("gateway_unreachable", "gateway complete failed"))),
    ]:
        state = handler.init_worker(backend, timeout_s=10)
        manifest, _ = run(state, client)
        text = json.dumps(manifest) + json.dumps(client.completions) + json.dumps(client.claims)
        for canary in CANARIES:
            assert canary not in text, canary
    out = capsys.readouterr()
    for canary in CANARIES:
        assert canary not in out.out and canary not in out.err, canary


def test_manifest_is_small():
    state = handler.init_worker(MockBackend(), timeout_s=10)
    manifest, _ = run(state, FakeClient())
    assert len(json.dumps(manifest).encode()) < handler.MAX_MANIFEST_BYTES


def test_init_timeout_is_bounded():
    class SlowBackend(MockBackend):
        def load(self):
            time.sleep(5)
            return super().load()

    started = time.perf_counter()
    with pytest.raises(WorkerFailure) as info:
        handler.init_worker(SlowBackend(), timeout_s=0.2)
    assert info.value.code == "timeout" and time.perf_counter() - started < 2


def test_init_failure_is_sanitised():
    class BadBackend(MockBackend):
        def load(self):
            raise RuntimeError("secret detail " + SECRET_CAPABILITY)

    with pytest.raises(WorkerFailure) as info:
        handler.init_worker(BadBackend(), timeout_s=5)
    assert info.value.code == "internal" and "CANARY" not in info.value.message


def test_jobs_are_serialised_per_process():
    state = handler.init_worker(MockBackend(latency_s=0.05), timeout_s=10)
    active = {"now": 0, "max": 0}
    lock = threading.Lock()

    def upload(path, upload, content_type):
        with lock:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        time.sleep(0.02)
        with lock:
            active["now"] -= 1

    threads = [threading.Thread(target=handler.process_job,
                                args=(runpod_job(), state, lambda cap: FakeClient(), upload))
               for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert active["max"] == 1
