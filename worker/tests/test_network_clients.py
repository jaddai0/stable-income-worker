from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from backend import WorkerFailure
from conftest import (CANARIES, SECRET_CAPABILITY, SECRET_URL, SECRET_WORKER_TOKEN, claim_payload)
from execution_claim import ClaimGranted, ClaimRejected, GatewayClient
from upload import upload_file


class Response:
    def __init__(self, status, body=None):
        self.status_code = status
        self._body = body

    def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body

    def close(self):
        pass


class Session:
    """Scripted requests.Session: each item is a Response or an exception to raise."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def _next(self, method, url, **kwargs):
        body = kwargs.get("data")
        self.calls.append({"method": method, "url": url, "headers": kwargs.get("headers"),
                           "json": kwargs.get("json"),
                           "data": body.read() if hasattr(body, "read") else body})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def post(self, url, **kwargs):
        return self._next("POST", url, **kwargs)

    def put(self, url, **kwargs):
        return self._next("PUT", url, **kwargs)


def client(script):
    session = Session(script)
    return GatewayClient("https://gw.example.invalid", SECRET_WORKER_TOKEN, SECRET_CAPABILITY,
                         session=session, sleep=lambda s: None), session


WORKER = {"backend": "mock", "model_id": "stabilityai/stable-audio-3-medium",
          "model_revision": "mock", "runtime_version": "test"}
JOB = "gen_abcdefghijklmnop1234"


def test_claim_sends_both_credentials_and_parses(validate):
    c, session = client([Response(200, claim_payload())])
    granted = c.claim(JOB, 1, "rp-1", WORKER)
    assert isinstance(granted, ClaimGranted) and granted.fencing_token == 7
    call = session.calls[0]
    assert call["url"] == f"https://gw.example.invalid/internal/jobs/{JOB}/claim"
    assert call["headers"]["Authorization"] == f"Bearer {SECRET_WORKER_TOKEN}"
    assert call["headers"]["X-Execution-Capability"] == SECRET_CAPABILITY
    validate(call["json"], "internal-api.schema.json", "/$defs/ClaimRequest")
    validate(claim_payload(), "internal-api.schema.json", "/$defs/ClaimResponse")


@pytest.mark.parametrize("status", [409, 410])
def test_claim_conflict_is_rejection(status):
    c, _ = client([Response(status, {})])
    result = c.claim(JOB, 1, "rp-1", WORKER)
    assert isinstance(result, ClaimRejected) and result.error is None


@pytest.mark.parametrize("status", [401, 403, 404, 422])
def test_claim_refusal_is_non_retryable_rejection(status):
    c, _ = client([Response(status, {})])
    result = c.claim(JOB, 1, "rp-1", WORKER)
    assert isinstance(result, ClaimRejected) and result.error["retryable"] is False


def test_claim_retries_transient_then_gives_up():
    c, session = client([Response(503), requests.ConnectionError(SECRET_URL), Response(500)])
    with pytest.raises(WorkerFailure) as info:
        c.claim(JOB, 1, "rp-1", WORKER)
    assert info.value.code == "gateway_unreachable" and info.value.retryable is True
    assert len(session.calls) == 3 and "CANARY" not in info.value.message


def test_claim_recovers_after_transient():
    c, session = client([Response(502), Response(200, claim_payload())])
    assert isinstance(c.claim(JOB, 1, "rp-1", WORKER), ClaimGranted)
    assert len(session.calls) == 2


def test_claim_for_wrong_job_is_refused():
    c, _ = client([Response(200, claim_payload(job_id="gen_zzzzzzzzzzzzzzzzzzzz"))])
    with pytest.raises(WorkerFailure) as info:
        c.claim(JOB, 1, "rp-1", WORKER)
    assert info.value.code == "internal"


def test_malformed_claim_is_internal():
    c, _ = client([Response(200, {"job_id": JOB})])
    with pytest.raises(WorkerFailure) as info:
        c.claim(JOB, 1, "rp-1", WORKER)
    assert info.value.code == "internal"


def test_http_gateway_refused_unless_allowed(monkeypatch):
    monkeypatch.delenv("SA3_ALLOW_INSECURE_GATEWAY", raising=False)
    with pytest.raises(WorkerFailure):
        GatewayClient("http://gw.example.invalid", "t", "c")
    monkeypatch.setenv("SA3_ALLOW_INSECURE_GATEWAY", "1")
    GatewayClient("http://localhost:8787", "t", "c")


def test_complete_returns_answer_and_accepts_conflict(validate):
    answer = {"job_id": JOB, "accepted": False, "state": "SUCCEEDED"}
    c, session = client([Response(409, answer)])
    assert c.complete(JOB, {"x": 1}) == answer
    validate(answer, "internal-api.schema.json", "/$defs/CompleteResponse")
    assert session.calls[0]["url"].endswith(f"/internal/jobs/{JOB}/complete")


def test_complete_unreachable_raises():
    c, _ = client([requests.Timeout(), requests.Timeout(), requests.Timeout()])
    with pytest.raises(WorkerFailure) as info:
        c.complete(JOB, {})
    assert info.value.code == "gateway_unreachable"


# --- upload -----------------------------------------------------------------------------

def upload_spec(**overrides):
    spec = dict(claim_payload()["upload"])
    spec.update(overrides)
    return spec


def put(path, script, **spec_overrides):
    session = Session(script)
    upload_file(path, upload_spec(**spec_overrides), "audio/mpeg", session=session, sleep=lambda s: None)
    return session


@pytest.fixture
def file(tmp_path) -> Path:
    path = tmp_path / "audio.mp3"
    path.write_bytes(b"\xff\xfb" + b"a" * 1000)
    return path


def test_upload_streams_file_with_headers(file):
    session = put(file, [Response(200)])
    call = session.calls[0]
    assert call["method"] == "PUT" and call["url"] == SECRET_URL
    assert call["data"] == file.read_bytes()
    assert call["headers"]["Content-Length"] == str(file.stat().st_size)
    assert call["headers"]["x-amz-meta-attempt"] == "1"


def test_upload_retries_transient(file):
    session = put(file, [Response(503), requests.ConnectionError(SECRET_URL), Response(200)])
    assert len(session.calls) == 3


def test_upload_permanent_failure_not_retried(file):
    with pytest.raises(WorkerFailure) as info:
        put(file, [Response(403)])
    assert info.value.code == "upload_failed" and info.value.retryable is False


def test_upload_gives_up_after_three(file, capsys):
    with pytest.raises(WorkerFailure) as info:
        put(file, [requests.ConnectionError(SECRET_URL)] * 3)
    assert info.value.retryable is True
    out = capsys.readouterr().out
    for canary in CANARIES:
        assert canary not in out and canary not in info.value.message


def test_upload_expired_authorization(file):
    with pytest.raises(WorkerFailure):
        put(file, [Response(200)], expires_at=1)


def test_upload_size_cap(file):
    with pytest.raises(WorkerFailure) as info:
        put(file, [Response(200)], max_bytes=10)
    assert info.value.code == "invalid_output"
