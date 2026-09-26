"""Worker side of the internal claim/complete API (contracts/internal-api.schema.json).

Every call carries two credentials: the worker bearer token (worker privilege) and the
per-job execution capability from the RunPod input. The gateway base URL comes only from
worker environment, never from job input.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Optional, Union

import requests

from backend import WorkerFailure
from telemetry import log_event

MAX_TRIES = 3
BACKOFF_S = (0.5, 2.0)
TIMEOUT_S = (5.0, 20.0)

CLAIM_REJECT_STATUSES = {409: "claim held by another holder or job terminal", 410: "job expired or cancelled"}
CLAIM_REFUSED_STATUSES = {400, 401, 403, 404, 422}


@dataclass(frozen=True)
class ClaimGranted:
    job_id: str
    attempt: int
    attempt_id: str
    fencing_token: int
    claim_expires_at: int
    generation: dict
    upload: dict


@dataclass(frozen=True)
class ClaimRejected:
    http_status: int
    reason: str
    error: Optional[dict] = None  # set when the refusal points at a bad credential or input


class GatewayClient:
    def __init__(self, base_url: str, worker_token: str, capability: str, *,
                 session: Optional[requests.Session] = None,
                 sleep: Callable[[float], None] = time.sleep):
        if not base_url:
            raise WorkerFailure("internal", "gateway base URL is not configured")
        if not base_url.startswith("https://") and os.environ.get("SA3_ALLOW_INSECURE_GATEWAY") != "1":
            raise WorkerFailure("internal", "gateway base URL must use https")
        if not worker_token:
            raise WorkerFailure("internal", "worker token is not configured")
        self._base = base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {worker_token}",
            "X-Execution-Capability": capability,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self._http = session or requests.Session()
        self._sleep = sleep

    @classmethod
    def from_env(cls, capability: str, **kwargs) -> "GatewayClient":
        return cls(os.environ.get("GATEWAY_BASE_URL", ""), os.environ.get("WORKER_TOKEN", ""),
                   capability, **kwargs)

    def _post(self, path: str, body: dict, stage: str) -> requests.Response:
        """POST with bounded retries on connection errors, 429 and 5xx only."""
        last_status = None
        for attempt in range(1, MAX_TRIES + 1):
            try:
                response = self._http.post(self._base + path, json=body, headers=self._headers,
                                           timeout=TIMEOUT_S)
                last_status = response.status_code
                if response.status_code < 500 and response.status_code != 429:
                    return response
                log_event("gateway_http_error", stage=stage, http_status=response.status_code,
                          **{"try": attempt})
            except requests.RequestException:
                log_event("gateway_connection_error", stage=stage, **{"try": attempt})
            if attempt < MAX_TRIES:
                self._sleep(BACKOFF_S[min(attempt - 1, len(BACKOFF_S) - 1)])
        detail = f"HTTP {last_status}" if last_status else "connection error"
        raise WorkerFailure("gateway_unreachable", f"gateway {stage} failed after {MAX_TRIES} tries ({detail})")

    def claim(self, job_id: str, attempt: int, runpod_job_id: str,
              worker: dict) -> Union[ClaimGranted, ClaimRejected]:
        body = {"attempt": attempt, "runpod_job_id": runpod_job_id, "worker": worker}
        response = self._post(f"/internal/jobs/{job_id}/claim", body, "claim")
        status = response.status_code
        if status in CLAIM_REJECT_STATUSES:
            return ClaimRejected(status, CLAIM_REJECT_STATUSES[status])
        if status in CLAIM_REFUSED_STATUSES:
            return ClaimRejected(status, "claim refused", {
                "code": "validation", "retryable": False,
                "message": f"gateway refused the execution claim (HTTP {status})"})
        if status != 200:
            raise WorkerFailure("gateway_unreachable", f"unexpected claim response (HTTP {status})")
        try:
            data = response.json()
            granted = ClaimGranted(
                job_id=str(data["job_id"]), attempt=int(data["attempt"]),
                attempt_id=str(data["attempt_id"]), fencing_token=int(data["fencing_token"]),
                claim_expires_at=int(data["claim_expires_at"]),
                generation=dict(data["generation"]), upload=dict(data["upload"]))
        except (ValueError, KeyError, TypeError):
            raise WorkerFailure("internal", "claim response is malformed") from None
        if granted.job_id != job_id or granted.attempt != attempt:
            raise WorkerFailure("internal", "claim response is for a different job or attempt")
        return granted

    def complete(self, job_id: str, body: dict) -> dict:
        """Returns the CompleteResponse. Raises gateway_unreachable if no answer arrived."""
        response = self._post(f"/internal/jobs/{job_id}/complete", body, "complete")
        if response.status_code in (200, 409):
            try:
                data = response.json()
                return {"job_id": str(data["job_id"]), "accepted": bool(data["accepted"]),
                        "state": str(data["state"])}
            except (ValueError, KeyError, TypeError):
                raise WorkerFailure("internal", "completion response is malformed") from None
        raise WorkerFailure("gateway_unreachable",
                            f"gateway rejected the completion (HTTP {response.status_code})",
                            retryable=False)
