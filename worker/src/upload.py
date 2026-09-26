"""Streamed PUT of the encoded file to the presigned per-attempt URL (build-plan decision 2).

The worker holds no storage credentials. The URL is a bearer secret: it is never logged,
and exception text from `requests` (which embeds the URL) is never propagated.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import requests

from backend import WorkerFailure
from telemetry import log_event

MAX_TRIES = 3
BACKOFF_S = (1.0, 3.0)
TIMEOUT_S = (10.0, 120.0)  # connect, read


def upload_file(path: Path, upload: dict, content_type: str, *,
                session: Optional[requests.Session] = None,
                sleep: Callable[[float], None] = time.sleep, now: Callable[[], float] = time.time) -> None:
    """PUT the file. Retries connection errors, 408, 429 and 5xx; other statuses fail fast."""
    if upload.get("method") != "PUT":
        raise WorkerFailure("internal", "claim returned an unsupported upload method")
    size = path.stat().st_size
    if size > int(upload["max_bytes"]):
        raise WorkerFailure("invalid_output", "encoded audio exceeds the upload size limit", retryable=False)
    headers = {str(k): str(v) for k, v in upload.get("headers", {}).items()}
    headers.setdefault("Content-Type", content_type)
    headers["Content-Length"] = str(size)
    http = session or requests.Session()
    last_status = None
    for attempt in range(1, MAX_TRIES + 1):
        if now() >= float(upload["expires_at"]):
            raise WorkerFailure("upload_failed", "upload authorization expired before the upload finished")
        try:
            with path.open("rb") as body:
                response = http.put(upload["url"], data=body, headers=headers, timeout=TIMEOUT_S)
            last_status = response.status_code
            response.close()
            if 200 <= response.status_code < 300:
                log_event("upload_done", bytes=size, **{"try": attempt})
                return
            transient = response.status_code in (408, 429) or response.status_code >= 500
            log_event("upload_http_error", http_status=response.status_code, **{"try": attempt})
            if not transient:
                raise WorkerFailure("upload_failed",
                                    f"storage rejected the upload (HTTP {response.status_code})",
                                    retryable=False)
        except requests.RequestException:
            log_event("upload_connection_error", **{"try": attempt})
        if attempt < MAX_TRIES:
            sleep(BACKOFF_S[min(attempt - 1, len(BACKOFF_S) - 1)])
    detail = f"HTTP {last_status}" if last_status else "connection error"
    raise WorkerFailure("upload_failed", f"upload failed after {MAX_TRIES} tries ({detail})")
