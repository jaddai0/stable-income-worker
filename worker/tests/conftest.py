from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = REPO_ROOT / "contracts"
os.environ.setdefault("SA3_CONTRACTS_DIR", str(CONTRACTS))
os.environ.setdefault("SA3_ALLOW_INSECURE_GATEWAY", "0")

SECRET_PROMPT = "PROMPT-CANARY-7f3a lush orchestral swell"
SECRET_URL = "https://r2.example.invalid/upload?X-Amz-Signature=URL-CANARY-91bd"
SECRET_CAPABILITY = "CAPABILITY-CANARY-" + "c" * 40
SECRET_WORKER_TOKEN = "WORKER-TOKEN-CANARY-55e1"
CANARIES = ("PROMPT-CANARY", "URL-CANARY", "CAPABILITY-CANARY", "WORKER-TOKEN-CANARY")


def _registry() -> Registry:
    resources = []
    for path in CONTRACTS.glob("*.json"):
        schema = json.loads(path.read_text())
        resource = Resource.from_contents(schema)
        resources.append((path.name, resource))
        resources.append((schema["$id"], resource))
    return Registry().with_resources(resources)


REGISTRY = _registry()


def validator_for(schema_file: str, pointer: str | None = None) -> Draft202012Validator:
    schema = json.loads((CONTRACTS / schema_file).read_text())
    if pointer:
        schema = {"$ref": f"{schema['$id']}#{pointer}"}
    return Draft202012Validator(schema, registry=REGISTRY)


@pytest.fixture
def validate():
    def _validate(instance, schema_file, pointer=None):
        errors = sorted(validator_for(schema_file, pointer).iter_errors(instance), key=str)
        assert not errors, [e.message for e in errors]
    return _validate


def job_input(job_id="gen_abcdefghijklmnop1234", attempt=1, **extra):
    data = {"schema_version": 1, "job_id": job_id, "attempt": attempt,
            "capability": SECRET_CAPABILITY}
    data.update(extra)
    return data


def runpod_job(**kwargs):
    return {"id": "rp-job-0001", "input": job_input(**kwargs)}


def claim_payload(job_id="gen_abcdefghijklmnop1234", attempt=1, *, prompt=SECRET_PROMPT,
                  duration=5, seed=18421, fmt="mp3", fencing=7, max_bytes=50_000_000):
    return {
        "job_id": job_id, "attempt": attempt, "attempt_id": "att_0123456789ab",
        "fencing_token": fencing, "claim_expires_at": 4_102_444_800,
        "generation": {"prompt": prompt, "duration_seconds": duration, "seed": seed, "output_format": fmt},
        "upload": {"method": "PUT", "url": SECRET_URL, "headers": {"x-amz-meta-attempt": "1"},
                   "object_key": f"test/jobs/{job_id}/att_0123456789ab/audio.{fmt}",
                   "expires_at": 4_102_444_800, "max_bytes": max_bytes},
    }
