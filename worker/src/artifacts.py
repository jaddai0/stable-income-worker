"""Fail-closed resolution of the pinned model snapshot (build-plan decision 4).

RunPod's cached-model feature places Hugging Face repositories under
/runpod-volume/huggingface-cache/hub/ in the standard HF cache layout but cannot pin a
revision. This module therefore refuses to start unless the snapshot directory for the
pinned revision exists and every required file has the expected size (and, where the
cache uses blob symlinks, the expected content-addressed blob name). Nothing is ever
downloaded here or at request time.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from backend import PINNED_MODEL_REVISION, WorkerFailure

DEFAULT_CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"
REPO_DIR_NAME = "models--stabilityai--stable-audio-3-medium"

# Sizes and blob ids from the Hugging Face tree API at the pinned revision (retrieved
# 2026-09-26). LFS blobs are named by content sha256; small git files by git blob oid.
REQUIRED_FILES: dict[str, dict] = {
    "model_config.json": {"size": 10360, "blob": "331bfb7e610b15000c930a38f0543f75f3b132be"},
    "model.safetensors": {
        "size": 9222116660,
        "blob": "48d9c65e290e7bcd5194e0633bfc2424a59ee9683f5c2d58762d997b7d8ce0b5",
        "sha256": "48d9c65e290e7bcd5194e0633bfc2424a59ee9683f5c2d58762d997b7d8ce0b5",
    },
    "t5gemma-b-b-ul2/config.json": {"size": 2540, "blob": "f5282e2a2964d231e0f19cb2a23f39e08de83128"},
    "t5gemma-b-b-ul2/generation_config.json": {"size": 156, "blob": "a1afbdafc25667679ab591fc9aca56d4a0b91c3d"},
    "t5gemma-b-b-ul2/model.safetensors": {
        "size": 1183022944,
        "blob": "9b05ea5a4f211d023832f706fb2c0e83e4fc721b6da35ab69ceb0b55eb7800d3",
        "sha256": "9b05ea5a4f211d023832f706fb2c0e83e4fc721b6da35ab69ceb0b55eb7800d3",
    },
    "t5gemma-b-b-ul2/special_tokens_map.json": {"size": 636, "blob": "8d6368f7e735fbe4781bf6e956b7c6ad0586df80"},
    "t5gemma-b-b-ul2/tokenizer.json": {
        "size": 34362429,
        "blob": "7794135caa3ea73918949c902a781cc61dab674a4b59c17d85931c77c1114cbd",
        "sha256": "7794135caa3ea73918949c902a781cc61dab674a4b59c17d85931c77c1114cbd",
    },
    "t5gemma-b-b-ul2/tokenizer.model": {
        "size": 4241003,
        "blob": "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2",
        "sha256": "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2",
    },
    "t5gemma-b-b-ul2/tokenizer_config.json": {"size": 46437, "blob": "4d625a54ce31c6629f25dadcc7d878e771b40ef7"},
}


@dataclass(frozen=True)
class Snapshot:
    root: Path
    revision: str

    def path(self, relative: str) -> Path:
        return self.root / relative


def set_offline_env() -> None:
    """Forbid Hugging Face network access, and prove it took effect.

    huggingface_hub reads HF_HUB_OFFLINE into a module constant when first imported, so
    setting the variable after that import is silently ignored. handler.py sets these
    before any import; this function repeats them for direct backend use and fails closed
    if the hub was already imported in online mode.
    """
    import sys

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    hub_constants = sys.modules.get("huggingface_hub.constants")
    if hub_constants is not None and not getattr(hub_constants, "HF_HUB_OFFLINE", False):
        raise WorkerFailure("runtime_mismatch",
                            "huggingface_hub was imported before offline mode was set")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_snapshot(cache_root: str | None = None, revision: str = PINNED_MODEL_REVISION,
                     verify_hashes: bool | None = None) -> Snapshot:
    """Return the pinned snapshot or raise WorkerFailure('missing_artifact').

    verify_hashes (env SA3_VERIFY_WEIGHT_HASHES=1) re-hashes the LFS files. That reads
    ~10.4 GB and lengthens every cold start, so it is off by default; blob-name and size
    checks run always.
    """
    root = Path(cache_root or os.environ.get("SA3_HF_CACHE", DEFAULT_CACHE_ROOT))
    if verify_hashes is None:
        verify_hashes = os.environ.get("SA3_VERIFY_WEIGHT_HASHES", "0") == "1"
    snapshot = root / REPO_DIR_NAME / "snapshots" / revision
    if not snapshot.is_dir():
        raise WorkerFailure("missing_artifact",
                            "pinned model revision is not present in the model cache")
    for relative, expected in REQUIRED_FILES.items():
        path = snapshot / relative
        if not path.is_file():
            raise WorkerFailure("missing_artifact", f"pinned snapshot is missing {relative}")
        size = path.stat().st_size
        if size != expected["size"]:
            raise WorkerFailure("missing_artifact", f"pinned snapshot file has the wrong size: {relative}")
        if path.is_symlink():
            blob_name = Path(os.readlink(path)).name
            if blob_name != expected["blob"]:
                raise WorkerFailure("missing_artifact",
                                    f"pinned snapshot file points at an unexpected blob: {relative}")
        if verify_hashes and "sha256" in expected and _sha256_file(path) != expected["sha256"]:
            raise WorkerFailure("missing_artifact", f"pinned snapshot file failed its checksum: {relative}")
    return Snapshot(root=snapshot, revision=revision)
