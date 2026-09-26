from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import artifacts
from artifacts import REPO_DIR_NAME, REQUIRED_FILES, require_snapshot
from backend import PINNED_MODEL_REVISION, WorkerFailure
from prompt_tokens import TokenizerJsonCounter, check_prompt

FIXTURES = Path(__file__).parent / "fixtures"


def make_snapshot(root: Path, *, symlinks: bool, wrong_blob: str | None = None,
                  wrong_size: str | None = None, skip: str | None = None) -> Path:
    repo = root / REPO_DIR_NAME
    snap = repo / "snapshots" / PINNED_MODEL_REVISION
    blobs = repo / "blobs"
    blobs.mkdir(parents=True)
    for relative, spec in REQUIRED_FILES.items():
        if relative == skip:
            continue
        size = spec["size"] + (1 if relative == wrong_size else 0)
        blob_name = "0" * 40 if relative == wrong_blob else spec["blob"]
        target = snap / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if symlinks:
            blob = blobs / blob_name
            with blob.open("wb") as fh:
                fh.truncate(size)  # sparse: no real 9 GB written
            target.symlink_to(os.path.relpath(blob, target.parent))
        else:
            with target.open("wb") as fh:
                fh.truncate(size)
    return snap


def test_missing_snapshot_fails_closed(tmp_path):
    with pytest.raises(WorkerFailure) as info:
        require_snapshot(str(tmp_path))
    assert info.value.code == "missing_artifact" and info.value.retryable is False


@pytest.mark.parametrize("symlinks", [True, False])
def test_valid_snapshot_resolves(tmp_path, symlinks):
    make_snapshot(tmp_path, symlinks=symlinks)
    snap = require_snapshot(str(tmp_path))
    assert snap.revision == PINNED_MODEL_REVISION


def test_other_revision_is_not_accepted(tmp_path):
    make_snapshot(tmp_path, symlinks=True)
    (tmp_path / REPO_DIR_NAME / "snapshots" / PINNED_MODEL_REVISION).rename(
        tmp_path / REPO_DIR_NAME / "snapshots" / ("f" * 40))
    with pytest.raises(WorkerFailure):
        require_snapshot(str(tmp_path))


@pytest.mark.parametrize("kwargs", [
    {"skip": "model.safetensors"},
    {"wrong_size": "t5gemma-b-b-ul2/model.safetensors"},
    {"wrong_blob": "model.safetensors"},
])
def test_corrupt_snapshot_fails_closed(tmp_path, kwargs):
    make_snapshot(tmp_path, symlinks=True, **kwargs)
    with pytest.raises(WorkerFailure) as info:
        require_snapshot(str(tmp_path))
    assert info.value.code == "missing_artifact"


def test_offline_env_is_set(monkeypatch):
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(name, raising=False)
    artifacts.set_offline_env()
    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_offline_env_detects_late_hub_import(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("huggingface_hub.constants")
    fake.HF_HUB_OFFLINE = False
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", fake)
    with pytest.raises(WorkerFailure) as info:
        artifacts.set_offline_env()
    assert info.value.code == "runtime_mismatch"


def test_handler_sets_offline_before_imports():
    source = (Path(__file__).parents[1] / "src" / "handler.py").read_text()
    offline_at = source.index('"HF_HUB_OFFLINE"')
    first_project_import = source.index("from backend import")
    assert offline_at < first_project_import


# --- prompt tokens ----------------------------------------------------------------------

def test_prompt_over_limit_is_validation_error():
    with pytest.raises(WorkerFailure) as info:
        check_prompt("x", lambda p: 257)
    assert info.value.code == "validation" and info.value.retryable is False
    assert "x" not in info.value.message.replace("exceeds", "")


def test_prompt_at_limit_passes():
    assert check_prompt("x", lambda p: 256) == 256


@pytest.mark.parametrize("prompt", ["", "   ", "a" * 2001])
def test_empty_or_long_prompt_rejected(prompt):
    with pytest.raises(WorkerFailure) as info:
        check_prompt(prompt, lambda p: 1)
    assert info.value.code == "validation"


def test_tokenizer_crash_is_sanitised():
    def boom(prompt):
        raise RuntimeError(prompt)

    with pytest.raises(WorkerFailure) as info:
        check_prompt("PROMPT-CANARY", boom)
    assert "CANARY" not in info.value.message


def _real_tokenizer_json() -> Path | None:
    candidates = [os.environ.get("SA3_TEST_TOKENIZER_JSON", "")]
    hub = Path.home() / ".cache/huggingface/hub" / REPO_DIR_NAME / "snapshots" / PINNED_MODEL_REVISION
    candidates.append(str(hub / "t5gemma-b-b-ul2" / "tokenizer.json"))
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    return None


@pytest.mark.skipif(_real_tokenizer_json() is None, reason="pinned tokenizer.json not present locally")
def test_real_tokenizer_counts():
    count = TokenizerJsonCounter(str(_real_tokenizer_json()))
    example = "Instrumental dark fantasy ambience, slow bowed strings, distant bells, no percussion"
    assert 0 < check_prompt(example, count) <= 256
    assert count("x " * 400) == 401  # measured parity with transformers AutoTokenizer 5.17.0
    with pytest.raises(WorkerFailure):
        check_prompt("x " * 400, count)


def _medium_config() -> dict:
    """The pinned upstream model_config.json when present (private repo only: it comes from the
    gated Hugging Face repo and is never published); otherwise a minimal synthetic stand-in with
    just the fields these tests exercise, so the public worker repo keeps the coverage."""
    real = FIXTURES / "medium_model_config.json"
    if real.exists():
        return json.loads(real.read_text())
    return {
        "sample_size": 16777216,
        "model": {"conditioning": {"configs": [
            {"id": "prompt", "type": "t5gemma", "config": {
                "max_length": 256, "padding_mode": "learned",
                "repo_id": "stabilityai/stable-audio-3-medium", "subfolder": "t5gemma-b-b-ul2"}},
            {"id": "seconds_total", "type": "number", "config": {"min_val": 0, "max_val": 384}},
        ]}},
    }


def test_pinned_config_conditioner_is_localised(tmp_path):
    from artifacts import Snapshot
    from backends.pytorch_backend import PyTorchBackend

    config = _medium_config()
    PyTorchBackend._localise_conditioner(config, Snapshot(tmp_path, PINNED_MODEL_REVISION))
    t5 = [c for c in config["model"]["conditioning"]["configs"] if c["type"] == "t5gemma"][0]["config"]
    assert "repo_id" not in t5 and "subfolder" not in t5
    assert t5["model_path"] == str(tmp_path / "t5gemma-b-b-ul2")


def test_pinned_config_sample_size_covers_380s():
    from backends.pytorch_backend import PyTorchBackend

    config = _medium_config()
    size = PyTorchBackend.max_sample_size(config)
    assert size == 16777216 and size >= 380 * 44100
    with pytest.raises(WorkerFailure):
        PyTorchBackend.max_sample_size({"sample_size": 5292032})  # upstream generate() default


def test_pytorch_backend_passes_sample_size():
    import numpy as np

    from backend import GenerationRequest
    from backends.pytorch_backend import PyTorchBackend

    calls = {}

    class FakeTensor:
        def __init__(self, arr):
            self.arr = arr

        def __getitem__(self, i):
            return FakeTensor(self.arr[i])

        def detach(self):
            return self

        def to(self, *a, **k):
            return self

        def numpy(self):
            return self.arr

    class FakeModel:
        def generate(self, **kwargs):
            calls.update(kwargs)
            n = kwargs["duration"] * 44100
            return FakeTensor(np.zeros((1, 2, n), dtype=np.float32))

    class FakeCuda:
        OutOfMemoryError = MemoryError

        @staticmethod
        def reset_peak_memory_stats():
            pass

        @staticmethod
        def synchronize():
            pass

        @staticmethod
        def max_memory_allocated():
            return 123

        @staticmethod
        def is_available():
            return False

    class FakeTorch:
        cuda = FakeCuda
        float32 = "float32"

    backend = PyTorchBackend()
    backend._model = FakeModel()
    backend._torch = FakeTorch
    backend._sample_size = 16777216
    result = backend.generate(GenerationRequest("p", 240, 5, "mp3"))
    assert calls["sample_size"] == 16777216 and calls["seed"] == 5 and calls["steps"] == 8
    assert calls["cfg_scale"] == 1.0 and calls["batch_size"] == 1 and calls["negative_prompt"] is None
    assert result.samples.shape == (2, 240 * 44100)
