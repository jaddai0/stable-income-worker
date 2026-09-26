"""Optimized candidate: the official TensorRT runtime (HANDOFF section 6.3).

Adapter over upstream optimized/tensorRT/scripts (commit 779434a9): SA3Inference with the
medium DiT at fp16 and the SAME-L decoder (canonical chunkable + baked limiter). Engines
are built offline for one GPU architecture and described by an `engines.lock.json`
manifest in the models directory; this backend refuses to start unless every engine file
matches the manifest, the detected GPU architecture matches, and the installed TensorRT is
the exact version the engines were built with.

Guarantees:
- No engine download or build in the request path: upstream's lazy `_ensure_files`
  (which downloads from Hugging Face or offers an interactive build) is replaced with a
  strict existence check before anything is constructed.
- No silent fallback to another backend, precision, decoder or architecture.
- Known upstream failure mode (broken SAME-L capture => byte-identical noise across seeds)
  is checked at init with two warm-up seeds.

Known upstream caveat to qualify on the GPU: upstream documents that the SAME-L decoder
engine carries internal state across calls, so repeated generations in one process can
drift slightly from a fresh process. Seed reproducibility is therefore only claimed for
matched call histories until measured.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from artifacts import set_offline_env
from backend import (MODEL_ID, SAMPLE_RATE_HZ, AudioResult, BackendIdentity, GenerationRequest,
                     WorkerFailure, float_audio_from_int16)
from telemetry import log_event

REQUIRED_TENSORRT_VERSION = "10.15.1.29"
DIT = "medium"
DECODER = "same-l"
PRECISION = "fp16"
DEC_PRECISION = "canonical"
STEPS = 8
TOKENIZER_SHA256 = "7794135caa3ea73918949c902a781cc61dab674a4b59c17d85931c77c1114cbd"
DEFAULT_SCRIPTS_DIR = "/opt/sa3/optimized/tensorRT/scripts"
MANIFEST_NAME = "engines.lock.json"
WARMUP_PROMPT = "Warm acoustic guitar melody with soft percussion"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_engine_manifest(models_dir: Path, arch: str, required_files: list[str],
                           tensorrt_version: str, verify_hashes: bool = True) -> dict:
    """Check the manifest against the files on disk. Returns the parsed manifest."""
    manifest_path = models_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise WorkerFailure("missing_artifact", "TensorRT engine manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text())
    except ValueError:
        raise WorkerFailure("missing_artifact", "TensorRT engine manifest is unreadable") from None
    if manifest.get("arch") != arch:
        raise WorkerFailure("unsupported_engine", "TensorRT engines were built for a different GPU architecture")
    if manifest.get("tensorrt_version") != REQUIRED_TENSORRT_VERSION or tensorrt_version != REQUIRED_TENSORRT_VERSION:
        raise WorkerFailure("unsupported_engine", "TensorRT version does not match the engines")
    if manifest.get("precision") != PRECISION:
        raise WorkerFailure("unsupported_engine", "TensorRT engines are not the qualified precision")
    files = manifest.get("files") or {}
    if sorted(files) != sorted(required_files):
        raise WorkerFailure("missing_artifact", "TensorRT engine manifest does not list the required engines")
    for relative, expected_sha in files.items():
        path = models_dir / arch / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise WorkerFailure("missing_artifact", f"TensorRT engine is missing: {relative}")
        if verify_hashes and _sha256_file(path) != expected_sha:
            raise WorkerFailure("missing_artifact", f"TensorRT engine failed its checksum: {relative}")
    manifest["_manifest_sha256"] = _sha256_file(manifest_path)
    return manifest


class TensorRTBackend:
    name = "tensorrt"

    def __init__(self, runtime_version: str = "unknown"):
        self._runtime_version = runtime_version
        self._inference = None
        self._canon = None

    def load(self) -> BackendIdentity:
        set_offline_env()
        scripts_dir = Path(os.environ.get("SA3_TRT_SCRIPTS", DEFAULT_SCRIPTS_DIR))
        models_value = os.environ.get("SA3_TRT_MODELS_DIR", "")
        if not models_value:
            raise WorkerFailure("missing_artifact", "TensorRT models directory is not configured")
        models_dir = Path(models_value)
        expected_arch = os.environ.get("SA3_TRT_EXPECTED_ARCH", "sm_89")
        tokenizer_path = scripts_dir / "tokenizer.json"
        if not tokenizer_path.is_file() or _sha256_file(tokenizer_path) != TOKENIZER_SHA256:
            raise WorkerFailure("missing_artifact", "TensorRT tokenizer does not match the pinned model tokenizer")

        sys.path.insert(0, str(scripts_dir))
        try:
            import tensorrt
            import sa3_trt
            import sa3_trt_core as canon
        except ImportError:
            raise WorkerFailure("runtime_mismatch", "TensorRT runtime is not installed in this image") from None
        self._canon = canon
        if canon.ARCH != expected_arch:
            raise WorkerFailure("unsupported_engine", "this GPU's architecture has no qualified engines")

        required = canon.get_engine_files(DIT, DECODER, PRECISION, with_encoder=False, dec_tier=DEC_PRECISION)
        manifest = verify_engine_manifest(
            models_dir, expected_arch, required, tensorrt.__version__,
            verify_hashes=os.environ.get("SA3_VERIFY_ENGINE_HASHES", "1") == "1")

        arch_dir = models_dir / expected_arch

        def strict_ensure_files(rel_paths):
            missing = [p for p in rel_paths if not (arch_dir / p).is_file()]
            if missing:
                raise WorkerFailure("missing_artifact", "a TensorRT engine is missing at load time")

        # Upstream would download from Hugging Face or prompt to build. Never here.
        canon._ensure_files = strict_ensure_files
        sa3_trt._ensure_files = strict_ensure_files

        try:
            self._inference = sa3_trt.SA3Inference(
                DIT, DECODER, precision=PRECISION, dec_precision=DEC_PRECISION,
                models_dir=models_dir, with_encoder=False, chunking=True, quiet=True)
        except WorkerFailure:
            raise
        except Exception:
            raise WorkerFailure("unsupported_engine", "TensorRT engines failed to load") from None
        if self._inference.precision != PRECISION:
            raise WorkerFailure("unsupported_engine", "TensorRT runtime changed the requested precision")

        if os.environ.get("SA3_WARMUP", "1") == "1":
            self._warmup()

        import torch

        props = torch.cuda.get_device_properties(0)
        return BackendIdentity(
            backend="tensorrt", model_id=MODEL_ID,
            model_revision=str(manifest.get("source_model_revision", "unknown")),
            runtime_version=self._runtime_version, engine_sha256=manifest["_manifest_sha256"],
            gpu_name=props.name, compute_capability=f"{props.major}.{props.minor}",
            cuda_version=torch.version.cuda, torch_version=torch.__version__,
            tensorrt_version=tensorrt.__version__)

    def _warmup(self) -> None:
        from encode import quality_flags

        first = self.generate(GenerationRequest(WARMUP_PROMPT, 5, 1, "wav"))
        second = self.generate(GenerationRequest(WARMUP_PROMPT, 5, 2, "wav"))
        flags = quality_flags(first.samples)
        log_event("warmup_done", backend="tensorrt", seconds=round(first.inference_s, 4),
                  quality_flags=flags)
        if np.array_equal(first.samples, second.samples):
            raise WorkerFailure("unsupported_engine", "different seeds produced identical audio (broken decoder capture)")
        if "static_suspect" in flags or "near_silent" in flags:
            raise WorkerFailure("unsupported_engine", "warm-up generation looks like static or silence")

    def count_prompt_tokens(self, prompt: str) -> int:
        # Mirrors upstream FastTokenizer.__call__: Tokenizer.encode(text) before truncation.
        return len(self._inference.tokenizer.tokenizer.encode(prompt).ids)

    def _process_vram_bytes(self) -> Optional[int]:
        try:
            return int(self._canon._my_vram_bytes())
        except Exception:
            return None

    def generate(self, request: GenerationRequest) -> AudioResult:
        if self._inference is None:
            raise WorkerFailure("internal", "backend used before load")
        start = time.perf_counter()
        try:
            # Eager per-stage path, not the captured mega-graph: graphs are cached per exact
            # latent length (LRU of 4), so a service with 376 possible lengths would rebuild
            # one on almost every request (4.5 s at 380 s on a 4090). Eager output measured
            # bit-identical to the graph path at the same seed, with no per-length build
            # (benchmarks/results/2026-09-26-tensorrt-4090/eager.json).
            pcm, timing = self._inference.generate_eager(
                request.prompt, seconds=float(request.duration_seconds), steps=STEPS,
                seed=int(request.seed), cfg=1.0)
        except WorkerFailure:
            raise
        except Exception:
            raise WorkerFailure("deterministic_runtime", "TensorRT generation raised an error",
                                retryable=False) from None
        inference_s = time.perf_counter() - start
        return AudioResult(
            samples=float_audio_from_int16(np.asarray(pcm)), sample_rate_hz=SAMPLE_RATE_HZ,
            channels=2, seed=int(request.seed), inference_s=inference_s, decode_s=None,
            peak_vram_bytes=self._process_vram_bytes(),
            extra={"graph_build_ms": round(float(timing.get("graph_build_ms", 0.0)), 3)})

    def close(self) -> None:
        if self._inference is not None:
            self._inference.close()
            self._inference = None
