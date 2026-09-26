"""Reference backend: the official Stable Audio 3 PyTorch path (HANDOFF section 6.1).

Upstream source is pinned to commit 779434a9 (installed in Dockerfile.pytorch). The model
is built from explicit local paths inside the pinned snapshot instead of
StableAudioModel.from_pretrained, which resolves the moving `main` ref. The T5Gemma
conditioner's repo_id/subfolder is rewritten to a local `model_path` in that snapshot, so
nothing can be fetched from the network.

Checks that fail the process at init rather than serving bad audio:
- Flash Attention must be importable *by the upstream module*. Upstream silently disables
  it on ImportError, and Medium then produces static with exit code 0.
- Weight loading must not skip any tensor (upstream prints "Skipping" and continues).
- A warm-up generation must not look like static.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import time
from typing import Optional

from artifacts import Snapshot, require_snapshot, set_offline_env
from backend import (CHANNELS, MAX_DURATION_S, MODEL_ID, SAMPLE_RATE_HZ, AudioResult,
                     BackendIdentity, GenerationRequest, WorkerFailure)
from telemetry import log_event

STEPS = 8
CFG_SCALE = 1.0
WARMUP_PROMPT = "Warm acoustic guitar melody with soft percussion"
WARMUP_SECONDS = 5
WARMUP_SEED = 1


class PyTorchBackend:
    name = "pytorch"

    def __init__(self, snapshot: Optional[Snapshot] = None, runtime_version: str = "unknown"):
        self._snapshot = snapshot
        self._runtime_version = runtime_version
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._sample_size = None

    def load(self) -> BackendIdentity:
        set_offline_env()
        snapshot = self._snapshot or require_snapshot()
        try:
            import torch
        except ImportError:
            raise WorkerFailure("runtime_mismatch", "torch is not installed in this image") from None
        if not torch.cuda.is_available():
            raise WorkerFailure("runtime_mismatch", "no CUDA device is visible to the worker")
        self._torch = torch

        import huggingface_hub.constants as hub_constants
        import stable_audio_3.models.transformer as upstream_transformer
        if not hub_constants.HF_HUB_OFFLINE:
            raise WorkerFailure("runtime_mismatch", "Hugging Face offline mode did not take effect")
        if (getattr(upstream_transformer, "flash_attn_func", None) is None
                or getattr(upstream_transformer, "flash_attn_varlen_func", None) is None):
            raise WorkerFailure("runtime_mismatch",
                                "flash-attn is not usable by the upstream model; output would be static")

        from stable_audio_3.loading_utils import load_diffusion_cond
        from stable_audio_3.model import StableAudioModel

        with snapshot.path("model_config.json").open() as fh:
            model_config = json.load(fh)
        self._localise_conditioner(model_config, snapshot)

        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            model = load_diffusion_cond(model_config, str(snapshot.path("model.safetensors")),
                                        device="cuda", model_half=True)
        skipped = [line for line in captured.getvalue().splitlines() if "Skipping" in line]
        if skipped:
            raise WorkerFailure("runtime_mismatch",
                                f"{len(skipped)} checkpoint tensors did not match the model and were skipped")
        model.use_lora = False
        model.lora_names = []
        self._model = StableAudioModel(model, model_config, "cuda", True)
        self._sample_size = self.max_sample_size(model_config)
        self._tokenizer = model.conditioner.conditioners["prompt"].tokenizer

        if os.environ.get("SA3_WARMUP", "1") == "1":
            self._warmup()

        props = torch.cuda.get_device_properties(0)
        return BackendIdentity(
            backend="pytorch", model_id=MODEL_ID, model_revision=snapshot.revision,
            runtime_version=self._runtime_version, gpu_name=props.name,
            compute_capability=f"{props.major}.{props.minor}", cuda_version=torch.version.cuda,
            torch_version=torch.__version__)

    @staticmethod
    def max_sample_size(model_config: dict) -> int:
        """Generation window in samples, from the pinned config (16777216 = ~380.4 s).

        Upstream generate() defaults sample_size to 5292032 (~120.0 s) and silently returns
        120 s for any longer request (measured on a 4090, 2026-09-26), so it must always be
        passed explicitly.
        """
        sample_size = int(model_config["sample_size"])
        if sample_size < MAX_DURATION_S * SAMPLE_RATE_HZ:
            raise WorkerFailure("runtime_mismatch", "model config cannot generate the maximum duration")
        return sample_size

    @staticmethod
    def _localise_conditioner(model_config: dict, snapshot: Snapshot) -> None:
        configs = model_config["model"]["conditioning"]["configs"]
        t5 = [c for c in configs if c.get("type") == "t5gemma"]
        if len(t5) != 1:
            raise WorkerFailure("runtime_mismatch", "unexpected text-conditioner layout in model config")
        conf = t5[0]["config"]
        if conf.get("repo_id") not in (None, MODEL_ID) or conf.get("max_length") != 256:
            raise WorkerFailure("runtime_mismatch", "text conditioner does not match the pinned model")
        local = snapshot.path(conf.get("subfolder") or "t5gemma-b-b-ul2")
        conf.pop("repo_id", None)
        conf.pop("subfolder", None)
        conf["model_path"] = str(local)

    def _warmup(self) -> None:
        from encode import quality_flags

        result = self.generate(GenerationRequest(WARMUP_PROMPT, WARMUP_SECONDS, WARMUP_SEED, "wav"))
        flags = quality_flags(result.samples)
        log_event("warmup_done", backend="pytorch", seconds=round(result.inference_s, 4),
                  quality_flags=flags)
        if "static_suspect" in flags or "near_silent" in flags:
            raise WorkerFailure("runtime_mismatch", "warm-up generation looks like static or silence")

    def count_prompt_tokens(self, prompt: str) -> int:
        # Same tokenizer object and special-token behaviour the conditioner uses; no
        # truncation so an overlong prompt is counted in full.
        return len(self._tokenizer(prompt)["input_ids"])

    def generate(self, request: GenerationRequest) -> AudioResult:
        torch = self._torch
        if self._model is None:
            raise WorkerFailure("internal", "backend used before load")
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            audio = self._model.generate(
                prompt=request.prompt, negative_prompt=None, duration=request.duration_seconds,
                steps=STEPS, cfg_scale=CFG_SCALE, batch_size=1, seed=int(request.seed),
                sample_size=self._sample_size)
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            raise WorkerFailure("oom", "GPU ran out of memory during generation", retryable=False) from None
        except WorkerFailure:
            raise
        except Exception:
            raise WorkerFailure("deterministic_runtime", "upstream generation raised an error",
                                retryable=False) from None
        inference_s = time.perf_counter() - start
        samples = audio[0].detach().to("cpu", dtype=torch.float32).numpy()
        if samples.shape[0] != CHANNELS:
            raise WorkerFailure("invalid_output", "generated audio is not stereo", retryable=False)
        return AudioResult(samples=samples, sample_rate_hz=SAMPLE_RATE_HZ, channels=CHANNELS,
                           seed=int(request.seed), inference_s=inference_s, decode_s=None,
                           peak_vram_bytes=int(torch.cuda.max_memory_allocated()))

    def close(self) -> None:
        self._model = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
