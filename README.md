# stable-income-worker

The GPU worker for a small, independently operated Stable Audio 3 Medium hosting service on
RunPod Serverless. This repository is a **published subset** of a private project: it holds
only the worker code, its message contracts, and the image build, so the public container
images can be built and inspected. It is exported from the private repository by a script;
changes made here directly are overwritten.

- Images: `ghcr.io/jaddai0/stable-income-worker-pytorch`, `ghcr.io/jaddai0/stable-income-worker-tensorrt`
- **No model weights or engines are in this repository or the images.** Workers load the
  gated Stable Audio 3 Medium weights (pinned revision) from RunPod's model cache at start and
  refuse to run on any other revision.
- The worker only talks to its own gateway; it has no public API of its own.

## Licences and attribution

- Stable Audio 3 Medium is Stability AI's model, used under the
  [Stability AI Community License](https://stability.ai/license). Powered by Stability AI.
  The T5Gemma text encoder shipped with it is under Google's Gemma terms.
- Upstream inference code: [Stability-AI/stable-audio-3](https://github.com/Stability-AI/stable-audio-3) (MIT), installed at a pinned commit.
- MP3 encoding uses `lameenc` (LAME, LGPL-3.0-or-later). FLAC and WAV use `soundfile`
  (libsndfile, LGPL-2.1-or-later; libFLAC, BSD). Their sources are available from those
  projects; the exact versions are pinned in `worker/uv.lock` and `worker/tensorrt/requirements.lock`.
- The project's own code in this repository is not yet released under an open-source licence.

## Tests

```bash
cd worker && uv run --frozen python -B -m pytest -q
```
