# Step 9: Infer (Local Gradio UI)

## Purpose

Local Gradio web app that loads VoxCPM2 + the step-8 LoRA adapter for A/B testing, reference-clip uploads, and prompt iteration. Runs on the local machine; no GCP VM.

## Module

`voicetune/stages/infer/` — run via `python -m voicetune.stages.infer`

## CLI Args

| Flag | Default | Description |
|------|---------|-------------|
| `--run-dir` | `./output` | Base output directory |
| `--lora-dir` | `<run-dir>/finetune/voxcpm2-lora/latest` | LoRA adapter dir to load |
| `--sample-dir` | `<run-dir>/voxcpm/data/me` | Dir of reference WAVs for the dropdown |
| `--base-model` | `openbmb/VoxCPM2` | HF repo or local path for the base model |
| `--base-only` | `False` | Skip LoRA loading (useful for sanity-checking the base) |
| `--port` | `7860` | Gradio port |
| `--share` | `False` | Expose via gradio.live share tunnel |

## Installation

VoxCPM + Gradio are heavy deps (pull in torch, transformers, librosa, ~2 GB total). They're isolated behind the `[infer]` extra:

```bash
pip install -e '.[infer]'
```

First launch downloads `openbmb/VoxCPM2` weights (~8 GB) into `~/.cache/huggingface/`.

## Running

```bash
python -m voicetune.stages.infer                       # default: latest LoRA + sample clips
python -m voicetune.stages.infer --base-only           # sanity check without LoRA
python -m voicetune.stages.infer --lora-dir .../step_0000500
python -m voicetune.stages.infer --share               # public gradio.live URL
```

Opens at `http://127.0.0.1:7860`. Ctrl+C to stop.

## Device Handling

The app respects whatever PyTorch picks for the current machine:

- CUDA (Linux/Windows): RTF ~0.3 on RTX 4090, ~1–2 on consumer GPUs
- MPS (Apple Silicon): 10–30 s per short utterance on an M-series chip; requires `PYTORCH_ENABLE_MPS_FALLBACK=1` (set in `app.py` before `torch` imports so MPS-less ops like RoPE complex don't crash)
- CPU: minutes per utterance

## UI Features

- Text input (multiline)
- Reference audio: upload, mic, or pick from a dropdown of training `me` clips
- Reference transcript (optional — enables "ultimate cloning" mode per VoxCPM docs)
- LoRA on/off toggle (disabled if `--base-only` or LoRA dir missing)
- CFG slider (1.0–5.0, default 2.0)
- Inference timesteps slider (4–30, default 10)
- Output audio + status line showing clip length and which model generated it

## Tips

- Use a clip from `output/voxcpm/data/me/` as reference audio — same distribution as training.
- For "ultimate cloning" mode, paste the exact transcript of the reference clip into the transcript box.
- Toggle the LoRA checkbox between generations on the same reference clip to A/B base vs LoRA.
- A `step_0000001/` test-mode adapter is indistinguishable from base; don't evaluate until you have a real 500-step run.

## Notes / Limitations

- **No cloud mode**: this stage is local only. The step 8 finetune runs on GCP; inference runs on your mac/linux box. If you need remote inference (e.g. no local GPU), you'd adapt VoxCPM's own `lora_ft_webui.py` or ship the LoRA to a GPU VM separately.
- **Python 3.12 warning**: VoxCPM officially supports Python 3.10–3.12 for inference. Python 3.13 is not supported.
- **Dependency conflicts**: if `pip install -e '.[infer]'` fights with existing pins, create a dedicated venv (`python -m venv .venv_infer; .venv_infer/bin/pip install -e '.[infer]'`) and run the infer stage from there.
