# Step 9: Finetune (Fish Speech S2 Pro LoRA)

## Purpose

Orchestrate Fish Speech S2 Pro LoRA fine-tuning on the exported `.wav` + `.lab` training data. Runs 5 sub-steps inside the fish-speech repo: download model, extract semantic tokens, build protobuf dataset, LoRA train, merge weights.

## Module

`voicetune/finetune/` — run via `python -m voicetune.finetune --fish-speech-dir /path/to/fish-speech`

## CLI Args

| Flag | Default | Description |
|------|---------|-------------|
| `--fish-speech-dir` | (required) | Path to cloned fish-speech repo |
| `--data-dir` | `./output/fish-speech/data` | Exported training data |
| `--project` | `my-voice` | Training project name |
| `--lora-rank` | `8` | LoRA rank |
| `--lora-alpha` | `16` | LoRA alpha |
| `--batch-size` | `16` | VQ extraction batch size |
| `--num-workers` | `1` | VQ extraction workers |
| `--dataset-workers` | `16` | Dataset build workers |
| `--max-steps` | `None` | Training steps (Fish Speech default if omitted) |
| `--skip-download` | `False` | Skip model download |
| `--skip-merge` | `False` | Skip LoRA merge step |
| `--checkpoint-step` | `None` | Specific checkpoint for merge (auto-detect latest if omitted) |
| `--output-dir` | `./output/finetune` | Where to write summary JSON |

## What It Does

### Sub-step 1: Data Link
Symlinks `{fish_speech_dir}/data` → `{data_dir}` (absolute path). Errors if `data/` exists as a real directory. Reuses existing correct symlink.

### Sub-step 2: Download Model
```
huggingface-cli download fishaudio/s2-pro --local-dir checkpoints/s2-pro
```
Skipped if `checkpoints/s2-pro/codec.pth` already exists.

### Sub-step 3: Extract Semantic Tokens
```
python tools/vqgan/extract_vq.py data \
    --num-workers 1 --batch-size 16 \
    --config-name modded_dac_vq \
    --checkpoint-path checkpoints/s2-pro/codec.pth
```
Skipped if `.npy` count >= `.wav` count in `data/me/`.

### Sub-step 4: Build Protobuf Dataset
```
python tools/llama/build_dataset.py \
    --input data --output data/protos \
    --text-extension .lab --num-workers 16
```
Skipped if `data/protos/` already has files.

### Sub-step 5: LoRA Training
```
python fish_speech/train.py --config-name text2semantic_finetune \
    project=my-voice \
    model.pretrained_checkpoint=checkpoints/s2-pro \
    +lora@model.model.lora_config=r_8_alpha_16
```
Appends `trainer.max_steps={N}` if `--max-steps` provided. Long-running (no timeout).

### Sub-step 6: Merge LoRA Weights
```
python tools/llama/merge_lora.py \
    --lora-config r_8_alpha_16 \
    --base-weight checkpoints/s2-pro \
    --lora-weight results/my-voice/checkpoints/step_XXXXX.ckpt \
    --output checkpoints/s2-pro-finetuned/
```
Auto-detects latest checkpoint by sorting `step_*.ckpt` numerically. Skippable via `--skip-merge`.

## Input/Output

**Input:** `output/fish-speech/data/me/*.wav` + `*.lab` (from export step)

**Output:**
- `output/finetune/finetune_summary.json` — timing and status for each sub-step
- Inside fish-speech repo: `checkpoints/s2-pro-finetuned/` — merged model weights

## Key Implementation Details

- All commands run with `cwd=fish_speech_dir` — Fish Speech scripts expect to be run from repo root
- Uses `sys.executable` for subprocess calls — user must activate the correct Python environment (with PyTorch, CUDA, etc.)
- Symlink for data avoids duplicating audio files
- Each sub-step is idempotent — checks for existing output before running
- Checkpoint auto-detection uses regex on `step_(\d+)` pattern, takes highest number
- GPU required: 24GB+ VRAM (A100/L40S recommended)

## Training Tips (from finetune-plan.md)

- Start with low LoRA rank (r=8) — S2 Pro is RL-trained, aggressive fine-tuning degrades quality
- 500-2000 steps usually enough for voice adaptation
- Monitor validation loss — divergence means reduce LR or LoRA rank
- Use reference audio prompts at inference for timbre consistency

## Dependencies

None added to this project — all heavy deps (PyTorch, fish-speech) are in the fish-speech repo's environment.
