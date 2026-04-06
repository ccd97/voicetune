"""Gradio UI for VoxCPM2 + optional LoRA adapter.

Launched by `python -m voicetune.stages.infer`.
"""

import json
import logging
import random
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from voxcpm import VoxCPM

log = logging.getLogger(__name__)

_SEED_RANDOM = -1


def _seed_all(seed: int) -> None:
    """Seed Python/numpy/torch before VoxCPM.generate().

    VoxCPM only draws from the global torch RNG (flow-matching latent + VAE
    NoiseBlock; LM is argmax), so this is enough for same-machine repro --
    except when retry_badcase=True re-draws on a retry.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device_hint() -> str:
    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_lora_config(lora_dir: Path):
    """Read lora_config.json next to the adapter weights and build the right
    LoRAConfig (V1 vs V2) for VoxCPM.from_pretrained. Without this, VoxCPM
    falls back to r=8 alpha=16 defaults, which mismatches any adapter trained
    at a different rank.
    """
    cfg_path = lora_dir / "lora_config.json"
    if not cfg_path.exists():
        log.warning(f"No lora_config.json in {lora_dir} -- letting VoxCPM use defaults (will fail if r!=8)")
        return None

    data = json.loads(cfg_path.read_text())
    fields = data.get("lora_config", data)

    # target_proj_modules including "fusion_concat_proj" is V2-only; detect and
    # pick the matching LoRAConfig class so field defaults line up.
    is_v2 = "fusion_concat_proj" in fields.get("target_proj_modules", [])
    if is_v2:
        from voxcpm.model.voxcpm2 import LoRAConfig
    else:
        from voxcpm.model.voxcpm import LoRAConfig

    allowed = set(LoRAConfig.__dataclass_fields__.keys()) if hasattr(LoRAConfig, "__dataclass_fields__") else None
    kwargs = {k: v for k, v in fields.items() if allowed is None or k in allowed}
    log.info(f"Loaded LoRA config from {cfg_path.name}: r={kwargs.get('r')}, alpha={kwargs.get('alpha')}, v2={is_v2}")
    return LoRAConfig(**kwargs)


def launch(
    base_repo: str,
    lora_dir: Path | None,
    samples_dir: Path | None,
    port: int = 7860,
    share: bool = False,
) -> None:
    log.info(f"Device: {_device_hint()}")

    # Single model instance with set_lora_enabled() toggle; avoids doubling 2B-param RAM/VRAM.
    has_lora = lora_dir is not None and lora_dir.is_dir()
    if has_lora:
        log.info(f"Loading base model {base_repo} with LoRA adapter {lora_dir}")
        model = VoxCPM.from_pretrained(
            base_repo,
            lora_weights_path=str(lora_dir),
            lora_config=_load_lora_config(lora_dir),
            load_denoiser=False,
        )
    else:
        log.info(f"Loading base model: {base_repo} (no LoRA)")
        model = VoxCPM.from_pretrained(base_repo, load_denoiser=False)

    samples: list[str] = []
    if samples_dir and samples_dir.is_dir():
        samples = sorted(str(p) for p in samples_dir.glob("*.wav"))
        log.info(f"Found {len(samples)} reference WAVs in {samples_dir}")

    def generate(text, ref_audio, ref_text, use_lora, cfg_value, timesteps, seed):
        if not text or not text.strip():
            return None, "Enter some text."

        seed_int = int(seed)
        if seed_int == _SEED_RANDOM:
            seed_int = random.randint(0, 2**31 - 1)
        _seed_all(seed_int)

        if has_lora:
            model.set_lora_enabled(bool(use_lora))
        kwargs = {
            "text": text,
            "cfg_value": float(cfg_value),
            "inference_timesteps": int(timesteps),
        }
        if ref_audio:
            ref_text_clean = (ref_text or "").strip()
            if ref_text_clean:
                # "Ultimate cloning" in VoxCPM docs: same clip as both prompt and reference.
                kwargs["prompt_wav_path"] = ref_audio
                kwargs["prompt_text"] = ref_text_clean
                kwargs["reference_wav_path"] = ref_audio
            else:
                kwargs["reference_wav_path"] = ref_audio

        try:
            wav = model.generate(**kwargs)
        except Exception as e:
            log.exception("Generation failed")
            return None, f"Generation failed: `{type(e).__name__}: {e}`"

        sr = model.tts_model.sample_rate
        out = Path(tempfile.mkdtemp()) / "out.wav"
        sf.write(str(out), wav, sr)
        tag = "LoRA" if (has_lora and use_lora) else "Base"
        return str(out), f"Generated {len(wav) / sr:.2f}s from {tag} model (seed={seed_int})."

    header = f"""# VoxCPM2 inference

**Base:** `{base_repo}`  
**LoRA:** {'`' + str(lora_dir) + '`' if has_lora else '_(base-only mode)_'}

Device: `{_device_hint()}`"""

    with gr.Blocks(title="VoxCPM2 + LoRA") as demo:
        gr.Markdown(header)
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Text to synthesize", lines=3)
                ref_audio = gr.Audio(
                    label="Reference audio (optional, for voice cloning)",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                if samples:
                    ref_picker = gr.Dropdown(
                        choices=[("-- none --", "")] + [(Path(p).name, p) for p in samples],
                        value="",
                        label="Reference clip",
                    )
                    ref_picker.change(lambda x: x or None, [ref_picker], [ref_audio])
                ref_text = gr.Textbox(
                    label="Reference transcript (ultimate cloning — exact text of ref clip)",
                    placeholder="Leave blank to use reference audio without transcript.",
                )
                use_lora = gr.Checkbox(
                    label="Use LoRA adapter",
                    value=has_lora,
                    interactive=has_lora,
                )
                with gr.Row():
                    cfg_value = gr.Slider(1.0, 5.0, value=2.0, step=0.1, label="CFG")
                    timesteps = gr.Slider(4, 30, value=10, step=1, label="Inference timesteps")
                seed = gr.Number(
                    value=_SEED_RANDOM,
                    precision=0,
                    label=f"Seed ({_SEED_RANDOM} = random; same seed + same inputs reproduce the take)",
                )
                btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                audio_out = gr.Audio(label="Output", type="filepath")
                status = gr.Markdown()

        btn.click(
            generate,
            inputs=[text, ref_audio, ref_text, use_lora, cfg_value, timesteps, seed],
            outputs=[audio_out, status],
        )

    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=port,
        share=share,
        show_error=True,
    )
