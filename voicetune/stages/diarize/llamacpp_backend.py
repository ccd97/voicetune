"""llama.cpp multimodal diarization backend.

Uses llama-cpp-python with Gemma 4's audio conformer encoder for
combined transcription + speaker diarization in a single model pass.
No separate pyannote step needed.

Requires: pip install llama-cpp-python (built with LLAVA_BUILD=ON for mtmd)
Requires: LLAMACPP_MODEL_PATH and LLAMACPP_MMPROJ_PATH env vars.
"""

import base64
import json
import logging
import os
import re
from pathlib import Path

from voicetune import prompts

from .utils import get_call_id, save_result

log = logging.getLogger(__name__)

_llm = None


def _get_llm():
    global _llm
    if _llm is not None:
        return _llm

    from llama_cpp import Llama
    # No Gemma4ChatHandler in llama-cpp-python yet; Gemma3's works for the mmproj path.
    from llama_cpp.llama_chat_format import Gemma3ChatHandler

    model_path = os.environ["LLAMACPP_MODEL_PATH"]
    mmproj_path = os.environ["LLAMACPP_MMPROJ_PATH"]

    log.info(f"Loading llama.cpp model: {model_path}")
    log.info(f"mmproj: {mmproj_path}")

    handler = Gemma3ChatHandler(clip_model_path=mmproj_path, verbose=False)
    _llm = Llama(
        model_path=model_path,
        chat_handler=handler,
        n_ctx=8192,
        n_threads=os.cpu_count() or 4,
        verbose=False,
    )
    return _llm


def _audio_to_data_uri(audio_path: Path) -> str:
    raw = audio_path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


def _parse_turns(text: str) -> list[dict]:
    """Parse the model's response into turns. Tries JSON first, falls back to regex."""
    text = text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            turns = []
            for item in parsed:
                speaker = str(item.get("speaker", "UNKNOWN"))
                t = item.get("text", "")
                if t:
                    turns.append({"speaker": speaker, "start": 0.0, "end": 0.0, "text": t.strip()})
            if turns:
                return turns
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: try to parse "Speaker_N: text" lines
    pattern = re.compile(r"^(Speaker[_ ]?\d+)\s*:\s*(.+)", re.MULTILINE)
    matches = pattern.findall(text)
    if matches:
        return [
            {"speaker": spk.replace(" ", "_"), "start": 0.0, "end": 0.0, "text": txt.strip()}
            for spk, txt in matches
        ]

    # Last resort: entire response as single speaker
    if text:
        return [{"speaker": "Speaker_1", "start": 0.0, "end": 0.0, "text": text}]

    return []


def diarize(audio_path: Path, output_dir: Path, num_speakers: int | None = None, language: str | None = None) -> dict:
    llm = _get_llm()

    log.info(f"Transcribing + diarizing with llama.cpp: {audio_path.name}")
    audio_uri = _audio_to_data_uri(audio_path)

    user_text = prompts.render("diarize_user.j2", num_speakers=num_speakers, language=language)
    user_content = [
        {"type": "image_url", "image_url": {"url": audio_uri}},
        {"type": "text", "text": user_text},
    ]

    messages = [
        {"role": "system", "content": prompts.render("diarize_system.j2")},
        {"role": "user", "content": user_content},
    ]

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=4096,
        stop=["<end_of_turn>", "<eos>"],
        temperature=0.0,
    )

    raw_text = response["choices"][0]["message"]["content"]
    log.info(f"Model response length: {len(raw_text)} chars")

    turns = _parse_turns(raw_text)
    if not turns:
        log.warning("Model produced no parseable turns")

    call_id = get_call_id(audio_path)
    output = {
        "call_id": call_id,
        "mode": "llamacpp",
        "language": language or "auto",
        "turns": turns,
    }

    save_result(output, output_dir)
    return output
