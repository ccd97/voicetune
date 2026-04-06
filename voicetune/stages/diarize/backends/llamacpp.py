"""llama.cpp multimodal diarization backend (Gemma 4 audio). Needs `LLAMACPP_MODEL_PATH`, `LLAMACPP_MMPROJ_PATH`."""

import logging
import os
import re
from pathlib import Path

from voicetune import prompts
from voicetune.common import audio_to_data_uri, extract_json, load_llm
from voicetune.stages.preprocess.paths import get_call_id

log = logging.getLogger(__name__)


def _parse_turns(text: str) -> list[dict]:
    """Parse the model's response into turns. Tries JSON first, falls back to regex."""
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    try:
        parsed = extract_json(text, strip_fences=False)
        if isinstance(parsed, list):
            turns = []
            for item in parsed:
                speaker = str(item.get("speaker", "UNKNOWN"))
                t = item.get("text", "")
                if t:
                    turns.append({"speaker": speaker, "start": 0.0, "end": 0.0, "text": t.strip()})
            if turns:
                return turns
    except ValueError:
        pass

    pattern = re.compile(r"^(Speaker[_ ]?\d+)\s*:\s*(.+)", re.MULTILINE)
    matches = pattern.findall(text)
    if matches:
        return [
            {"speaker": spk.replace(" ", "_"), "start": 0.0, "end": 0.0, "text": txt.strip()}
            for spk, txt in matches
        ]

    if text:
        return [{"speaker": "Speaker_1", "start": 0.0, "end": 0.0, "text": text}]

    return []


def diarize(audio_path: Path, num_speakers: int | None = None, language: str | None = None) -> dict:
    llm = load_llm(n_ctx=8192, mmproj_path=os.environ["LLAMACPP_MMPROJ_PATH"])

    log.info(f"Transcribing + diarizing with llama.cpp: {audio_path.name}")
    audio_uri = audio_to_data_uri(audio_path)

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

    return output
