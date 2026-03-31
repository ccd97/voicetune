"""Translation pipeline — translates diarized turns to English via Bedrock or llama.cpp."""

import json
import logging
import os
from pathlib import Path

from voicetune import prompts

log = logging.getLogger(__name__)

ENGLISH_CODES = {"en-US", "en-GB", "en-AU", "en-IN", "en"}
BATCH_SIZE = 20

_llm = None


def _get_llm():
    global _llm
    if _llm is not None:
        return _llm

    from llama_cpp import Llama

    model_path = os.environ["LLAMACPP_MODEL_PATH"]
    log.info(f"Loading llama.cpp model for translate: {model_path}")

    _llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=-1,
        n_threads=os.cpu_count() or 4,
        verbose=False,
    )
    return _llm


def is_latin_text(text: str, threshold: float = 0.7) -> bool:
    """Check if text is predominantly Latin script (i.e., likely English).

    Catches cases where Indian English is misdetected as Hindi/Marathi.
    """
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return True
    latin_count = sum(1 for c in alpha_chars if ord(c) < 0x0250)
    return latin_count / len(alpha_chars) >= threshold


def needs_translation(text: str, language: str) -> bool:
    """Determine if a turn actually needs translation."""
    if language in ENGLISH_CODES:
        return False
    if is_latin_text(text):
        return False
    return True


def _build_prompt(turns: list[dict], source_lang: str) -> str:
    numbered_turns = "\n".join(f"{i+1}. {t['text']}" for i, t in enumerate(turns))
    return prompts.render("translate.j2", source_lang=source_lang, numbered_turns=numbered_turns)


def _parse_translations(response_text: str) -> list[str]:
    translations = []
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split(".", 1)
        if len(parts) == 2 and parts[0].strip().isdigit():
            translations.append(parts[1].strip())
        else:
            translations.append(line)
    return translations


def _translate_batch_bedrock(http_client, base_url: str, auth_token: str, model: str, turns: list[dict], source_lang: str) -> list[str]:
    prompt = _build_prompt(turns, source_lang)

    response = http_client.post(
        f"{base_url}/model/{model}/invoke",
        headers={
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
        },
        json={
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    response.raise_for_status()
    result = response.json()
    return _parse_translations(result["content"][0]["text"])


def _translate_batch_llamacpp(llm, turns: list[dict], source_lang: str) -> list[str]:
    prompt = _build_prompt(turns, source_lang)

    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=4096,
    )
    return _parse_translations(response["choices"][0]["message"]["content"])


def process_file(input_path: Path, output_dir: Path, backend: str = "llamacpp") -> dict:
    """Read a diarized JSON, add English translations, write to output."""
    if backend == "bedrock":
        import httpx
        base_url = os.environ["ANTHROPIC_BEDROCK_BASE_URL"]
        auth_token = os.environ["ANTHROPIC_AUTH_TOKEN"]
        model = os.environ.get("TRANSLATE_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
        ca_certs = os.environ.get("NODE_EXTRA_CA_CERTS", True)
        http_client = httpx.Client(verify=ca_certs)
        translate_fn = lambda turns, lang: _translate_batch_bedrock(http_client, base_url, auth_token, model, turns, lang)
        model_label = model
    else:
        llm = _get_llm()
        translate_fn = lambda turns, lang: _translate_batch_llamacpp(llm, turns, lang)
        model_label = os.environ["LLAMACPP_MODEL_PATH"]

    with open(input_path) as f:
        data = json.load(f)

    call_id = data["call_id"]
    source_lang_raw = data.get("language", "unknown")

    log.info(f"Processing {call_id}: language={source_lang_raw}, backend={backend}, model={model_label}")

    turns_to_translate = []
    translate_indices = []

    for i, turn in enumerate(data["turns"]):
        if needs_translation(turn["text"], source_lang_raw):
            turns_to_translate.append(turn)
            translate_indices.append(i)

    translated_turns = []
    for turn in data["turns"]:
        new_turn = dict(turn)
        new_turn["language"] = source_lang_raw
        new_turn["text_en"] = turn["text"]
        translated_turns.append(new_turn)

    if turns_to_translate:
        for batch_start in range(0, len(turns_to_translate), BATCH_SIZE):
            batch = turns_to_translate[batch_start:batch_start + BATCH_SIZE]
            batch_indices = translate_indices[batch_start:batch_start + BATCH_SIZE]

            log.info(f"  Translating batch {batch_start // BATCH_SIZE + 1} ({len(batch)} turns)...")
            translations = translate_fn(batch, source_lang_raw)

            for idx, translation in zip(batch_indices, translations):
                translated_turns[idx]["text_en"] = translation

        log.info(f"  Translated: {len(turns_to_translate)}, Skipped (already English): {len(data['turns']) - len(turns_to_translate)}")
    else:
        log.info(f"  All {len(data['turns'])} turns already in English, skipped translation")

    result = {
        "call_id": call_id,
        "mode": data.get("mode", "unknown"),
        "language": source_lang_raw,
        "turns": translated_turns,
    }

    out_path = output_dir / f"{call_id}_translated.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    log.info(f"Saved {len(translated_turns)} turns to {out_path}")
    return result
