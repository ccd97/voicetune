"""Translation pipeline — uses Claude via Bedrock gateway to translate diarized turns."""

import json
import logging
import os
from pathlib import Path

from voicetune import prompts

log = logging.getLogger(__name__)

ENGLISH_CODES = {"en-US", "en-GB", "en-AU", "en-IN", "en"}


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


def translate_batch(http_client, base_url: str, auth_token: str, model: str, turns: list[dict], source_lang: str) -> list[str]:
    """Translate multiple turns in a single LLM call for efficiency."""
    numbered_turns = "\n".join(f"{i+1}. {t['text']}" for i, t in enumerate(turns))
    prompt = prompts.render("translate.j2", source_lang=source_lang, numbered_turns=numbered_turns)

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

    response_text = result["content"][0]["text"]

    # Parse numbered responses
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


def process_file(input_path: Path, output_dir: Path) -> dict:
    """Read a diarized JSON, add English translations via Claude, write to output."""
    import httpx

    base_url = os.environ["ANTHROPIC_BEDROCK_BASE_URL"]
    auth_token = os.environ["ANTHROPIC_AUTH_TOKEN"]
    model = os.environ.get("TRANSLATE_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
    ca_certs = os.environ.get("NODE_EXTRA_CA_CERTS", True)
    http_client = httpx.Client(verify=ca_certs)

    with open(input_path) as f:
        data = json.load(f)

    call_id = data["call_id"]
    source_lang_raw = data.get("language", "unknown")

    log.info(f"Processing {call_id}: language={source_lang_raw}, model={model}")

    # Split turns into those needing translation and those that don't
    turns_to_translate = []
    translate_indices = []

    for i, turn in enumerate(data["turns"]):
        if needs_translation(turn["text"], source_lang_raw):
            turns_to_translate.append(turn)
            translate_indices.append(i)

    # Build result with text_en for all turns
    translated_turns = []
    for turn in data["turns"]:
        new_turn = dict(turn)
        new_turn["language"] = source_lang_raw
        new_turn["text_en"] = turn["text"]  # default: keep original
        translated_turns.append(new_turn)

    # Batch translate non-English turns
    if turns_to_translate:
        batch_size = 20
        for batch_start in range(0, len(turns_to_translate), batch_size):
            batch = turns_to_translate[batch_start:batch_start + batch_size]
            batch_indices = translate_indices[batch_start:batch_start + batch_size]

            log.info(f"  Translating batch {batch_start//batch_size + 1} ({len(batch)} turns)...")
            translations = translate_batch(http_client, base_url, auth_token, model, batch, source_lang_raw)

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
