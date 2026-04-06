"""Scrub pipeline — uses llama-cpp-python to detect and discard turns containing sensitive data."""

import logging
from enum import Enum
from pathlib import Path

from voicetune import prompts
from voicetune.common import generate_text, load_llm, read_json, write_json

log = logging.getLogger(__name__)

BATCH_SIZE = 20


class SensitiveDataType(Enum):
    """Categories of sensitive data to detect. Edit this enum to change what gets flagged."""
    PERSON_NAME = "person name"
    PHONE_NUMBER = "phone number"
    ACCOUNT_NUMBER = "account or ID number"
    ADDRESS = "physical address"
    EMAIL = "email address"
    DATE_OF_BIRTH = "date of birth"
    FINANCIAL = "financial info (card number, bank account, balance)"
    PERSONAL_INFO = "personal info (marital status, health conditions, family details, political opinions, religious beliefs, sexual orientation)"
    CONFIDENTIAL = "confidential business info (internal policies, trade secrets, proprietary data)"
    GOVERNMENT_ID = "government ID (SSN, Aadhaar, passport number, driver's license)"
    CREDENTIALS = "credentials (passwords, PINs, security questions and answers)"
    LOCATION = "real-time location or specific whereabouts"


SENSITIVE_TYPES_LIST = ", ".join(t.value for t in SensitiveDataType)


def build_prompt(turns: list[dict]) -> str:
    numbered_turns = "\n".join(f"{i+1}. [{t['speaker']}] {t['text']}" for i, t in enumerate(turns))
    return prompts.render("scrub.j2", sensitive_types=SENSITIVE_TYPES_LIST, numbered_turns=numbered_turns)


def classify_batch(llm, turns: list[dict]) -> list[bool]:
    """Send a batch of turns to the local LLM. Returns list of bools: True = sensitive (discard)."""
    prompt = build_prompt(turns)

    response_text = generate_text(llm, prompt, max_tokens=len(turns) * 20)

    flags = [False] * len(turns)
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split(".", 1)
        if len(parts) == 2 and parts[0].strip().isdigit():
            idx = int(parts[0].strip()) - 1
            verdict = parts[1].strip().upper()
            if 0 <= idx < len(turns):
                flags[idx] = "SENSITIVE" in verdict

    return flags


def process_file(input_path: Path, output_dir: Path) -> dict:
    """Read a diarized JSON, classify turns via llama.cpp, discard sensitive ones, write output."""
    llm = load_llm(n_ctx=4096)

    data = read_json(input_path)

    call_id = data["call_id"]
    turns = data["turns"]
    log.info(f"Processing {call_id}: {len(turns)} turns")

    sensitive_flags = [False] * len(turns)

    for batch_start in range(0, len(turns), BATCH_SIZE):
        batch = turns[batch_start:batch_start + BATCH_SIZE]
        log.info(f"  Classifying batch {batch_start // BATCH_SIZE + 1} ({len(batch)} turns)...")
        batch_flags = classify_batch(llm, batch)
        for i, flag in enumerate(batch_flags):
            sensitive_flags[batch_start + i] = flag

    clean_turns = [t for t, is_sensitive in zip(turns, sensitive_flags) if not is_sensitive]
    discarded = len(turns) - len(clean_turns)

    log.info(f"  Kept {len(clean_turns)} turns, discarded {discarded} with sensitive data")

    result = {
        "call_id": call_id,
        "mode": data.get("mode", "unknown"),
        "language": data.get("language", "unknown"),
        "num_speakers": data.get("num_speakers"),
        "turns": clean_turns,
    }

    out_path = output_dir / input_path.name
    write_json(out_path, result)

    log.info(f"Saved {len(clean_turns)} turns to {out_path}")
    return result
