"""Claude-based speaker correction.

Takes translated AWS transcripts and uses Claude to correct speaker
assignments using conversational context. AWS diarization often
misattributes turns — Claude can use dialogue flow, names, and
context to fix these errors.
"""

import json
import logging
import os
from enum import Enum
from pathlib import Path

from voicetune import prompts

log = logging.getLogger(__name__)


class RejectReason(str, Enum):
    """Quality issues reported by the LLM during correction."""
    IMPROPER_DIARIZATION = "improper_diarization"
    INCORRECT_SPEAKER_ASSIGNMENT = "incorrect_speaker_assignment"
    INCORRECT_SPEAKER_COUNT = "incorrect_speaker_count"
    GARBLED_TRANSCRIPT = "garbled_transcript"
    NONSENSICAL_CONVERSATION = "nonsensical_conversation"
    LANGUAGE_MISMATCH = "language_mismatch"


BATCH_SIZE = 80
CONTEXT_OVERLAP = 10

MIN_CONFIDENCE = 0.60
MIN_TURNS = 4
MIN_SPEAKERS = 2
MAX_SPEAKERS = 4
MAX_PARSE_RETRIES = 2


def build_prompt(turns: list[dict], offset: int = 0, context: list[dict] | None = None) -> str:
    """Build a prompt for Claude to re-assign speaker labels.

    Args:
        turns: The turns to correct (indices in the response will be offset-based).
        offset: Global index offset for turn numbering.
        context: Previous turns (already corrected) included for continuity but NOT in the response.
    """
    parts = []

    if context:
        context_lines = []
        for i, turn in enumerate(context):
            text = turn.get("text_en", turn["text"])
            ctx_idx = offset - len(context) + i
            context_lines.append(
                f"[{ctx_idx}] {turn['speaker']} ({turn['start']:.1f}s - {turn['end']:.1f}s): {text}"
            )
        parts.append("PREVIOUS CONTEXT (already corrected — do NOT include in your response):\n" + "\n".join(context_lines))

    transcript_lines = []
    for i, turn in enumerate(turns):
        text = turn.get("text_en", turn["text"])
        global_idx = offset + i
        transcript_lines.append(
            f"[{global_idx}] {turn['speaker']} ({turn['start']:.1f}s - {turn['end']:.1f}s): {text}"
        )
    parts.append("TURNS TO CORRECT:\n" + "\n".join(transcript_lines))

    transcript_section = "\n\n".join(parts)

    return prompts.render(
        "correction.j2",
        transcript_section=transcript_section,
        offset=offset,
        end_index=offset + len(turns) - 1,
    )


def parse_response(response_text: str, num_turns: int) -> tuple[list[dict], float, list[RejectReason]]:
    """Parse Claude's response into speaker assignments, confidence, and issues."""
    text = response_text.strip()
    valid_reasons = {r.value for r in RejectReason}

    # Try parsing as a JSON object with confidence + assignments + issues
    obj_start = text.find("{")
    obj_end = text.rfind("}")
    if obj_start != -1 and obj_end != -1:
        parsed = json.loads(text[obj_start:obj_end + 1])
        if "assignments" in parsed:
            confidence = float(parsed.get("confidence", -1.0))
            assignments = parsed["assignments"]
            raw_issues = parsed.get("issues", [])
            issues = [RejectReason(i) for i in raw_issues if i in valid_reasons]
            if len(assignments) != num_turns:
                log.warning(
                    f"Expected {num_turns} assignments, got {len(assignments)}. "
                    "Using original labels for missing turns."
                )
            return assignments, confidence, issues

    # Fallback: bare JSON array (no confidence available)
    arr_start = text.find("[")
    arr_end = text.rfind("]")
    if arr_start == -1 or arr_end == -1:
        raise ValueError("No JSON found in response")

    assignments = json.loads(text[arr_start:arr_end + 1])
    if len(assignments) != num_turns:
        log.warning(
            f"Expected {num_turns} assignments, got {len(assignments)}. "
            "Using original labels for missing turns."
        )
    return assignments, -1.0, []


def _call_llm(http_client, base_url: str, auth_token: str, model: str, prompt: str, max_tokens: int = 8192) -> str:
    response = http_client.post(
        f"{base_url}/model/{model}/invoke",
        headers={
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
        },
        json={
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["content"][0]["text"]


def process_file(input_path: Path, output_dir: Path) -> dict:
    """Correct speaker assignments in a translated transcript using Claude."""
    import httpx

    base_url = os.environ["ANTHROPIC_BEDROCK_BASE_URL"]
    auth_token = os.environ["ANTHROPIC_AUTH_TOKEN"]
    model = os.environ.get("CORRECTION_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
    ca_certs = os.environ.get("NODE_EXTRA_CA_CERTS", True)
    http_client = httpx.Client(verify=ca_certs)

    with open(input_path) as f:
        data = json.load(f)

    call_id = data["call_id"]
    turns = data["turns"]
    num_batches = (len(turns) + BATCH_SIZE - 1) // BATCH_SIZE

    log.info(f"Correcting speakers for {call_id}: {len(turns)} turns, model={model}"
             + (f", {num_batches} batches" if num_batches > 1 else ""))

    all_assignments = []
    batch_confidences = []
    all_issues: list[RejectReason] = []
    corrected_so_far = []

    for batch_idx in range(num_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(turns))
        batch = turns[start:end]

        context = corrected_so_far[-CONTEXT_OVERLAP:] if corrected_so_far else None
        prompt = build_prompt(batch, offset=start, context=context)

        if num_batches > 1:
            log.info(f"  Batch {batch_idx + 1}/{num_batches} (turns {start}-{end - 1})")

        for attempt in range(1 + MAX_PARSE_RETRIES):
            response_text = _call_llm(http_client, base_url, auth_token, model, prompt)
            try:
                batch_assignments, confidence, issues = parse_response(response_text, len(batch))
                break
            except (json.JSONDecodeError, ValueError) as e:
                if attempt < MAX_PARSE_RETRIES:
                    log.warning(f"  Invalid JSON from Claude (attempt {attempt + 1}), retrying: {e}")
                else:
                    raise
        all_assignments.extend(batch_assignments)
        if confidence >= 0:
            batch_confidences.append(confidence)
        all_issues.extend(issues)

        # Build corrected versions of this batch for context in the next batch
        batch_map = {a["index"]: a for a in batch_assignments}
        for i, turn in enumerate(batch):
            global_idx = start + i
            if global_idx in batch_map:
                corrected = dict(turn)
                corrected["speaker"] = batch_map[global_idx]["speaker"]
                corrected_so_far.append(corrected)
            else:
                corrected_so_far.append(turn)

    assignment_map = {a["index"]: a for a in all_assignments}
    corrected_turns = []
    changes = 0
    speaker_names = {}

    for i, turn in enumerate(turns):
        new_turn = dict(turn)

        if i in assignment_map:
            a = assignment_map[i]
            new_speaker = a["speaker"]
            name = a.get("name")

            if new_speaker != turn["speaker"]:
                log.debug(
                    f"  Turn {i}: {turn['speaker']} -> {new_speaker} "
                    f"({a.get('reasoning', 'no reason')})"
                )
                changes += 1

            new_turn["speaker"] = new_speaker
            if name:
                speaker_names[new_speaker] = name

        corrected_turns.append(new_turn)

    speakers = sorted(set(t["speaker"] for t in corrected_turns))

    # Quality gate
    overall_confidence = min(batch_confidences) if batch_confidences else -1.0
    if overall_confidence >= 0:
        log.info(f"  Confidence: {overall_confidence:.2f}")

    # LLM-reported issues
    llm_reasons = list(dict.fromkeys(all_issues))  # dedupe, preserve order
    if llm_reasons:
        log.warning(f"  LLM issues: {', '.join(r.value for r in llm_reasons)}")

    # Deterministic checks
    reject = bool(llm_reasons) or (0 <= overall_confidence < MIN_CONFIDENCE)
    if len(corrected_turns) < MIN_TURNS:
        reject = True
        log.warning(f"  Too few turns: {len(corrected_turns)}")
    if len(speakers) < MIN_SPEAKERS:
        reject = True
        log.warning(f"  Mono speaker")
    if len(speakers) > MAX_SPEAKERS:
        reject = True
        log.warning(f"  Too many speakers: {len(speakers)}")

    if reject:
        log.warning(f"  REJECTED {call_id}")
        return {
            "rejected": True,
            "call_id": call_id,
            "reasons": [r.value for r in llm_reasons],
            "confidence": overall_confidence,
            "num_turns": len(corrected_turns),
            "num_speakers": len(speakers),
        }

    # Build output — compatible with segment pipeline input format
    output = {
        "call_id": call_id,
        "mode": data.get("mode", "unknown"),
        "language": data.get("language", "unknown"),
        "speaker_names": speaker_names,
        "correction_confidence": overall_confidence,
        "turns": corrected_turns,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{call_id}_corrected.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    log.info(
        f"  Done: {changes} changes, {len(speakers)} speakers "
        f"({', '.join(f'{s}={speaker_names.get(s, '?')}' for s in speakers)})"
    )
    log.info(f"  Saved to {out_path}")

    return output
