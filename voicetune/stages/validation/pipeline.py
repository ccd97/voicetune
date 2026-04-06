"""LLM-based speaker validation.

Takes diarized transcripts and uses an LLM to validate speaker
assignments using conversational context. Diarization often
misattributes turns — the LLM uses dialogue flow, names, and
context to fix these errors. Runs on a local llama.cpp model.
"""

import logging
import os
from enum import Enum
from pathlib import Path

from voicetune import prompts
from voicetune.common import (
    extract_json,
    generate_text,
    load_llm,
    read_json,
    unique_speakers,
    write_json,
)

log = logging.getLogger(__name__)


class RejectReason(str, Enum):
    """Quality issues reported by the LLM during validation."""
    # Per-turn issues (reported on individual assignments)
    IMPROPER_DIARIZATION = "improper_diarization"
    INCORRECT_SPEAKER_ASSIGNMENT = "incorrect_speaker_assignment"
    GARBLED_TRANSCRIPT = "garbled_transcript"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    # File-level issues (reported in top-level "issues")
    INCORRECT_SPEAKER_COUNT = "incorrect_speaker_count"
    NONSENSICAL_CONVERSATION = "nonsensical_conversation"
    LANGUAGE_MISMATCH = "language_mismatch"
    # Pre-LLM checks
    LANGUAGE_NOT_ALLOWED = "language_not_allowed"
    MONO_SPEAKER = "mono_speaker"
    # Confidence gate
    LOW_CONFIDENCE = "low_confidence"


TURN_ISSUES = {
    RejectReason.IMPROPER_DIARIZATION,
    RejectReason.INCORRECT_SPEAKER_ASSIGNMENT,
    RejectReason.GARBLED_TRANSCRIPT,
    RejectReason.TOO_SHORT,
    RejectReason.TOO_LONG,
}

FILE_ISSUES = {
    RejectReason.INCORRECT_SPEAKER_COUNT,
    RejectReason.NONSENSICAL_CONVERSATION,
    RejectReason.LANGUAGE_MISMATCH,
}

MIN_TURN_DURATION = 0.5
MAX_TURN_DURATION = 60.0


BATCH_SIZE = 80
CONTEXT_OVERLAP = 10

MIN_CONFIDENCE = 0.60
ALLOWED_LANGUAGES = set(os.environ.get("ALLOWED_LANGS", "en").split(","))
MAX_PARSE_RETRIES = 2


def build_prompt(turns: list[dict], offset: int = 0, context: list[dict] | None = None) -> str:
    """Prompt the LLM to re-assign speaker labels for `turns`; `context` is prior already-validated turns."""
    parts = []

    if context:
        context_lines = []
        for i, turn in enumerate(context):
            text = turn["text"]
            ctx_idx = offset - len(context) + i
            context_lines.append(
                f"[{ctx_idx}] {turn['speaker']} ({turn['start']:.1f}s - {turn['end']:.1f}s): {text}"
            )
        parts.append("PREVIOUS CONTEXT (already validated — do NOT include in your response):\n" + "\n".join(context_lines))

    transcript_lines = []
    for i, turn in enumerate(turns):
        text = turn["text"]
        global_idx = offset + i
        transcript_lines.append(
            f"[{global_idx}] {turn['speaker']} ({turn['start']:.1f}s - {turn['end']:.1f}s): {text}"
        )
    parts.append("TURNS TO CORRECT:\n" + "\n".join(transcript_lines))

    transcript_section = "\n\n".join(parts)

    return prompts.render(
        "validation.j2",
        transcript_section=transcript_section,
        offset=offset,
        end_index=offset + len(turns) - 1,
    )


def parse_response(response_text: str, num_turns: int) -> tuple[list[dict], float, list[RejectReason]]:
    """Parse the LLM's response into speaker assignments, confidence, and issues."""
    parsed = extract_json(response_text, prefer="object", strip_fences=False)

    if isinstance(parsed, dict) and "assignments" in parsed:
        assignments = parsed["assignments"]
        confidence = float(parsed.get("confidence", -1.0))
        valid_reasons = {r.value for r in RejectReason}
        issues = [RejectReason(i) for i in parsed.get("issues", []) if i in valid_reasons]
    elif isinstance(parsed, list):
        assignments = parsed
        confidence = -1.0
        issues = []
    else:
        raise ValueError("No JSON assignments found in response")

    if len(assignments) != num_turns:
        log.warning(
            f"Expected {num_turns} assignments, got {len(assignments)}. "
            "Using original labels for missing turns."
        )
    return assignments, confidence, issues


def process_file(input_path: Path, output_dir: Path) -> dict:
    """Correct speaker assignments in a diarized transcript."""
    data = read_json(input_path)

    call_id = data["call_id"]
    turns = data["turns"]
    language = data.get("language", "unknown")
    num_speakers = data["num_speakers"]

    reject_reasons = []
    if language not in ALLOWED_LANGUAGES:
        reject_reasons.append(RejectReason.LANGUAGE_NOT_ALLOWED.value)
    if num_speakers < 2:
        reject_reasons.append(RejectReason.MONO_SPEAKER.value)

    if reject_reasons:
        log.warning(f"  REJECTED {call_id} ({', '.join(reject_reasons)})")
        output = dict(data)
        output["rejected"] = True
        output["reject_reasons"] = reject_reasons
        output["validation_confidence"] = -1.0

        out_path = output_dir / f"{call_id}_validated.json"
        write_json(out_path, output)

        return {
            "rejected": True,
            "call_id": call_id,
            "reasons": reject_reasons,
            "confidence": -1.0,
            "num_turns": len(turns),
            "num_speakers": num_speakers,
        }

    llm = load_llm(n_ctx=8192)
    model_label = os.environ["LLAMACPP_MODEL_PATH"]

    num_batches = (len(turns) + BATCH_SIZE - 1) // BATCH_SIZE

    log.info(f"Correcting speakers for {call_id}: {len(turns)} turns, model={model_label}"
             + (f", {num_batches} batches" if num_batches > 1 else ""))

    all_assignments = []
    batch_confidences = []
    all_issues: list[RejectReason] = []
    validated_so_far = []

    for batch_idx in range(num_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(turns))
        batch = turns[start:end]

        context = validated_so_far[-CONTEXT_OVERLAP:] if validated_so_far else None
        prompt = build_prompt(batch, offset=start, context=context)

        if num_batches > 1:
            log.info(f"  Batch {batch_idx + 1}/{num_batches} (turns {start}-{end - 1})")

        for attempt in range(1 + MAX_PARSE_RETRIES):
            response_text = generate_text(llm, prompt, max_tokens=8192)
            try:
                batch_assignments, confidence, issues = parse_response(response_text, len(batch))
                break
            except ValueError as e:
                if attempt < MAX_PARSE_RETRIES:
                    log.warning(f"  Invalid JSON from LLM (attempt {attempt + 1}), retrying: {e}")
                else:
                    raise
        all_assignments.extend(batch_assignments)
        if confidence >= 0:
            batch_confidences.append(confidence)
        all_issues.extend(issues)

        batch_map = {a["index"]: a for a in batch_assignments}
        for i, turn in enumerate(batch):
            global_idx = start + i
            if global_idx in batch_map:
                validated = dict(turn)
                validated["speaker"] = batch_map[global_idx]["speaker"]
                validated_so_far.append(validated)
            else:
                validated_so_far.append(turn)

    assignment_map = {a["index"]: a for a in all_assignments}
    validated_turns = []
    changes = 0
    speaker_names = {}

    turn_issue_values = {r.value for r in TURN_ISSUES}
    all_turn_issues: set[str] = set()

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

            turn_issues = [v for v in a.get("issues", []) if v in turn_issue_values]
            if turn_issues:
                new_turn["issues"] = turn_issues
                all_turn_issues.update(turn_issues)

        duration = new_turn["end"] - new_turn["start"]
        timing_issues = []
        if duration < MIN_TURN_DURATION:
            timing_issues.append(RejectReason.TOO_SHORT.value)
        if duration > MAX_TURN_DURATION:
            timing_issues.append(RejectReason.TOO_LONG.value)
        if timing_issues:
            existing = new_turn.get("issues", [])
            new_turn["issues"] = existing + timing_issues
            all_turn_issues.update(timing_issues)

        validated_turns.append(new_turn)

    speakers = unique_speakers(validated_turns)

    # Quality gate
    overall_confidence = min(batch_confidences) if batch_confidences else -1.0
    if overall_confidence >= 0:
        log.info(f"  Confidence: {overall_confidence:.2f}")

    # File-level LLM issues
    llm_reasons = list(dict.fromkeys(all_issues))
    if llm_reasons:
        log.warning(f"  File-level issues: {', '.join(r.value for r in llm_reasons)}")
    if all_turn_issues:
        log.warning(f"  Turn-level issues: {', '.join(sorted(all_turn_issues))}")

    # Only file-level issues and confidence cause rejection; per-turn issues
    # are preserved on individual turns for the filter step to handle.
    file_level_values = {r.value for r in FILE_ISSUES}
    file_reject_reasons = [r.value for r in llm_reasons if r.value in file_level_values]
    if 0 <= overall_confidence < MIN_CONFIDENCE:
        file_reject_reasons.append(RejectReason.LOW_CONFIDENCE.value)
    reject = bool(file_reject_reasons)

    if reject:
        log.warning(f"  REJECTED {call_id}")
        output = dict(data)
        output["rejected"] = True
        output["reject_reasons"] = file_reject_reasons
        output["validation_confidence"] = overall_confidence
        output["turns"] = validated_turns

        out_path = output_dir / f"{call_id}_validated.json"
        write_json(out_path, output)
        log.info(f"  Saved to {out_path}")

        return {
            "rejected": True,
            "call_id": call_id,
            "reasons": file_reject_reasons,
            "confidence": overall_confidence,
            "num_turns": len(validated_turns),
            "num_speakers": len(speakers),
        }

    # Build output — compatible with segment pipeline input format
    output = {
        "call_id": call_id,
        "mode": data.get("mode", "unknown"),
        "language": data.get("language", "unknown"),
        "num_speakers": num_speakers,
        "speaker_names": speaker_names,
        "validation_confidence": overall_confidence,
        "turns": validated_turns,
    }

    out_path = output_dir / f"{call_id}_validated.json"
    write_json(out_path, output)

    speakers_fmt = ", ".join(f"{s}={speaker_names.get(s, '?')}" for s in speakers)
    log.info(
        f"  Done: {changes} changes, {len(speakers)} speakers ({speakers_fmt})"
    )
    log.info(f"  Saved to {out_path}")

    return output
