"""LLM-based speaker validation.

Takes translated transcripts and uses an LLM to validate speaker
assignments using conversational context. Diarization often
misattributes turns — the LLM uses dialogue flow, names, and
context to fix these errors. Supports Bedrock (Claude) and llama.cpp backends.
"""

import json
import logging
import os
from enum import Enum
from pathlib import Path

from voicetune import prompts

log = logging.getLogger(__name__)

_llm = None


def _get_llm():
    global _llm
    if _llm is not None:
        return _llm

    from llama_cpp import Llama

    model_path = os.environ["LLAMACPP_MODEL_PATH"]
    log.info(f"Loading llama.cpp model for validation: {model_path}")

    _llm = Llama(
        model_path=model_path,
        n_ctx=8192,
        n_gpu_layers=-1,
        n_threads=os.cpu_count() or 4,
        verbose=False,
    )
    return _llm


class RejectReason(str, Enum):
    """Quality issues reported by the LLM during validation."""
    # Per-turn issues (reported on individual assignments)
    IMPROPER_DIARIZATION = "improper_diarization"
    INCORRECT_SPEAKER_ASSIGNMENT = "incorrect_speaker_assignment"
    GARBLED_TRANSCRIPT = "garbled_transcript"
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
}


BATCH_SIZE = 80
CONTEXT_OVERLAP = 10

MIN_CONFIDENCE = 0.60
ALLOWED_LANGUAGES = set(os.environ.get("ALLOWED_LANGS", "en").split(","))
MAX_PARSE_RETRIES = 2


def build_prompt(turns: list[dict], offset: int = 0, context: list[dict] | None = None) -> str:
    """Build a prompt for Claude to re-assign speaker labels.

    Args:
        turns: The turns to correct (indices in the response will be offset-based).
        offset: Global index offset for turn numbering.
        context: Previous turns (already validated) included for continuity but NOT in the response.
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
        parts.append("PREVIOUS CONTEXT (already validated — do NOT include in your response):\n" + "\n".join(context_lines))

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
        "validation.j2",
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


def _call_llm_bedrock(http_client, base_url: str, auth_token: str, model: str, prompt: str, max_tokens: int = 8192) -> str:
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


def _call_llm_llamacpp(llm, prompt: str, max_tokens: int = 8192) -> str:
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return response["choices"][0]["message"]["content"]


def process_file(input_path: Path, output_dir: Path, backend: str = "llamacpp") -> dict:
    """Correct speaker assignments in a translated transcript."""
    with open(input_path) as f:
        data = json.load(f)

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

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{call_id}_validated.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        return {
            "rejected": True,
            "call_id": call_id,
            "reasons": reject_reasons,
            "confidence": -1.0,
            "num_turns": len(turns),
            "num_speakers": num_speakers,
        }

    if backend == "bedrock":
        import httpx
        base_url = os.environ["ANTHROPIC_BEDROCK_BASE_URL"]
        auth_token = os.environ["ANTHROPIC_AUTH_TOKEN"]
        model = os.environ.get("VALIDATION_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
        ca_certs = os.environ.get("NODE_EXTRA_CA_CERTS", True)
        http_client = httpx.Client(verify=ca_certs)
        call_fn = lambda prompt, max_tokens=8192: _call_llm_bedrock(http_client, base_url, auth_token, model, prompt, max_tokens)
        model_label = model
    else:
        llm = _get_llm()
        call_fn = lambda prompt, max_tokens=8192: _call_llm_llamacpp(llm, prompt, max_tokens)
        model_label = os.environ["LLAMACPP_MODEL_PATH"]

    num_batches = (len(turns) + BATCH_SIZE - 1) // BATCH_SIZE

    log.info(f"Correcting speakers for {call_id}: {len(turns)} turns, backend={backend}, model={model_label}"
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
            response_text = call_fn(prompt)
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

        validated_turns.append(new_turn)

    speakers = sorted(set(t["speaker"] for t in validated_turns))

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

    all_reject_reasons = [r.value for r in llm_reasons] + sorted(all_turn_issues - {r.value for r in llm_reasons})
    if 0 <= overall_confidence < MIN_CONFIDENCE:
        all_reject_reasons.append(RejectReason.LOW_CONFIDENCE.value)
    reject = bool(all_reject_reasons)

    if reject:
        log.warning(f"  REJECTED {call_id}")
        output = dict(data)
        output["rejected"] = True
        output["reject_reasons"] = all_reject_reasons
        output["validation_confidence"] = overall_confidence
        output["turns"] = validated_turns

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{call_id}_validated.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        log.info(f"  Saved to {out_path}")

        return {
            "rejected": True,
            "call_id": call_id,
            "reasons": all_reject_reasons,
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

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{call_id}_validated.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    log.info(
        f"  Done: {changes} changes, {len(speakers)} speakers "
        f"({', '.join(f'{s}={speaker_names.get(s, '?')}' for s in speakers)})"
    )
    log.info(f"  Saved to {out_path}")

    return output
