"""Claude-based speaker correction.

Takes translated AWS transcripts and uses Claude to correct speaker
assignments using conversational context. AWS diarization often
misattributes turns — Claude can use dialogue flow, names, and
context to fix these errors.
"""

import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


def build_prompt(turns: list[dict]) -> str:
    """Build a prompt for Claude to re-assign speaker labels."""
    transcript_lines = []
    for i, turn in enumerate(turns):
        text = turn.get("text_en", turn["text"])
        transcript_lines.append(
            f"[{i}] {turn['speaker']} ({turn['start']:.1f}s - {turn['end']:.1f}s): {text}"
        )

    transcript = "\n".join(transcript_lines)

    return f"""You are analyzing a phone call transcript that was automatically diarized by AWS Transcribe.
The speaker labels (spk_0, spk_1, spk_2, etc.) may have errors — the same person might be
split across multiple labels, or different people might share a label.

Your task: review the conversation and correct the speaker assignments based on:
1. Conversational flow (who responds to whom, topic continuity)
2. Names mentioned (people referring to each other by name)
3. Consistency of speaking style and topic knowledge
4. Turn-taking patterns

Here is the transcript with turn indices:

{transcript}

Return a JSON array where each element corresponds to a turn (by index), with the corrected
speaker label. Use the format:

[
  {{"index": 0, "speaker": "spk_0", "name": "...", "reasoning": "..."}},
  ...
]

Rules:
- Keep the original spk_N labels where they are correct
- If you identify a speaker by name, include it in the "name" field (otherwise null)
- Use "reasoning" to briefly explain your assignment
- If a turn is clearly misattributed, reassign it
- The number of distinct speakers should reflect the actual conversation participants
- Return ONLY the JSON array, no other text"""


def parse_response(response_text: str, num_turns: int) -> list[dict]:
    """Parse Claude's response into speaker assignments."""
    text = response_text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("No JSON array found in response")

    assignments = json.loads(text[start:end + 1])

    if len(assignments) != num_turns:
        log.warning(
            f"Expected {num_turns} assignments, got {len(assignments)}. "
            "Using original labels for missing turns."
        )

    return assignments


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

    log.info(f"Correcting speakers for {call_id}: {len(turns)} turns, model={model}")

    prompt = build_prompt(turns)

    response = http_client.post(
        f"{base_url}/model/{model}/invoke",
        headers={
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
        },
        json={
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=120,
    )
    response.raise_for_status()
    result = response.json()
    response_text = result["content"][0]["text"]

    assignments = parse_response(response_text, len(turns))

    # Build assignment lookup by index
    assignment_map = {a["index"]: a for a in assignments}

    # Apply corrected labels
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
                log.info(
                    f"  Turn {i}: {turn['speaker']} -> {new_speaker} "
                    f"({a.get('reasoning', 'no reason')})"
                )
                changes += 1

            new_turn["speaker"] = new_speaker
            if name:
                speaker_names[new_speaker] = name

        corrected_turns.append(new_turn)

    # Build output — compatible with segment pipeline input format
    output = {
        "call_id": call_id,
        "mode": data.get("mode", "unknown"),
        "language": data.get("language", "unknown"),
        "speaker_names": speaker_names,
        "turns": corrected_turns,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{call_id}_corrected.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    speakers = sorted(set(t["speaker"] for t in corrected_turns))
    log.info(
        f"  Done: {changes} changes, {len(speakers)} speakers "
        f"({', '.join(f'{s}={speaker_names.get(s, '?')}' for s in speakers)})"
    )
    log.info(f"  Saved to {out_path}")

    return output
