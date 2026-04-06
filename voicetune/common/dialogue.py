"""Dialogue / turn transformations shared across stages."""


def unique_speakers(turns: list[dict]) -> list[str]:
    """Return the sorted set of distinct speaker labels in `turns`."""
    return sorted({t["speaker"] for t in turns})


def turn_text_chars(turns: list[dict]) -> int:
    """Total character count across all turns' `text` fields."""
    return sum(len(t.get("text", "")) for t in turns)


def turn_duration(turn: dict) -> float:
    """Duration in seconds; prefers `duration`, falls back to `end - start`, clamped to >= 0."""
    raw = turn.get("duration", turn.get("end", 0) - turn.get("start", 0))
    return max(0.0, float(raw))


def turns_total_duration(turns: list[dict]) -> float:
    """Span from the earliest `start` to the latest `end` across `turns`. Zero for empty input."""
    if not turns:
        return 0.0
    ends = [t.get("end", 0) for t in turns]
    starts = [t.get("start", 0) for t in turns]
    return float(max(ends) - min(starts))


def speaker_samples(
    turns: list[dict],
    k: int = 3,
    text_max: int = 80,
) -> dict[str, list[str]]:
    """Group turns by speaker; keep at most `k` text samples (first-seen order, truncated)."""
    samples: dict[str, list[str]] = {}
    for turn in turns:
        bucket = samples.setdefault(turn["speaker"], [])
        if len(bucket) < k:
            bucket.append(turn["text"][:text_max])
    return samples


def merge_segments_to_turns(segments: list[dict]) -> list[dict]:
    """Merge consecutive same-speaker segments into turns."""
    turns = []
    current_speaker = None
    current_texts: list[str] = []
    current_start = None
    current_end = None

    for seg in segments:
        speaker = seg["speaker"]
        start = seg["start"]
        end = seg["end"]
        text = seg["text"].strip()

        if not text:
            continue

        if speaker != current_speaker and current_texts:
            turns.append({
                "speaker": current_speaker,
                "start": round(current_start, 2),
                "end": round(current_end, 2),
                "text": " ".join(current_texts),
            })
            current_texts = []
            current_start = None

        current_speaker = speaker
        if current_start is None:
            current_start = start
        current_end = end
        current_texts.append(text)

    if current_texts:
        turns.append({
            "speaker": current_speaker,
            "start": round(current_start, 2),
            "end": round(current_end, 2),
            "text": " ".join(current_texts),
        })

    return turns
