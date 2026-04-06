"""Path conventions for segment output (`{call_id}_turn_NNN.wav`)."""

import re

_TURN_WAV_RE = re.compile(r"^(?P<call_id>.+)_turn_(?P<turn>\d{3})\.wav$")


def turn_wav_name(call_id: str, turn_number: int) -> str:
    """Canonical per-turn WAV filename for a call: `{call_id}_turn_{NNN}.wav`."""
    return f"{call_id}_turn_{turn_number:03d}.wav"


def parse_turn_wav_name(name: str) -> tuple[str, int] | None:
    """Inverse of `turn_wav_name`. Returns `(call_id, turn_number)` or `None` if the name doesn't match."""
    m = _TURN_WAV_RE.match(name)
    if not m:
        return None
    return m.group("call_id"), int(m.group("turn"))
