"""AWS Transcribe diarization backend."""

import json
import logging
import os
import time
import urllib.request
import uuid
from pathlib import Path

from voicetune.common import get_call_id

log = logging.getLogger(__name__)


def _join_words(words: list[str]) -> str:
    """Join words, attaching punctuation to the preceding word."""
    if not words:
        return ""
    result = words[0]
    for w in words[1:]:
        if w in ".,!?;:'\")-":
            result += w
        else:
            result += " " + w
    return result


def diarize(audio_path: Path, num_speakers: int | None = None, language: str | None = None) -> dict:
    """Run diarization + transcription via AWS Transcribe.

    Language is auto-detected between English, Hindi, and Marathi unless overridden.
    """
    import boto3

    region = os.environ.get("AWS_REGION", "us-east-1")
    bucket = os.environ["AWS_S3_BUCKET"]

    s3 = boto3.client("s3", region_name=region)
    transcribe = boto3.client("transcribe", region_name=region)

    # Upload to S3
    s3_key = f"diarization-input/{audio_path.stem}_{uuid.uuid4().hex[:8]}.wav"
    log.info(f"Uploading {audio_path.name} to s3://{bucket}/{s3_key}")
    s3.upload_file(str(audio_path), bucket, s3_key)

    # Start transcription job
    job_name = f"diarize-{audio_path.stem}-{uuid.uuid4().hex[:8]}"
    media_uri = f"s3://{bucket}/{s3_key}"
    settings = {"ShowSpeakerLabels": True, "MaxSpeakerLabels": num_speakers or 3}

    log.info(f"Starting transcription job: {job_name} (speakers: {'auto-detect' if num_speakers is None else num_speakers})")

    lang_kwargs = {}
    if language:
        lang_kwargs["LanguageCode"] = language
    else:
        lang_kwargs["IdentifyLanguage"] = True
        lang_kwargs["LanguageOptions"] = ["en-US", "hi-IN", "mr-IN"]

    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": media_uri},
        MediaFormat="wav",
        Settings=settings,
        **lang_kwargs,
    )

    # Poll for completion
    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        job_status = status["TranscriptionJob"]["TranscriptionJobStatus"]

        if job_status == "COMPLETED":
            log.info("Transcription job completed")
            break
        elif job_status == "FAILED":
            reason = status["TranscriptionJob"].get("FailureReason", "Unknown")
            raise RuntimeError(f"Transcription job failed: {reason}")
        else:
            log.info(f"Job status: {job_status} — waiting 15s...")
            time.sleep(15)

    # Download results
    result_url = status["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
    with urllib.request.urlopen(result_url) as resp:
        raw_result = json.loads(resp.read().decode())

    # Clean up S3
    log.info(f"Cleaning up s3://{bucket}/{s3_key}")
    s3.delete_object(Bucket=bucket, Key=s3_key)

    # Parse
    call_id = get_call_id(audio_path)
    turns = _parse_result(raw_result)
    detected_lang = status["TranscriptionJob"].get("LanguageCode", "unknown")
    log.info(f"Language: {detected_lang}")

    result = {
        "call_id": call_id,
        "mode": "aws",
        "language": detected_lang,
        "turns": turns,
    }

    return result


def _parse_result(raw: dict) -> list[dict]:
    """Parse AWS Transcribe JSON into speaker-labeled turns."""
    items = raw["results"]["items"]
    speaker_segments = raw["results"]["speaker_labels"]["segments"]

    word_speaker_map = {}
    for segment in speaker_segments:
        for item in segment["items"]:
            if "start_time" in item:
                word_speaker_map[item["start_time"]] = segment["speaker_label"]

    turns = []
    current_speaker = None
    current_words = []
    current_start = None
    current_end = None

    for item in items:
        if item["type"] == "punctuation":
            if current_words:
                current_words.append(item["alternatives"][0]["content"])
            continue

        start_time = item.get("start_time")
        end_time = item.get("end_time")
        word = item["alternatives"][0]["content"]
        speaker = word_speaker_map.get(start_time, current_speaker)

        if speaker != current_speaker and current_words:
            turns.append({
                "speaker": current_speaker,
                "start": float(current_start),
                "end": float(current_end),
                "text": _join_words(current_words),
            })
            current_words = []
            current_start = None

        current_speaker = speaker
        if current_start is None:
            current_start = start_time
        current_end = end_time
        current_words.append(word)

    if current_words:
        turns.append({
            "speaker": current_speaker,
            "start": float(current_start),
            "end": float(current_end),
            "text": _join_words(current_words),
        })

    return turns
