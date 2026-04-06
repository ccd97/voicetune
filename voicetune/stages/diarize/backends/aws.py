"""AWS Transcribe diarization backend."""

import json
import logging
import os
import time
import urllib.request
import uuid
from pathlib import Path

from voicetune.stages.preprocess.paths import get_call_id

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


def _make_clients():
    import boto3

    region = os.environ.get("AWS_REGION", "us-east-1")
    bucket = os.environ["AWS_S3_BUCKET"]
    s3 = boto3.client("s3", region_name=region)
    transcribe = boto3.client("transcribe", region_name=region)
    return s3, transcribe, bucket


def _upload_and_start(audio_path: Path, num_speakers: int | None, language: str | None,
                      s3, transcribe, bucket) -> dict:
    """Upload WAV to S3 and start a transcription job. Returns job metadata."""
    s3_key = f"diarization-input/{audio_path.stem}_{uuid.uuid4().hex[:8]}.wav"
    log.info(f"Uploading {audio_path.name} to s3://{bucket}/{s3_key}")
    s3.upload_file(str(audio_path), bucket, s3_key)

    job_name = f"diarize-{audio_path.stem}-{uuid.uuid4().hex[:8]}"
    media_uri = f"s3://{bucket}/{s3_key}"
    settings = {"ShowSpeakerLabels": True, "MaxSpeakerLabels": num_speakers or 3}

    lang_kwargs = {}
    if language:
        lang_kwargs["LanguageCode"] = language
    else:
        lang_kwargs["IdentifyLanguage"] = True
        lang_kwargs["LanguageOptions"] = ["en-US", "hi-IN", "mr-IN"]

    log.info(f"Starting transcription job: {job_name}")
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": media_uri},
        MediaFormat="wav",
        Settings=settings,
        **lang_kwargs,
    )

    return {"job_name": job_name, "s3_key": s3_key, "audio_path": audio_path}


def _poll_jobs(jobs: list[dict], transcribe) -> tuple[list[dict], list[dict]]:
    """Poll all jobs until each completes or fails."""
    pending = {j["job_name"]: j for j in jobs}
    completed = []
    failed = []

    while pending:
        for job_name in list(pending):
            status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
            job_status = status["TranscriptionJob"]["TranscriptionJobStatus"]

            if job_status == "COMPLETED":
                pending[job_name]["status_response"] = status
                completed.append(pending.pop(job_name))
            elif job_status == "FAILED":
                reason = status["TranscriptionJob"].get("FailureReason", "Unknown")
                pending[job_name]["error"] = reason
                failed.append(pending.pop(job_name))

        if pending:
            log.info(f"Polling: {len(completed)} done, {len(failed)} failed, {len(pending)} pending — waiting 15s")
            time.sleep(15)

    return completed, failed


def _collect_result(job: dict, s3, bucket) -> dict:
    """Download result, clean up S3, parse into standard format."""
    status = job["status_response"]
    audio_path = job["audio_path"]

    result_url = status["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
    with urllib.request.urlopen(result_url) as resp:
        raw_result = json.loads(resp.read().decode())

    log.info(f"Cleaning up s3://{bucket}/{job['s3_key']}")
    s3.delete_object(Bucket=bucket, Key=job["s3_key"])

    call_id = get_call_id(audio_path)
    turns = _parse_result(raw_result)
    detected_lang = status["TranscriptionJob"].get("LanguageCode", "unknown")
    log.info(f"{audio_path.name}: {len(turns)} turns, language={detected_lang}")

    return {
        "call_id": call_id,
        "mode": "aws",
        "language": detected_lang,
        "turns": turns,
    }


def diarize_batch(audio_paths: list[Path], num_speakers: int | None = None,
                  language: str | None = None) -> list[tuple[Path, dict | str]]:
    """Submit all files, poll concurrently, return per-file results.

    Returns list of (audio_path, result_dict) for successes and
    (audio_path, error_string) for failures.
    """
    if not audio_paths:
        return []

    s3, transcribe, bucket = _make_clients()

    jobs = []
    submit_errors = []
    for path in audio_paths:
        try:
            job = _upload_and_start(path, num_speakers, language, s3, transcribe, bucket)
            jobs.append(job)
        except Exception as e:
            log.error(f"Failed to submit {path.name}: {e}")
            submit_errors.append((path, str(e)))

    log.info(f"Submitted {len(jobs)} job(s), {len(submit_errors)} submit failure(s)")

    completed, failed = _poll_jobs(jobs, transcribe)

    results: list[tuple[Path, dict | str]] = []
    for job in completed:
        try:
            result = _collect_result(job, s3, bucket)
            results.append((job["audio_path"], result))
        except Exception as e:
            log.error(f"Failed to collect result for {job['audio_path'].name}: {e}")
            results.append((job["audio_path"], str(e)))

    for job in failed:
        try:
            s3.delete_object(Bucket=bucket, Key=job["s3_key"])
        except Exception as e:
            log.warning(f"Failed to clean up s3://{bucket}/{job['s3_key']}: {e}")
        results.append((job["audio_path"], job["error"]))

    results.extend(submit_errors)
    return results


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
