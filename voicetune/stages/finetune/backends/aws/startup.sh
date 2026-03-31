#!/bin/bash
set -euo pipefail

# BUCKET and MAX_STEPS are baked into user data by provider.py
FISH_DIR="/opt/fish-speech"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
set_status() { echo -n "$1" | aws s3 cp - "s3://${BUCKET}/status.txt"; }

trap 'set_status "FAILED: bootstrap line $LINENO"' ERR

set_status "STARTING"

log "Installing system deps..."
apt-get update -qq && apt-get install -y -qq python3.12-venv git portaudio19-dev > /dev/null 2>&1

if ! nvidia-smi > /dev/null 2>&1; then
    log "NVIDIA drivers not found — DLAMI should have them pre-installed"
    exit 1
fi
nvidia-smi
set_status "DRIVERS_READY"

log "Cloning fish-speech..."
git clone --depth 1 https://github.com/fishaudio/fish-speech.git "$FISH_DIR"
cd "$FISH_DIR"

log "Installing fish-speech deps..."
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -e ".[cu129]" --extra-index-url https://download.pytorch.org/whl/cu129
.venv/bin/pip install -q --upgrade protobuf
.venv/bin/pip install -q --upgrade wandb
.venv/bin/pip install -q boto3
set_status "DEPS_INSTALLED"

log "Starting training orchestrator..."
aws s3 cp "s3://${BUCKET}/train.py" /opt/train.py
trap - ERR
.venv/bin/python /opt/train.py \
    --bucket "$BUCKET" \
    --max-steps "$MAX_STEPS" \
    --fish-dir "$FISH_DIR"
