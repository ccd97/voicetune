#!/bin/bash
set -euo pipefail

BUCKET=$(curl -sf -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/BUCKET 2>/dev/null || echo "gs://voicetune-finetune-cdcunha")
MAX_STEPS=$(curl -sf -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/MAX_STEPS 2>/dev/null || echo 1000)
FISH_DIR="/opt/fish-speech"
GUEST_ATTR="http://metadata.google.internal/computeMetadata/v1/instance/guest-attributes/voicetune/status"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
set_status() { curl -sf -X PUT -d "$1" -H "Metadata-Flavor: Google" "$GUEST_ATTR"; }

trap 'set_status "FAILED: bootstrap line $LINENO"' ERR

set_status "STARTING"

log "Installing system deps..."
apt-get update -qq && apt-get install -y -qq python3.12-venv git portaudio19-dev > /dev/null 2>&1

if ! nvidia-smi > /dev/null 2>&1; then
    log "Installing NVIDIA drivers..."
    /opt/deeplearning/install-driver.sh
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
.venv/bin/pip install -q google-cloud-storage
set_status "DEPS_INSTALLED"

# Hand off to Python — it handles status and VM lifecycle from here
log "Starting training orchestrator..."
gcloud storage cp "${BUCKET}/train_gcp.py" /opt/train_gcp.py
trap - ERR
.venv/bin/python /opt/train_gcp.py \
    --bucket "$BUCKET" \
    --max-steps "$MAX_STEPS" \
    --fish-dir "$FISH_DIR"
