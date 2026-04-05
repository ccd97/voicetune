#!/bin/bash
set -euo pipefail

BUCKET=$(curl -sf -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/BUCKET 2>/dev/null || echo "gs://voicetune-finetune-cdcunha")
MAX_STEPS=$(curl -sf -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/MAX_STEPS 2>/dev/null || echo 500)
HF_TOKEN=$(curl -sf -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/HF_TOKEN 2>/dev/null || echo "")
export HF_TOKEN
VOX_DIR="/opt/voxcpm"
GUEST_ATTR="http://metadata.google.internal/computeMetadata/v1/instance/guest-attributes/voicetune/status"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
set_status() { curl -sf -X PUT -d "$1" -H "Metadata-Flavor: Google" "$GUEST_ATTR"; }

trap 'set_status "FAILED: bootstrap line $LINENO"' ERR

set_status "STARTING"

log "Installing system deps..."
apt-get update -qq && apt-get install -y -qq python3.12-venv git > /dev/null 2>&1

if ! nvidia-smi > /dev/null 2>&1; then
    log "Installing NVIDIA drivers..."
    /opt/deeplearning/install-driver.sh
fi
nvidia-smi
set_status "DRIVERS_READY"

log "Cloning VoxCPM..."
git clone https://github.com/OpenBMB/VoxCPM.git "$VOX_DIR"
cd "$VOX_DIR"

log "Installing VoxCPM deps..."
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu129
.venv/bin/pip install -q -e .
.venv/bin/pip install -q tensorboardX google-cloud-storage huggingface_hub pyyaml
set_status "DEPS_INSTALLED"

log "Starting training orchestrator..."
gcloud storage cp "${BUCKET}/train_gcp.py" /opt/train_gcp.py
trap - ERR
HF_TOKEN="$HF_TOKEN" .venv/bin/python /opt/train_gcp.py \
    --bucket "$BUCKET" \
    --max-steps "$MAX_STEPS" \
    --vox-dir "$VOX_DIR"
