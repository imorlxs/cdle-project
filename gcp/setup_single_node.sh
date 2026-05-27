#!/bin/bash
# ---------------------------------------------------------------------------
# GCP: Create a single high-memory VM for LOCAL execution benchmarks
#
# Equivalent to the blog's AWS i3.16xlarge (64 vCPUs, 488 GB RAM).
# GCP closest match: n2-highmem-64 (64 vCPUs, 512 GB RAM)
#
# Prerequisites:
#   gcloud auth login
#   gcloud config set project YOUR_PROJECT_ID
#
# Usage:
#   bash gcp/setup_single_node.sh
# ---------------------------------------------------------------------------
set -e

PROJECT=$(gcloud config get-value project)
ZONE="europe-southwest1-c"
INSTANCE_NAME="cdle-single-node"
MACHINE_TYPE="n2-highmem-16"
DISK_SIZE="500GB"
IMAGE_FAMILY="debian-11"
IMAGE_PROJECT="debian-cloud"
BUCKET_NAME="${PROJECT}-cdle-data"     # GCS bucket for taxi data

echo "=== Creating GCS bucket: ${BUCKET_NAME} ==="
gcloud storage buckets create "gs://${BUCKET_NAME}" \
    --location=us-central1 \
    --uniform-bucket-level-access 2>/dev/null || echo "(bucket already exists)"

echo "=== Creating VM: ${INSTANCE_NAME} (${MACHINE_TYPE}) ==="
gcloud compute instances create "${INSTANCE_NAME}" \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    --machine-type="${MACHINE_TYPE}" \
    --image-family="${IMAGE_FAMILY}" \
    --image-project="${IMAGE_PROJECT}" \
    --boot-disk-size="${DISK_SIZE}" \
    --boot-disk-type="pd-ssd" \
    --scopes="cloud-platform" \
    --metadata="startup-script=#! /bin/bash
        apt-get update -y
        apt-get install -y python3-pip python3-venv openjdk-11-jdk git wget
        echo 'export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64' >> /etc/environment
    "

echo ""
echo "=== VM created.  Next steps: ==="
echo "  1. SSH: gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE}"
echo "  2. Clone repo and install deps (see gcp/install_deps.sh)"
echo "  3. Set env vars:"
echo "       export GCS_BUCKET=${BUCKET_NAME}"
echo "       export DATA_DIR=/home/\$USER/data"
echo "  4. Download data: python scripts/download_data.py"
echo "  5. Run notebooks: jupyter notebook --no-browser --port=8888"
echo "     (forward port via: gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} -- -L 8888:localhost:8888)"
