#!/bin/bash
# ---------------------------------------------------------------------------
# GCP: Create a Dataproc cluster for DISTRIBUTED execution benchmarks
#
# Blog configuration: 1 master + 3 workers, each i3.4xlarge (16 vCPUs, 122 GB)
# GCP equivalent:     n2-highmem-16 (16 vCPUs, 128 GB RAM) per node
#
# Dataproc runs PySpark (and therefore Koalas/pyspark.pandas) natively.
# Dask-on-Spark can also run here, or use a separate Dask scheduler VM.
#
# Usage:
#   bash gcp/setup_dataproc_cluster.sh
# ---------------------------------------------------------------------------
set -e

PROJECT=$(gcloud config get-value project)
REGION="us-central1"
ZONE="${REGION}-a"
CLUSTER_NAME="cdle-cluster"
MASTER_TYPE="n2-highmem-16"     # downscale to n2-highmem-8 if credits are low
WORKER_TYPE="n2-highmem-16"     # 3 workers  (blog: 3 × i3.4xlarge)
NUM_WORKERS=3
DISK_SIZE="200GB"
BUCKET_NAME="${PROJECT}-cdle-data"
SPARK_VERSION="3.4"
PYTHON_VERSION="3.10"

echo "=== Creating Dataproc cluster: ${CLUSTER_NAME} ==="
gcloud dataproc clusters create "${CLUSTER_NAME}" \
    --project="${PROJECT}" \
    --region="${REGION}" \
    --zone="${ZONE}" \
    --master-machine-type="${MASTER_TYPE}" \
    --master-boot-disk-size="${DISK_SIZE}" \
    --num-workers="${NUM_WORKERS}" \
    --worker-machine-type="${WORKER_TYPE}" \
    --worker-boot-disk-size="${DISK_SIZE}" \
    --image-version="${SPARK_VERSION}.0-debian11" \
    --properties="spark:spark.executor.memory=100g,spark:spark.driver.memory=20g,\
spark:spark.sql.shuffle.partitions=200,\
dataproc:dataproc.conscrypt.provider.enable=false" \
    --optional-components=JUPYTER \
    --enable-component-gateway \
    --bucket="${BUCKET_NAME}" \
    --metadata="PIP_PACKAGES=pyspark pandas pyarrow numpy scikit-learn xgboost seaborn matplotlib"

echo ""
echo "=== Cluster created.  Next steps: ==="
echo "  Open Jupyter via Component Gateway:"
echo "    gcloud dataproc clusters describe ${CLUSTER_NAME} --region=${REGION} | grep -A2 endpointConfig"
echo ""
echo "  Or SSH to master:"
echo "    gcloud compute ssh ${CLUSTER_NAME}-m --zone=${ZONE} -- -L 8888:localhost:8888"
echo ""
echo "  Data path on GCS: gs://${BUCKET_NAME}/taxi-benchmark/"
echo ""
echo "  To DELETE the cluster when done (saves credits):"
echo "    gcloud dataproc clusters delete ${CLUSTER_NAME} --region=${REGION}"
