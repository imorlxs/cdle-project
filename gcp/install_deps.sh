#!/bin/bash
# ---------------------------------------------------------------------------
# Install Python dependencies on a GCP VM (single-node setup).
# Run this after SSH-ing into the VM created by setup_single_node.sh.
#
# Usage:
#   bash gcp/install_deps.sh
# ---------------------------------------------------------------------------
set -e

REPO_DIR="$HOME/cdle-project"

echo "=== Cloning repo (if not already present) ==="
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/imorlxs/cdle-project.git "$REPO_DIR"
fi
cd "$REPO_DIR"

echo "=== Creating Python virtual environment ==="
python3 -m venv venv
source venv/bin/activate

echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install -r requirements-pyspark.txt

echo "=== Checking Java for PySpark ==="
java -version 2>&1 | head -1
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
echo "JAVA_HOME=$JAVA_HOME"

echo "=== Writing environment file ==="
cat > "$REPO_DIR/.env" <<EOF
# Source this file before running notebooks:  source .env
export JAVA_HOME=$JAVA_HOME
export GCS_BUCKET=\${PROJECT}-cdle-data    # replace PROJECT with your GCP project id
export DATA_DIR=$HOME/data
export PYSPARK_PYTHON=$(which python3)
export PYSPARK_DRIVER_PYTHON=$(which python3)
EOF

echo ""
echo "=== Done.  To activate: ==="
echo "  cd $REPO_DIR && source venv/bin/activate && source .env"
echo "  jupyter notebook --no-browser --port=8888"
