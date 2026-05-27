#!/bin/bash
# ---------------------------------------------------------------------------
# Install Python dependencies on a GCP VM (single-node setup).
# Run this after SSH-ing into the VM created by setup_single_node.sh.
#
# Uses pyenv to install Python 3.7.5 so that exact blog library versions
# (koalas 1.7.0, pyarrow 1.0.1, etc.) can be installed without conflicts.
#
# Usage:
#   bash gcp/install_deps.sh
# ---------------------------------------------------------------------------
set -e

REPO_DIR="$HOME/cdle-project"
PYTHON_VERSION="3.7.5"

echo "=== Cloning repo (if not already present) ==="
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/imorlxs/cdle-project.git "$REPO_DIR"
fi
cd "$REPO_DIR"

echo "=== Installing pyenv build dependencies ==="
sudo apt-get update -y
sudo apt-get install -y \
    make build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm \
    libncurses5-dev libncursesw5-dev xz-utils tk-dev \
    libffi-dev liblzma-dev

echo "=== Installing pyenv ==="
if [ ! -d "$HOME/.pyenv" ]; then
    curl https://pyenv.run | bash
fi

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Persist pyenv in shell startup files
for RC in "$HOME/.bashrc" "$HOME/.bash_profile"; do
    if ! grep -q 'pyenv init' "$RC" 2>/dev/null; then
        cat >> "$RC" <<'PYENVEOF'

# pyenv
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
PYENVEOF
    fi
done

echo "=== Installing Python ${PYTHON_VERSION} via pyenv ==="
pyenv install -s "${PYTHON_VERSION}"
pyenv local "${PYTHON_VERSION}"
python --version

echo "=== Creating Python ${PYTHON_VERSION} virtual environment ==="
python -m venv venv37
source venv37/bin/activate
python --version

echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install -r requirements-pyspark.txt

echo "=== Configuring Java for PySpark 3.1.x (requires Java 8 or 11, NOT 17) ==="
sudo apt-get install -y openjdk-11-jdk 2>/dev/null || true
sudo update-alternatives --set java /usr/lib/jvm/java-11-openjdk-amd64/bin/java 2>/dev/null || true
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
java -version 2>&1 | head -1
echo "JAVA_HOME=$JAVA_HOME"

echo "=== Writing environment file ==="
cat > "$REPO_DIR/.env" <<EOF
# Source this file before running notebooks:  source .env
# PySpark 3.1.x requires Java 11 (not 17)
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export GCS_BUCKET=steel-watch-488511-k6-cdle-data
export DATA_DIR=$HOME/data
export PYSPARK_PYTHON=$(which python)
export PYSPARK_DRIVER_PYTHON=$(which python)
EOF

echo ""
echo "=== Done.  To activate: ==="
echo "  cd $REPO_DIR && source venv37/bin/activate && source .env"
echo "  jupyter notebook --no-browser --port=8888"
