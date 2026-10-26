#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ENV_NAME="dredd_env"

echo "Creating Conda environment: $ENV_NAME"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH."
    exit 1
fi

# Create environment with python 3.10
conda create -n $ENV_NAME python=3.10 -y

# Determine conda base path and activate profile
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Activate environment
conda activate $ENV_NAME

# Install requirements relative to script directory
REQ_FILE="$SCRIPT_DIR/requirements.txt"
if [ -f "$REQ_FILE" ]; then
    echo "Installing dependencies from $REQ_FILE..."
    pip install -r "$REQ_FILE"
else
    echo "Warning: requirements.txt not found at $REQ_FILE"
fi

echo "Environment $ENV_NAME created and configured successfully."
echo "To activate it, run: conda activate $ENV_NAME"