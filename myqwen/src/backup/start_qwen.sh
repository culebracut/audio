#!/bin/bash

# 1. Set environment variables to handle the /workspace move
export CONDA_ENVS_PATH="/data/miniconda3/envs"
export PIP_CACHE_DIR="/data/.cache/pip"
export PYTHONNOUSERSITE=1  # Prevents conflicts with ~/.local packages

# 2. Source Conda and activate the environment
# Adjust the path below if your miniconda is in a different spot
source /home/system/miniconda3/etc/profile.d/conda.sh
conda activate qwen3-tts

# 3. Verify Transformers version before starting
VERSION=$(python -c "import transformers; print(transformers.__version__)")
echo "Starting Qwen3-TTS with Transformers $VERSION..."

# 4. Launch your app (Replace 'main.py' with your actual script name)
python /data/qwen/src/main.py
