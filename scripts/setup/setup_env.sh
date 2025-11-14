#!/bin/bash
# Setup script for DLLM-Delta-Compute development environment
# Run on PACE ICE cluster

set -e

echo "=== DLLM-Delta-Compute Environment Setup ==="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda/Miniconda first."
    exit 1
fi

# Create conda environment
ENV_NAME="dcllm"
echo "Creating conda environment: $ENV_NAME"

if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' already exists. Skipping creation."
else
    conda create -n $ENV_NAME python=3.10 -y
fi

# Activate environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Install PyTorch (CUDA 12.1 compatible)
echo "Installing PyTorch 2.5.1 with CUDA 12.1..."
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install transformers and other dependencies
echo "Installing transformers and dependencies..."
pip install transformers==4.46.2
pip install accelerate==0.34.2
pip install datasets==2.14.5
pip install sentencepiece==0.1.99
pip install protobuf==3.20.3

# Install visualization dependencies
echo "Installing visualization tools..."
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
pip install numpy==1.24.3

# Install eval dependencies
echo "Installing evaluation tools..."
pip install scikit-learn==1.3.0
pip install tqdm==4.66.1

# Verify installation
echo ""
echo "=== Verifying Installation ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')"

echo ""
echo "=== Setup Complete! ==="
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To test the installation:"
echo "  python test_poc_v1a.py --device cpu"
