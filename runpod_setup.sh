#!/bin/bash

# RunPod GPU Pod Setup Script for RL Weather Training
# This script sets up everything needed to train the model on RunPod
# Usage: bash runpod_setup.sh

set -e  # Exit on any error

echo "=================================================="
echo "RL Weather - RunPod GPU Pod Setup"
echo "=================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Update system packages
echo -e "${BLUE}[1/6] Updating system packages...${NC}"
apt-get update -qq
apt-get install -y -qq git curl wget >/dev/null 2>&1
echo -e "${GREEN}✓ System packages updated${NC}"

# Step 2: Clone repository
echo -e "${BLUE}[2/6] Cloning repository...${NC}"
cd /workspace || cd /root || cd /tmp
REPO_URL="https://github.com/Baswold/weather-ai.git"
REPO_DIR="weather-ai"

if [ -d "$REPO_DIR" ]; then
    echo "Repository already exists, updating..."
    cd "$REPO_DIR"
    git pull origin main -q
else
    git clone "$REPO_URL" -q
    cd "$REPO_DIR"
fi

REPO_PATH=$(pwd)
echo -e "${GREEN}✓ Repository cloned/updated at $REPO_PATH${NC}"

# Step 3: Install Python dependencies
echo -e "${BLUE}[3/6] Installing Python dependencies...${NC}"
pip install -q --upgrade pip setuptools wheel >/dev/null 2>&1

# Install PyTorch with CUDA support (optimized for GPU)
echo "  Installing PyTorch with CUDA support..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 >/dev/null 2>&1

# Install other dependencies
pip install -q -r requirements.txt >/dev/null 2>&1
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Step 4: Verify setup
echo -e "${BLUE}[4/6] Verifying setup...${NC}"
python tests/verify_setup.py > /tmp/setup_check.txt 2>&1
if grep -q "ALL PRE-FLIGHT CHECKS PASSED" /tmp/setup_check.txt; then
    echo -e "${GREEN}✓ Pre-flight checks passed${NC}"
else
    echo -e "${YELLOW}⚠ Some checks failed (continuing anyway)${NC}"
fi

# Step 5: Detect GPU and configure
echo -e "${BLUE}[5/6] Detecting GPU and configuring...${NC}"
python -c "
import torch
cuda_available = torch.cuda.is_available()
if cuda_available:
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'✓ GPU: {gpu_name}')
    print(f'✓ GPU Memory: {gpu_memory:.1f} GB')

    # Recommend config based on GPU memory
    if gpu_memory >= 40:
        config = '24gb'
        print(f'Recommended config: {config} (40GB+ GPU)')
    elif gpu_memory >= 24:
        config = '24gb'
        print(f'Recommended config: {config}')
    elif gpu_memory >= 16:
        config = 'production'
        print(f'Recommended config: {config}')
    else:
        config = 'default'
        print(f'Recommended config: {config}')
else:
    print('✗ No GPU detected')
"

echo -e "${GREEN}✓ GPU configured${NC}"

# Step 6: Create training command
echo -e "${BLUE}[6/6] Setting up training...${NC}"

# Detect available GPU memory for auto-config
PYTHON_CODE='
import torch
import psutil

cuda_available = torch.cuda.is_available()
if cuda_available:
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    # Use 80% of GPU memory
    target_gb = gpu_memory * 0.8
    print(f"--auto --target-gb {target_gb:.0f}")
else:
    # Fallback to CPU with 75% system RAM
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    print(f"--auto --target-gb {available_gb * 0.75:.0f}")
'

TRAIN_ARGS=$(python -c "$PYTHON_CODE")

echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""
echo "=================================================="
echo "Ready to train!"
echo "=================================================="
echo ""
echo "To start training, run:"
echo ""
echo -e "${YELLOW}cd $REPO_PATH${NC}"
echo -e "${YELLOW}python train.py $TRAIN_ARGS --epochs 3 --chunked${NC}"
echo ""
echo "Or choose specific config:"
echo "  python train.py --config 24gb --epochs 5"
echo "  python train.py --config production --epochs 3"
echo "  python train.py --config default --epochs 2"
echo ""
echo "Monitor memory:"
echo "  watch -n 1 'nvidia-smi' (if GPU available)"
echo ""
echo "View progress:"
echo "  tail -f checkpoints/*/training.log (if logging enabled)"
echo ""

# Optional: Start training automatically
read -p "Start training now? (y/N): " -t 30 response

if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
    echo ""
    echo -e "${BLUE}Starting training with: python train.py $TRAIN_ARGS --epochs 3 --chunked${NC}"
    echo ""
    cd "$REPO_PATH"
    python train.py $TRAIN_ARGS --epochs 3 --chunked
else
    echo "Setup complete. Run 'cd $REPO_PATH && python train.py ...' to start training"
fi
