#!/bin/bash
#SBATCH --job-name=test_phaseDE
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia-gpu
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/test_phaseDE_%j.out
#SBATCH --error=logs/test_phaseDE_%j.err

# ==============================================================================
# Minimal Test for Phase D and E Implementations
# ==============================================================================
# Quick sanity check to verify Phase D/E code loads and runs without errors.
# ==============================================================================

set -e

module load cuda/12.1
module load anaconda3/2023.03
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

echo "=============================================="
echo "Phase D/E Minimal Test"
echo "=============================================="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=============================================="

# Test 1: Import Phase D/E modules
echo ""
echo "=== Test 1: Import Phase D/E Modules ==="
python -c "
from src.skip_router import SkipRouter, SkipRouterConfig
from src.flexi_adapter import FlexiDepthManager, FlexiRouterConfig, FlexiAdapter
print('✓ Phase D/E imports successful')

# Create Phase D skip router
config_d = SkipRouterConfig(num_layers=32)
skip_router = SkipRouter(config_d)
print(f'✓ SkipRouter created with {skip_router.num_params:,} params')

# Create Phase E FlexiDepth manager
config_e = FlexiRouterConfig(
    num_layers=32,
    hidden_dim=4096,
    adapted_layers=[24, 25, 26, 27, 28, 29, 30, 31],
    adapter_dim=256,
)
flexi_manager = FlexiDepthManager(config_e)
print(f'✓ FlexiDepthManager created with {flexi_manager.total_params:,} params')
"

# Test 2: Save and load checkpoints
echo ""
echo "=== Test 2: Save/Load Checkpoints ==="
python -c "
import torch
import tempfile
import os

from src.skip_router import SkipRouter, SkipRouterConfig
from src.flexi_adapter import FlexiDepthManager, FlexiRouterConfig

with tempfile.TemporaryDirectory() as tmpdir:
    # Test Phase D
    config_d = SkipRouterConfig(num_layers=32)
    skip_router = SkipRouter(config_d)
    skip_path = os.path.join(tmpdir, 'skip_router.pt')
    skip_router.save(skip_path)
    loaded_skip = SkipRouter.load(skip_path)
    print(f'✓ SkipRouter save/load successful')
    
    # Test Phase E
    config_e = FlexiRouterConfig(
        num_layers=32,
        hidden_dim=4096,
        adapted_layers=[24, 25, 26, 27],
        adapter_dim=256,
    )
    flexi_manager = FlexiDepthManager(config_e)
    flexi_path = os.path.join(tmpdir, 'flexi_depth.pt')
    flexi_manager.save(flexi_path)
    loaded_flexi = FlexiDepthManager.load(flexi_path)
    print(f'✓ FlexiDepthManager save/load successful')
"

# Test 3: Forward pass
echo ""
echo "=== Test 3: Forward Pass (CPU) ==="
python -c "
import torch
from src.skip_router import SkipRouter, SkipRouterConfig
from src.flexi_adapter import FlexiDepthManager, FlexiRouterConfig

# Phase D forward
config_d = SkipRouterConfig(num_layers=32)
skip_router = SkipRouter(config_d)

progress = 0.5
decisions = skip_router.get_skip_decisions(progress, threshold=0.5, device='cpu')
print(f'✓ SkipRouter decisions at p={progress}: recompute={sum(decisions)}/{len(decisions)} layers')

# Phase E forward
config_e = FlexiRouterConfig(
    num_layers=32,
    hidden_dim=4096,
    adapted_layers=[24, 25, 26, 27],
    adapter_dim=256,
)
flexi_manager = FlexiDepthManager(config_e)

gates = flexi_manager.router.get_all_gates(progress, device='cpu')
print(f'✓ FlexiRouter gates at p={progress}: {len(gates)} layers')
for layer_idx, score in gates.items():
    path = 'deep' if score >= 0.5 else 'shallow'
    print(f'  Layer {layer_idx}: {score:.3f} -> {path}')
"

echo ""
echo "=============================================="
echo "All Phase D/E tests passed!"
echo "=============================================="
