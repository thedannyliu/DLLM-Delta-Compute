import sys
import os
import torch
from pathlib import Path
import shutil

# Add src
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from tracing import TraceCollector, LayerStepStats

def create_dummy_trace(path):
    collector = TraceCollector(enabled=True, num_layers=4)
    for step in range(10):
        collector.start_step(step)
        for layer in range(4):
            # Random output
            out = torch.randn(1, 10, 64)
            collector.record_layer(layer, step, out)
        collector.end_step()
    collector.save(path)

def test_training():
    # Setup
    test_dir = Path("tests/temp_training_test")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(exist_ok=True)
    trace_dir = test_dir / "traces"
    trace_dir.mkdir(exist_ok=True)
    
    # Create dummy traces
    print("Creating dummy traces...")
    for i in range(5):
        create_dummy_trace(str(trace_dir / f"trace_{i}.pt"))
        
    # 1. Test Oracle Label Generation
    print("\nTesting Oracle Label Generation...")
    ret = os.system(f"python scripts/training/generate_oracle_labels.py --trace_dir {trace_dir} --output {test_dir}/oracle.json")
    if ret != 0:
        print("Oracle Label Generation FAILED")
        return
        
    # 2. Test P3 Training
    print("\nTesting P3 Training...")
    ret = os.system(f"python scripts/training/train_learned_gate.py --trace_dir {trace_dir} --oracle_labels {test_dir}/oracle.json --output_dir {test_dir}/p3 --num_epochs 1 --device cpu")
    if ret != 0:
        print("P3 Training FAILED")
        return

    # 3. Test Phase B Training
    print("\nTesting Phase B Training...")
    ret = os.system(f"python scripts/training/train_learned_router.py --trace_dir {trace_dir} --output_dir {test_dir}/phase_b --num_epochs 1 --num_layers 4 --num_steps 10 --device cpu")
    if ret != 0:
        print("Phase B Training FAILED")
        return

    # 4. Test Phase C Training
    print("\nTesting Phase C Training...")
    ret = os.system(f"python scripts/training/train_continuous_router.py --trace_dir {trace_dir} --output_dir {test_dir}/phase_c --num_epochs 1 --num_layers 4 --total_steps 10 --device cpu")
    if ret != 0:
        print("Phase C Training FAILED")
        return
        
    print("\nALL TESTS PASSED")
    # Cleanup
    shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_training()
