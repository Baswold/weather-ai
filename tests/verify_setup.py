#!/usr/bin/env python3
"""
Pre-flight checklist: Verify everything is set up correctly before training.

Checks:
1. Dependencies are installed
2. Data pipeline works end-to-end
3. Model can do a forward pass with real data
4. GPU/device detection works
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import numpy as np
from datetime import datetime

print("\n" + "="*70)
print("PRE-FLIGHT CHECKLIST")
print("="*70)

# Step 1: Check dependencies
print("\n1. Checking dependencies...")
try:
    import pandas as pd
    import requests
    import psutil
    print("   ✓ All required packages found")
except ImportError as e:
    print(f"   ✗ Missing dependency: {e}")
    print("   Install with: pip install -r requirements.txt")
    sys.exit(1)

# Step 2: Check PyTorch and device
print("\n2. Checking PyTorch and device...")
print(f"   PyTorch version: {torch.__version__}")
cuda_available = torch.cuda.is_available()
if cuda_available:
    print(f"   ✓ CUDA available")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
else:
    print(f"   ⚠ CUDA not available (will use CPU - slower)")

# Step 3: Check system memory
print("\n3. Checking system memory...")
mem = psutil.virtual_memory()
total_gb = mem.total / (1024**3)
available_gb = mem.available / (1024**3)
print(f"   Total RAM: {total_gb:.1f} GB")
print(f"   Available RAM: {available_gb:.1f} GB")

# Step 4: Test data pipeline
print("\n4. Testing data pipeline...")
try:
    from src.data.openmeteo import OpenMeteoClient
    from src.data.loader import WeatherDataLoader

    client = OpenMeteoClient()
    print("   ✓ OpenMeteoClient initialized")

    # Quick test: fetch 1 week of data
    print("   Fetching test data (1 week for 1 location)...")
    df = client.get_forecast_verification(
        latitude=40.71,
        longitude=-74.01,
        start_date="2024-01-01",
        end_date="2024-01-07",
    )
    print(f"   ✓ Successfully fetched {len(df)} days of data")
    print(f"   Columns: {list(df.columns)}")

except Exception as e:
    print(f"   ✗ Data pipeline failed: {e}")
    sys.exit(1)

# Step 5: Test data loader
print("\n5. Testing WeatherDataLoader...")
try:
    loader = WeatherDataLoader(
        locations=[{"name": "Test", "latitude": 40.71, "longitude": -74.01}],
        start_date="2024-01-01",
        end_date="2024-01-31",
        window_size=7,
        data_dir="data/test",
        cache=False,
    )

    # Get one batch
    date, batch = next(iter(loader))
    print(f"   ✓ Data loader works")
    print(f"   Batch shapes:")
    print(f"     History: {batch.history.shape}")
    print(f"     Current forecast: {batch.current_forecast.shape}")
    print(f"     Target: {batch.target.shape}")

except Exception as e:
    print(f"   ✗ Data loader failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Test model architecture
print("\n6. Testing model architecture...")
try:
    from src.models import ActorCritic, TransformerConfig
    from src.utils import count_parameters, get_model_size_mb

    config = TransformerConfig(
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=512,
        num_weather_vars=6,
        window_size=7,
        dropout=0.1,
    )

    model = ActorCritic(config, use_twin_critic=True)
    print(f"   ✓ Model created")
    print(f"   Parameters: {count_parameters(model):,}")
    print(f"   Model size: {get_model_size_mb(model):.1f} MB")

    # Test forward pass
    device = torch.device("cuda" if cuda_available else "cpu")
    model = model.to(device)

    # Use real batch data (convert to float64 first to ensure compatibility)
    history_array = np.asarray(batch.history, dtype=np.float64)
    current_array = np.asarray(batch.current_forecast, dtype=np.float64)
    history_tensor = torch.from_numpy(history_array).float().to(device)
    current_tensor = torch.from_numpy(current_array).float().to(device)

    with torch.no_grad():
        action = model.get_action(history_tensor, current_tensor, deterministic=True)

    print(f"   ✓ Forward pass successful")
    print(f"   Input shapes:")
    print(f"     History: {history_tensor.shape}")
    print(f"     Current: {current_tensor.shape}")
    print(f"   Output shape: {action.shape}")

except Exception as e:
    print(f"   ✗ Model failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 7: Test trainer
print("\n7. Testing trainer setup...")
try:
    from src.rl import RLTrainer, TrainerConfig, WeightedReward

    trainer_config = TrainerConfig(
        model_config=config,
        learning_rate=1e-4,
        batch_size=1,
        replay_buffer_size=100,
        replay_start_size=10,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.1,
        policy_freq=2,
        exploration_noise=0.1,
        exploration_anneal=0.99,
        device="cuda" if cuda_available else "cpu",
        checkpoint_dir="checkpoints",
        checkpoint_interval=90,
        log_interval=5,
    )

    reward_fn = WeightedReward(
        variable_weights={0: 2.0, 1: 2.0, 2: 1.0, 3: 3.0, 4: 1.5, 5: 1.0},
        num_vars=6,
    )

    trainer = RLTrainer(model, reward_fn, trainer_config)
    print(f"   ✓ Trainer initialized")

except Exception as e:
    print(f"   ✗ Trainer initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "="*70)
print("✓ ALL PRE-FLIGHT CHECKS PASSED")
print("="*70)
print("\nYou're ready to train! Try:")
print("  python train.py --config low_memory --epochs 1")
print("  python train.py --auto --target-gb 8")
print("\n" + "="*70)
