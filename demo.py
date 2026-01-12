#!/usr/bin/env python3
"""
Simple demo/training script for limited hardware.

This demonstrates the framework with minimal resource usage:
- 3 locations
- 1 year of data
- Small model
- Supervised learning (simpler than full RL)

Good for:
- Testing the setup
- Understanding the data flow
- Quick experimentation on a laptop
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data import WeatherDataLoader
from src.data.openmeteo import OpenMeteoClient
from src.models import LightweightWeatherModel
from src.utils import print_system_info, set_seed


def main():
    """Run a simple demo training loop."""
    print("=" * 50)
    print("RL Weather - Lightweight Demo")
    print("=" * 50)

    # Print system info
    print_system_info()

    # Configuration
    locations = OpenMeteoClient.get_sample_locations()[:3]
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    window_size = 7
    learning_rate = 1e-3
    epochs = 2

    print(f"\nDemo Configuration:")
    print(f"  Locations: {[loc['name'] for loc in locations]}")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Window size: {window_size} days")
    print(f"  Epochs: {epochs}")

    # Load data
    print("\n" + "=" * 50)
    print("Loading data...")
    print("=" * 50)

    data_loader = WeatherDataLoader(
        locations=locations,
        start_date=start_date,
        end_date=end_date,
        window_size=window_size,
        cache=True,
    )

    # Create model
    print("\n" + "=" * 50)
    print("Creating model...")
    print("=" * 50)

    model = LightweightWeatherModel(
        num_weather_vars=len(data_loader.variables),
        window_size=window_size,
        hidden_dim=64,
        num_layers=2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training
    print("\n" + "=" * 50)
    print("Training...")
    print("=" * 50)

    model.train()

    for epoch in range(epochs):
        epoch_losses = []
        pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for date, batch in pbar:
            # Convert to tensors
            history = torch.from_numpy(batch.history).float().to(device)
            current = torch.from_numpy(batch.current_forecast).float().to(device)
            target = torch.from_numpy(batch.target).float().to(device)

            # Forward pass
            prediction = model(history, current)
            loss = criterion(prediction, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch + 1} - Average loss: {avg_loss:.4f}")

    # Test prediction
    print("\n" + "=" * 50)
    print("Testing prediction...")
    print("=" * 50)

    model.eval()
    with torch.no_grad():
        # Get a sample batch
        date, batch = next(iter(data_loader))
        history = torch.from_numpy(batch.history).float().to(device)
        current = torch.from_numpy(batch.current_forecast).float().to(device)
        target = torch.from_numpy(batch.target).float().to(device)

        prediction = model(history, current)

        # Show first example
        print(f"\nSample prediction for {date.date()}:")
        print(f"  Input shape: {history.shape}")
        print(f"  Prediction: {prediction[0].cpu().numpy()}")
        print(f"  Target: {target[0].cpu().numpy()}")
        print(f"  Error: {(prediction[0] - target[0]).cpu().numpy()}")

    print("\n" + "=" * 50)
    print("Demo complete!")
    print("=" * 50)
    print("\nTo run full training:")
    print("  python train.py --config low_memory --epochs 1")
    print("\nTo test API connection:")
    print("  python test_api.py")


if __name__ == "__main__":
    set_seed(42)
    main()
