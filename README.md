# RL Weather

A framework for training weather prediction models using **sequential temporal reinforcement learning**.

## Overview

Unlike traditional supervised learning that trains on static shuffled datasets, this framework trains models to "live through" weather history chronologically. The model experiences each day from the earliest available forecast data (2016 with Open-Meteo) up to the present, learning and adapting as it goes.

### Key Features

- **Sequential temporal training**: Model experiences weather history in chronological order
- **Chronological causality**: Never sees future data when predicting the past
- **Continual learning**: Can continue learning from live weather data in production
- **Actor-critic RL**: Uses TD3-style training for continuous action spaces
- **Multiple data sources**: Integrated with Open-Meteo Historical Forecast API

## Project Structure

```
rl-weather/
├── src/
│   ├── data/              # Data loading and preprocessing
│   │   ├── openmeteo.py   # Open-Meteo API client
│   │   ├── loader.py      # Sequential data loader
│   │   └── replay_buffer.py  # Experience replay buffers
│   ├── models/            # Model architectures
│   │   ├── transformer.py # Transformer-based predictor
│   │   └── actor_critic.py # Actor-critic for RL
│   ├── rl/                # RL training framework
│   │   ├── rewards.py     # Reward functions
│   │   ├── metrics.py     # Evaluation metrics
│   │   └── trainer.py     # Main training loop
│   └── utils.py           # Utilities
├── configs/               # Configuration presets
├── data/                  # Data cache
├── checkpoints/           # Model checkpoints
├── train.py               # Main training script
├── demo.py                # Lightweight demo
└── test_api.py            # API connection test
```

## Installation

```bash
# Clone the repository
cd rl-weather

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Test the API connection

First, verify that the Open-Meteo API works:

```bash
python test_api.py
```

This will fetch a small sample of weather data and display the format.

### 2. Run the lightweight demo

For testing on limited hardware (laptops, small RAM):

```bash
python demo.py
```

This runs a simple supervised training loop with:
- 3 locations
- 1 year of data (2023)
- Small model (~50K parameters)

### 3. Full training

For a proper RL training run:

```bash
python train.py --config low_memory --epochs 1
```

Configuration presets:
- `low_memory`: 3 locations, 1 year, small model (for testing)
- `default`: 10 locations, 5 years, medium model
- `production`: 20+ locations, 2016-2024, large model

## Data Sources

| Source | Type | Coverage | Notes |
|--------|------|----------|-------|
| [Open-Meteo Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api) | Forecast + Actual | Global, 2016+ | Free, no API key needed |
| [NOAA NCEI](https://www.ncei.noaa.gov/) | Observations | US focused | Primarily actual weather |
| [NOAA NDFD](https://www.ncei.noaa.gov/products/weather-climate-models/national-digital-forecast-database) | Forecasts | US | National Digital Forecast Database |

## Model Architecture

```
Input (Historical Window)
├── Day -7: [forecast, actual]
├── Day -6: [forecast, actual]
├── ...
├── Day -1: [forecast, actual]
└── Day 0:  [forecast]  ← What we're predicting

        ↓

    Transformer Encoder
    (Self-attention over history)

        ↓

    Prediction Head
        ↓

    Output: Tomorrow's Weather
```

## Training Approach

The model trains **sequentially through time**:

```python
for year in [2016, 2017, ..., 2024]:
    for day in year:
        for location in locations:
            # Get historical context
            state = get_state(location, day, window=7)

            # Model predicts tomorrow
            prediction = model(state)

            # Get actual weather
            actual = get_actual(location, day + 1)

            # Calculate reward
            reward = -mse(prediction, actual)

            # Store and update
            replay_buffer.add(state, prediction, reward)

        # Batch update across locations
        model.update(replay_buffer.sample(batch_size=40))
```

## Hardware Requirements

| Config | RAM | GPU | Training Time |
|--------|-----|-----|---------------|
| low_memory | 2 GB | Optional | ~10 min |
| default | 4 GB | Optional | ~1 hour |
| production | 8 GB+ | Optional | ~4-8 hours |
| **24gb** | **24 GB** | **Optional** | **~8-12 hours** |
| **extended** | **24 GB** | **Optional** | **~12-24 hours** |

### 24GB Configuration

For systems with 32GB RAM, the `--config 24gb` preset is optimized to use ~24GB:

```bash
python train.py --config 24gb --epochs 5
```

**Memory breakdown:**
- Replay buffer: ~12 GB (12M prioritized transitions)
- Data cache: ~6 GB (full 2016-2024 dataset)
- Model + activations: ~2 GB
- Training overhead: ~4 GB

**Features:**
- 20 diverse locations worldwide
- Full historical range (2016-2024)
- 14-day historical window
- Large transformer (d_model=512, 8 layers)
- Prioritized experience replay
- Extreme event bonus rewards

### Extended Configuration

For maximum spatial coverage (100+ locations):

```bash
python train.py --config extended --epochs 5
```

This covers all climate zones: tropical, desert, temperate, continental, polar, mountain, and highland climates across 6 continents.

## Limitations

- **No physics knowledge**: Model learns purely from data, without explicit atmospheric physics
- **Single-location prediction**: Each location is predicted independently (no spatial relationships)
- **Historical availability**: Forecast data available from ~2016 with Open-Meteo
- **Computational cost**: Training through decades of data requires significant compute

## Future Extensions

- Multi-location modeling with graph neural networks
- Probabilistic forecasting with distributional RL
- Physics-informed constraints
- Spatial attention mechanisms
- Multi-step prediction (3-day, 7-day forecasts)

## References

- GenCast (Google DeepMind, 2024): State-of-the-art ML weather forecasting
- RAIN (2024): RL for tuning climate model parameters
- Time-R1 (2025): RL with LLMs for time series forecasting

## License

MIT
