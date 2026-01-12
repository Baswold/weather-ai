# RL Weather

A framework for training weather prediction models using **sequential temporal reinforcement learning**.

## Overview

Unlike traditional supervised learning that trains on static shuffled datasets, this framework trains models to "live through" weather history chronologically. The model experiences each day from the earliest available data (1940-1950 with Open-Meteo ERA5/ERA5-Land) up to the present, learning and adapting as it goes.

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
- `production`: 20+ locations, 2010-2024, large model (15 years)
- `24gb`: 20+ locations, 2016-2024, very large model (~24GB RAM)
- `historical`: 20+ locations, 1950-2024, large model (75 years!)
- `climate`: 20+ locations, 1970-2024, large model (55 years, climate analysis)

## Data Sources

| Source | Type | Coverage | Notes |
|--------|------|----------|-------|
| [Open-Meteo Archive API](https://open-meteo.com/en/docs/historical-weather-api) | Reanalysis (ERA5/ERA5-Land) | Global, **1940-present** (80+ years!) | Free, no API key needed, 10-25km resolution |
| [NOAA NCEI](https://www.ncei.noaa.gov/) | Observations | US focused | Primarily actual weather |
| [NOAA NDFD](https://www.ncei.noaa.gov/products/weather-climate-models/national-digital-forecast-database) | Forecasts | US | National Digital Forecast Database |

**Note**: Open-Meteo provides:
- **ERA5** data from 1940 onwards (25km resolution)
- **ERA5-Land** data from 1950 onwards (10km resolution, higher quality)

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

| Config | RAM | GPU | Training Time | Data Range |
|--------|-----|-----|---------------|------------|
| low_memory | 2 GB | Optional | ~10 min | 2023 (1 year) |
| default | 4 GB | Optional | ~1 hour | 2020-2024 (5 years) |
| production | 8 GB+ | Optional | ~6-10 hours | 2010-2024 (15 years) |
| **24gb** | **24 GB** | **Optional** | **~8-12 hours** | **2016-2024 (9 years)** |
| **extended** | **24 GB** | **Optional** | **~12-24 hours** | **2016-2024 (100+ locations)** |
| **historical** | **32 GB+** | **Optional** | **~2-4 days** | **1950-2024 (75 years!)** |
| **climate** | **32 GB+** | **Optional** | **~1-2 days** | **1970-2024 (55 years)** |

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

### Historical Configuration (NEW!)

For maximum temporal coverage using 75 years of ERA5-Land data:

```bash
python train.py --config historical --epochs 2
```

**What makes this special:**
- **1950-2024**: 75 years of consistent global weather data
- **Climate learning**: Model learns multi-decadal patterns (El Niño cycles, PDO, AMO)
- **Extreme events**: Better coverage of rare weather events across decades
- **Trend detection**: Can learn long-term climate change signals

**Memory requirements:**
- ~32GB+ RAM for the full dataset
- Replay buffer: ~20GB (20M transitions)
- Training time: 2-4 days for 2 epochs

### Climate Analysis Configuration (NEW!)

For climate-focused research with 55 years of data:

```bash
python train.py --config climate --epochs 3
```

**Optimized for:**
- Climate change impact analysis (1970s onwards)
- Decadal oscillation patterns
- 30-day historical windows for monthly pattern recognition
- Balanced between coverage and computation

## Limitations

- **No physics knowledge**: Model learns purely from data, without explicit atmospheric physics
- **Single-location prediction**: Each location is predicted independently (no spatial relationships)
- **Data is reanalysis, not forecasts**: Open-Meteo provides ERA5 reanalysis (hindcasts) rather than original forecasts
- **Computational cost**: Training through 75 years of data requires significant compute and time

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
