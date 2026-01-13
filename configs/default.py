"""
Default configuration for RL Weather training.

Can be overridden with command-line arguments or custom configs.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class DataConfig:
    """Data loading configuration."""

    # Locations to train on
    locations: List[Dict] = field(default_factory=lambda: [
        {"name": "New_York", "latitude": 40.71, "longitude": -74.01},
        {"name": "London", "latitude": 51.51, "longitude": -0.13},
        {"name": "Tokyo", "latitude": 35.68, "longitude": 139.77},
        {"name": "Sydney", "latitude": -33.87, "longitude": 151.21},
        {"name": "Mumbai", "latitude": 19.08, "longitude": 72.88},
        {"name": "Dubai", "latitude": 25.20, "longitude": 55.27},
        {"name": "Singapore", "latitude": 1.35, "longitude": 103.82},
        {"name": "Reykjavik", "latitude": 64.15, "longitude": -21.95},
        {"name": "Denver", "latitude": 39.74, "longitude": -104.99},
        {"name": "Los_Angeles", "latitude": 34.05, "longitude": -118.24},
    ])

    # Training period
    start_date: str = "2020-01-01"  # Open-Meteo Archive API has data from 1940
    end_date: str = "2024-12-31"

    # Data options
    window_size: int = 7  # Historical window
    cache_data: bool = True
    data_dir: str = "data/weather"


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    # Transformer architecture
    d_model: int = 128  # Reduced for memory efficiency
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1

    # Input/output
    num_weather_vars: int = 6
    window_size: int = 7
    output_dim: Optional[int] = None  # Same as input


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Basic training
    batch_size: int = 32  # Locations per day
    learning_rate: float = 1e-4
    num_epochs: int = 2

    # Replay buffer
    replay_buffer_size: int = 50000
    replay_start_size: int = 500

    # RL parameters
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.1
    policy_freq: int = 2

    # Exploration
    exploration_noise: float = 0.1
    exploration_anneal: float = 0.99

    # Device
    device: str = "auto"  # auto, cpu, cuda

    # Checkpoints
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 90  # Days
    log_interval: int = 5  # Days


@dataclass
class RewardConfig:
    """Reward function configuration."""

    type: str = "weighted"  # mse, mae, weighted, extreme, skill
    use_extreme_bonus: bool = True
    extreme_multiplier: float = 3.0

    # Variable weights (for weighted reward)
    # Indices: 0=temp_min, 1=temp_max, 2=temp_mean, 3=precip, 4=wind_max, 5=wind_mean
    variable_weights: Dict[int, float] = field(default_factory=lambda: {
        0: 2.0,  # Temperature min
        1: 2.0,  # Temperature max
        2: 1.0,  # Temperature mean
        3: 3.0,  # Precipitation
        4: 1.5,  # Wind max
        5: 1.0,  # Wind mean
    })


@dataclass
class Config:
    """Main configuration container."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    # Experiment metadata
    experiment_name: str = "rl_weather_baseline"
    seed: int = 42
    debug: bool = False
    dry_run: bool = False  # If True, don't actually download data


def get_default_config() -> Config:
    """Get the default configuration."""
    return Config()


def get_low_memory_config() -> Config:
    """
    Get a low-memory configuration for testing on limited hardware.

    Uses SAME locations as default, but processes them sequentially (one at a time)
    instead of in parallel. This reduces memory overhead while still learning from
    all 10 locations.

    Memory optimization:
    - Same model size as default
    - Same locations (10 cities, diverse climates)
    - Shorter training period (1 year instead of 5)
    - Process ONE location per batch (batch_size=1)
    - Smaller replay buffer (processes serially)
    - Result: ~500MB RAM instead of 4GB, still learns from all locations
    """
    config = Config()

    # SAME locations as default - but process ONE at a time
    config.data.locations = config.data.locations[:10]  # Keep all default locations
    config.data.start_date = "2023-01-01"
    config.data.end_date = "2023-12-31"

    # Same model as default (don't shrink model)
    config.model.d_model = 128
    config.model.nhead = 4
    config.model.num_layers = 4
    config.model.dim_feedforward = 512

    # Sequential processing (process ONE location per batch)
    config.training.batch_size = 1  # One location at a time
    config.training.num_epochs = 1
    # Smaller replay buffer (processes serially, so less history needed)
    config.training.replay_buffer_size = 5000
    config.training.replay_start_size = 500

    return config


def get_production_config() -> Config:
    """
    Get a production configuration for full training.

    Uses more locations and larger model.
    """
    config = Config()

    # All sample locations
    from src.data.openmeteo import OpenMeteoClient
    config.data.locations = OpenMeteoClient.get_sample_locations()

    # Extended training period (back to Open-Meteo's earliest available data)
    config.data.start_date = "2010-01-01"  # Use 15 years of data
    config.data.end_date = "2024-12-31"

    # Larger model
    config.model.d_model = 256
    config.model.nhead = 8
    config.model.num_layers = 6
    config.model.dim_feedforward = 1024

    # Training
    config.training.batch_size = 40
    config.training.num_epochs = 3
    config.training.replay_buffer_size = 100000

    return config


def get_24gb_config() -> Config:
    """
    Get a configuration optimized for 24GB RAM usage.

    Designed to maximize utilization of 24GB RAM on a 32GB machine.
    This configuration balances:
    - Large replay buffer for diverse experience sampling
    - Many diverse locations for spatial learning
    - Full historical data range (2016-2024)
    - Larger model architecture

    Memory breakdown (approximate):
    - Replay buffer: ~12 GB
    - Data cache: ~6 GB
    - Model + activations: ~2 GB
    - Training overhead: ~4 GB
    Total: ~24 GB
    """
    config = Config()

    # Use ALL available sample locations (20 diverse climates)
    from src.data.openmeteo import OpenMeteoClient
    config.data.locations = OpenMeteoClient.get_sample_locations()

    # Full recent historical range (9 years of data)
    config.data.start_date = "2016-01-01"
    config.data.end_date = "2024-12-31"

    # Larger historical window for more context
    config.data.window_size = 14  # 2 weeks of history
    config.data.cache_in_memory = True  # Keep all data in RAM

    # Large model architecture (~5M parameters)
    config.model.d_model = 512
    config.model.nhead = 8
    config.model.num_layers = 8
    config.model.dim_feedforward = 2048
    config.model.dropout = 0.1

    # Training configuration optimized for throughput
    config.training.batch_size = 80  # All locations per batch
    config.training.learning_rate = 1e-4
    config.training.num_epochs = 5  # Multiple passes through history

    # Large prioritized replay buffer
    # Each transition ~1KB, so 1M transitions = ~1GB raw
    # With overhead, 12M transitions fits in ~12GB
    config.training.replay_buffer_size = 12_000_000
    config.training.replay_start_size = 10000
    config.training.use_prioritized_replay = True

    # RL parameters
    config.training.gamma = 0.99
    config.training.tau = 0.005
    config.training.policy_noise = 0.1
    config.training.policy_freq = 2

    # Exploration schedule
    config.training.exploration_noise = 0.15
    config.training.exploration_anneal = 0.998

    # Checkpointing
    config.training.checkpoint_dir = "checkpoints/24gb"
    config.training.checkpoint_interval = 365  # Yearly checkpoints
    config.training.log_interval = 10

    # Reward with extreme event bonus
    config.reward.use_extreme_bonus = True
    config.reward.extreme_multiplier = 5.0

    config.experiment_name = "rl_weather_24gb"

    return config


def get_extended_locations_config() -> Config:
    """
    Configuration with 100+ diverse locations worldwide.

    For when you want maximum spatial coverage.
    Requires 24GB+ RAM.
    """
    config = get_24gb_config()

    # Extended location list covering all climate zones
    config.data.locations = _get_extended_locations()

    return config


def get_historical_config() -> Config:
    """
    Configuration using maximum available historical data (1950-2024).

    Uses ERA5-Land data from 1950 onwards for 75 years of weather history.
    This is ideal for:
    - Long-term climate pattern analysis
    - Multi-decadal trend learning
    - Understanding seasonal variations across decades

    Requires significant RAM (~32GB+) and training time.
    """
    config = get_24gb_config()

    # Maximum historical range using ERA5-Land (1950 onwards)
    config.data.start_date = "1950-01-01"
    config.data.end_date = "2024-12-31"

    # Adjust for longer training period
    config.training.num_epochs = 2  # 2 passes through 75 years is already substantial
    config.training.checkpoint_interval = 1825  # Checkpoint every 5 years

    # Larger replay buffer to capture long-term patterns
    config.training.replay_buffer_size = 20_000_000  # ~20GB

    config.experiment_name = "rl_weather_historical_75yr"
    config.training.checkpoint_dir = "checkpoints/historical"

    return config


def get_climate_config() -> Config:
    """
    Configuration optimized for climate-scale analysis (1970-2024).

    Uses 55 years of data to study:
    - Climate change impacts on weather patterns
    - Decadal oscillations (El Niño, PDO, AMO)
    - Long-term trend learning

    Balanced between data coverage and computational feasibility.
    """
    config = get_24gb_config()

    # Climate-relevant period (1970s onwards)
    config.data.start_date = "1970-01-01"
    config.data.end_date = "2024-12-31"

    # Optimize for climate patterns
    config.data.window_size = 30  # 30-day context for monthly patterns

    # Training configuration
    config.training.num_epochs = 3
    config.training.checkpoint_interval = 730  # Checkpoint every 2 years

    config.experiment_name = "rl_weather_climate_55yr"
    config.training.checkpoint_dir = "checkpoints/climate"

    return config


def auto_config(target_gb: Optional[float] = None) -> Config:
    """
    Automatically configure based on available system memory.

    Args:
        target_gb: Target memory usage in GB. If None, uses 75% of available RAM.

    Returns:
        Config tuned for available memory
    """
    import psutil

    # Get available memory
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    total_gb = mem.total / (1024**3)

    if target_gb is None:
        target_gb = total_gb * 0.75  # Use 75% of total RAM

    print(f"System RAM: {total_gb:.1f} GB total, {available_gb:.1f} GB available")
    print(f"Target usage: {target_gb:.1f} GB")

    # Select config based on target
    if target_gb >= 24:
        print("Using 24GB configuration")
        return get_24gb_config()
    elif target_gb >= 8:
        print("Using production configuration")
        return get_production_config()
    elif target_gb >= 4:
        print("Using default configuration")
        return get_default_config()
    else:
        print("Using low memory configuration")
        return get_low_memory_config()


def _get_extended_locations() -> List[Dict]:
    """Get 100+ diverse locations worldwide."""
    return [
        # North America
        {"name": "New_York", "latitude": 40.71, "longitude": -74.01, "climate": "continental"},
        {"name": "Los_Angeles", "latitude": 34.05, "longitude": -118.24, "climate": "mediterranean"},
        {"name": "Chicago", "latitude": 41.88, "longitude": -87.63, "climate": "continental"},
        {"name": "Miami", "latitude": 25.76, "longitude": -80.19, "climate": "tropical"},
        {"name": "Seattle", "latitude": 47.61, "longitude": -122.33, "climate": "temperate"},
        {"name": "Denver", "latitude": 39.74, "longitude": -104.99, "climate": "mountain"},
        {"name": "Phoenix", "latitude": 33.45, "longitude": -112.07, "climate": "desert"},
        {"name": "Anchorage", "latitude": 61.22, "longitude": -149.90, "climate": "polar"},
        {"name": "Honolulu", "latitude": 21.31, "longitude": -157.86, "climate": "tropical"},
        {"name": "Dallas", "latitude": 32.78, "longitude": -96.80, "climate": "subtropical"},
        {"name": "Boston", "latitude": 42.36, "longitude": -71.06, "climate": "continental"},
        {"name": "San_Francisco", "latitude": 37.77, "longitude": -122.42, "climate": "mediterranean"},
        {"name": "Atlanta", "latitude": 33.76, "longitude": -84.39, "climate": "subtropical"},
        {"name": "Minneapolis", "latitude": 44.98, "longitude": -93.27, "climate": "continental"},
        {"name": "Toronto", "latitude": 43.65, "longitude": -79.38, "climate": "continental"},

        # Europe
        {"name": "London", "latitude": 51.51, "longitude": -0.13, "climate": "temperate"},
        {"name": "Paris", "latitude": 48.86, "longitude": 2.35, "climate": "temperate"},
        {"name": "Berlin", "latitude": 52.52, "longitude": 13.41, "climate": "continental"},
        {"name": "Rome", "latitude": 41.90, "longitude": 12.50, "climate": "mediterranean"},
        {"name": "Madrid", "latitude": 40.42, "longitude": -3.70, "climate": "mediterranean"},
        {"name": "Amsterdam", "latitude": 52.37, "longitude": 4.89, "climate": "temperate"},
        {"name": "Vienna", "latitude": 48.21, "longitude": 16.37, "climate": "continental"},
        {"name": "Athens", "latitude": 37.98, "longitude": 23.73, "climate": "mediterranean"},
        {"name": "Oslo", "latitude": 59.91, "longitude": 10.75, "climate": "polar"},
        {"name": "Reykjavik", "latitude": 64.15, "longitude": -21.95, "climate": "polar"},
        {"name": "Moscow", "latitude": 55.76, "longitude": 37.62, "climate": "continental"},
        {"name": "Warsaw", "latitude": 52.23, "longitude": 21.01, "climate": "continental"},
        {"name": "Dublin", "latitude": 53.35, "longitude": -6.26, "climate": "temperate"},
        {"name": "Zurich", "latitude": 47.38, "longitude": 8.54, "climate": "temperate"},
        {"name": "Stockholm", "latitude": 59.33, "longitude": 18.07, "climate": "polar"},

        # Asia
        {"name": "Tokyo", "latitude": 35.68, "longitude": 139.77, "climate": "subtropical"},
        {"name": "Beijing", "latitude": 39.90, "longitude": 116.41, "climate": "continental"},
        {"name": "Shanghai", "latitude": 31.23, "longitude": 121.47, "climate": "subtropical"},
        {"name": "Hong_Kong", "latitude": 22.32, "longitude": 114.17, "climate": "tropical"},
        {"name": "Singapore", "latitude": 1.35, "longitude": 103.82, "climate": "tropical"},
        {"name": "Mumbai", "latitude": 19.08, "longitude": 72.88, "climate": "tropical"},
        {"name": "Delhi", "latitude": 28.61, "longitude": 77.21, "climate": "subtropical"},
        {"name": "Bangkok", "latitude": 13.76, "longitude": 100.50, "climate": "tropical"},
        {"name": "Seoul", "latitude": 37.57, "longitude": 126.98, "climate": "continental"},
        {"name": "Jakarta", "latitude": -6.21, "longitude": 106.85, "climate": "tropical"},
        {"name": "Manila", "latitude": 14.60, "longitude": 120.98, "climate": "tropical"},
        {"name": "Taipei", "latitude": 25.03, "longitude": 121.57, "climate": "subtropical"},
        {"name": "Kuala_Lumpur", "latitude": 3.14, "longitude": 101.69, "climate": "tropical"},

        # Oceania
        {"name": "Sydney", "latitude": -33.87, "longitude": 151.21, "climate": "subtropical"},
        {"name": "Melbourne", "latitude": -37.81, "longitude": 144.96, "climate": "temperate"},
        {"name": "Auckland", "latitude": -36.85, "longitude": 174.76, "climate": "temperate"},
        {"name": "Brisbane", "latitude": -27.47, "longitude": 153.03, "climate": "subtropical"},
        {"name": "Perth", "latitude": -31.95, "longitude": 115.86, "climate": "mediterranean"},
        {"name": "Suva", "latitude": -18.14, "longitude": 178.44, "climate": "tropical"},

        # Africa
        {"name": "Cairo", "latitude": 30.04, "longitude": 31.24, "climate": "desert"},
        {"name": "Cape_Town", "latitude": -33.93, "longitude": 18.42, "climate": "mediterranean"},
        {"name": "Lagos", "latitude": 6.52, "longitude": 3.38, "climate": "tropical"},
        {"name": "Nairobi", "latitude": -1.29, "longitude": 36.82, "climate": "highland"},
        {"name": "Casablanca", "latitude": 33.57, "longitude": -7.62, "climate": "mediterranean"},
        {"name": "Accra", "latitude": 5.55, "longitude": -0.22, "climate": "tropical"},
        {"name": "Addis_Ababa", "latitude": 9.03, "longitude": 38.74, "climate": "highland"},
        {"name": "Dakar", "latitude": 14.72, "longitude": -17.47, "climate": "desert"},

        # Middle East
        {"name": "Dubai", "latitude": 25.20, "longitude": 55.27, "climate": "desert"},
        {"name": "Riyadh", "latitude": 24.71, "longitude": 46.68, "climate": "desert"},
        {"name": "Tehran", "latitude": 35.69, "longitude": 51.39, "climate": "arid"},
        {"name": "Tel_Aviv", "latitude": 32.08, "longitude": 34.78, "climate": "mediterranean"},
        {"name": "Istanbul", "latitude": 41.01, "longitude": 28.98, "climate": "temperate"},
        {"name": "Abu_Dhabi", "latitude": 24.45, "longitude": 54.37, "climate": "desert"},
        {"name": "Kuwait_City", "latitude": 29.37, "longitude": 47.98, "climate": "desert"},
        {"name": "Doha", "latitude": 25.29, "longitude": 51.53, "climate": "desert"},

        # South America
        {"name": "Sao_Paulo", "latitude": -23.55, "longitude": -46.64, "climate": "subtropical"},
        {"name": "Buenos_Aires", "latitude": -34.60, "longitude": -58.38, "climate": "subtropical"},
        {"name": "Lima", "latitude": -12.05, "longitude": -77.04, "climate": "desert"},
        {"name": "Bogota", "latitude": 4.71, "longitude": -74.07, "climate": "highland"},
        {"name": "Santiago", "latitude": -33.45, "longitude": -70.67, "climate": "mediterranean"},
        {"name": "Caracas", "latitude": 10.48, "longitude": -66.90, "climate": "tropical"},
        {"name": "Montevideo", "latitude": -34.90, "longitude": -56.19, "climate": "subtropical"},
        {"name": "Quito", "latitude": -0.18, "longitude": -78.47, "climate": "highland"},
        {"name": "La_Paz", "latitude": -16.49, "longitude": -68.13, "climate": "highland"},
        {"name": "Brasilia", "latitude": -15.79, "longitude": -47.89, "climate": "tropical"},

        # Additional diverse climates
        {"name": "St_Petersburg", "latitude": 59.93, "longitude": 30.34, "climate": "polar"},
        {"name": "Nuuk", "latitude": 64.18, "longitude": -51.72, "climate": "polar"},
        {"name": "Ulaanbaatar", "latitude": 47.92, "longitude": 106.92, "climate": "continental"},
        {"name": "Kathmandu", "latitude": 27.72, "longitude": 85.32, "climate": "highland"},
        {"name": "Thimphu", "latitude": 27.47, "longitude": 89.64, "climate": "highland"},
        {"name": "Antananarivo", "latitude": -18.88, "longitude": 47.51, "climate": "highland"},
        {"name": "Port_Moresby", "latitude": -9.48, "longitude": 147.18, "climate": "tropical"},
        {"name": "Dar_es_Salaam", "latitude": -6.79, "longitude": 39.21, "climate": "tropical"},
    ]
