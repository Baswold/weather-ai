"""
Configuration system for RL Weather.
"""

from .default import (
    Config,
    get_default_config,
    get_low_memory_config,
    get_production_config,
    get_24gb_config,
    get_extended_locations_config,
    get_historical_config,
    get_climate_config,
    auto_config,
)

__all__ = [
    "Config",
    "get_default_config",
    "get_low_memory_config",
    "get_production_config",
    "get_24gb_config",
    "get_extended_locations_config",
    "get_historical_config",
    "get_climate_config",
    "auto_config",
]
