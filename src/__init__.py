"""
RL Weather - Reinforcement Learning for Weather Prediction

A framework for training weather prediction models using sequential temporal RL.
"""

__version__ = "0.1.0"

from .data import WeatherDataLoader, WeatherBatch, OpenMeteoClient
from .models import WeatherTransformer, TransformerConfig, ActorCritic
from .rl import RLTrainer, TrainerConfig, RewardFunction, MSEReward
