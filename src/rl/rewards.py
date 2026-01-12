"""
Reward functions for RL-based weather prediction.

The reward signal guides the model to make better predictions.
Different reward functions can emphasize different aspects of prediction quality.
"""

import torch
import numpy as np
from typing import Dict, Optional
from abc import ABC, abstractmethod


class RewardFunction(ABC):
    """Base class for reward functions."""

    @abstractmethod
    def compute(
        self,
        prediction: torch.Tensor,
        actual: torch.Tensor,
        info: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Compute reward for a prediction.

        Args:
            prediction: (batch_size, num_vars) model's prediction
            actual: (batch_size, num_vars) ground truth
            info: Optional metadata (location, date, etc.)

        Returns:
            reward: (batch_size,) reward per example
        """
        pass


class MSEReward(RewardFunction):
    """
    Negative Mean Squared Error reward.

    Higher reward (less negative) = better prediction.
    Simple and commonly used.
    """

    def __init__(self, scale: float = 1.0):
        """
        Args:
            scale: Scaling factor for the reward
        """
        self.scale = scale

    def compute(
        self,
        prediction: torch.Tensor,
        actual: torch.Tensor,
        info: Optional[Dict] = None,
    ) -> torch.Tensor:
        """Compute negative MSE as reward."""
        mse = torch.mean((prediction - actual) ** 2, dim=-1)
        return -self.scale * mse


class MAEReward(RewardFunction):
    """Negative Mean Absolute Error reward."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def compute(
        self,
        prediction: torch.Tensor,
        actual: torch.Tensor,
        info: Optional[Dict] = None,
    ) -> torch.Tensor:
        """Compute negative MAE as reward."""
        mae = torch.mean(torch.abs(prediction - actual), dim=-1)
        return -self.scale * mae


class WeightedReward(RewardFunction):
    """
    Weighted reward that differentiates between weather variables.

    Different variables have different importance:
    - Temperature error of 5°C is different from wind error of 5 km/h
    - Precipitation accuracy might be more important than cloud cover
    """

    def __init__(
        self,
        variable_weights: Optional[Dict[str, float]] = None,
        num_vars: int = 6,
    ):
        """
        Args:
            variable_weights: Map of variable index to weight
            num_vars: Total number of variables
        """
        self.num_vars = num_vars

        # Default weights (can be overridden)
        if variable_weights is None:
            # Common weather variables with relative importance
            self.weights = torch.tensor([
                2.0,  # temperature min (very important)
                2.0,  # temperature max (very important)
                1.0,  # temperature mean
                3.0,  # precipitation (critical for many applications)
                1.5,  # wind speed max (important for safety)
                1.0,  # wind speed mean
            ][:num_vars])
        else:
            self.weights = torch.tensor([variable_weights.get(i, 1.0) for i in range(num_vars)])

        # Normalize weights
        self.weights = self.weights / self.weights.sum()

    def compute(
        self,
        prediction: torch.Tensor,
        actual: torch.Tensor,
        info: Optional[Dict] = None,
    ) -> torch.Tensor:
        """Compute weighted negative MSE."""
        # Move weights to same device
        weights = self.weights.to(prediction.device)

        # Per-variable squared error
        squared_error = (prediction - actual) ** 2

        # Weighted sum
        weighted_mse = (squared_error * weights.unsqueeze(0)).sum(dim=-1)

        return -weighted_mse


class ExtremeEventBonus(RewardFunction):
    """
    Reward function with bonus for correctly predicting extreme events.

    Extreme events (hurricanes, heat waves, heavy rain) are rare but important.
    This reward function gives extra credit for getting them right.
    """

    def __init__(
        self,
        base_reward: Optional[RewardFunction] = None,
        bonus_multiplier: float = 5.0,
        extreme_thresholds: Optional[Dict[int, tuple]] = None,
    ):
        """
        Args:
            base_reward: Base reward function (default: MSE)
            bonus_multiplier: How much to multiply reward for extreme events
            extreme_thresholds: Map of variable_idx to (low, high) thresholds
                               for defining "extreme"
        """
        self.base_reward = base_reward or MSEReward()
        self.bonus_multiplier = bonus_multiplier

        # Default extreme thresholds (normalized values)
        # For temperature: > 2 std from mean, or < -2 std
        # For precipitation: > 90th percentile
        if extreme_thresholds is None:
            self.extreme_thresholds = {
                # Temperature: extreme if > 2 or < -2 (assuming normalized)
                0: (-2.0, 2.0),  # temp min
                1: (-2.0, 2.0),  # temp max
                # Precipitation: extreme if > 2 (heavy rain)
                3: (None, 2.0),
            }
        else:
            self.extreme_thresholds = extreme_thresholds

    def _is_extreme(self, values: torch.Tensor, var_idx: int) -> torch.Tensor:
        """Check if values are extreme for a given variable."""
        if var_idx not in self.extreme_thresholds:
            return torch.zeros(len(values), dtype=torch.bool, device=values.device)

        low, high = self.extreme_thresholds[var_idx]
        is_extreme = torch.zeros(len(values), dtype=torch.bool, device=values.device)

        if low is not None:
            is_extreme |= values < low
        if high is not None:
            is_extreme |= values > high

        return is_extreme

    def compute(
        self,
        prediction: torch.Tensor,
        actual: torch.Tensor,
        info: Optional[Dict] = None,
    ) -> torch.Tensor:
        """Compute reward with extreme event bonus."""
        # Get base reward
        base = self.base_reward.compute(prediction, actual, info)

        # Check for extreme events in actual weather
        is_extreme = torch.zeros(len(actual), dtype=torch.bool, device=actual.device)
        for var_idx in self.extreme_thresholds.keys():
            if var_idx < actual.shape[1]:
                is_extreme |= self._is_extreme(actual[:, var_idx], var_idx)

        # Check if prediction was also extreme (correctly predicted extreme)
        # For simplicity, just check if actual was extreme
        bonus_mask = is_extreme.float()

        # Apply bonus
        reward = base * (1 + bonus_mask * (self.bonus_multiplier - 1))

        return reward


class SkillScoreReward(RewardFunction):
    """
    Reward based on forecast skill score.

    Compares model prediction to a baseline (e.g., persistence or climatology).
    Positive reward means model beats baseline.
    """

    def __init__(
        self,
        baseline: str = "persistence",
        scale: float = 10.0,
    ):
        """
        Args:
            baseline: "persistence", "climatology", or "forecast"
            scale: Scaling factor
        """
        self.baseline = baseline
        self.scale = scale

    def compute(
        self,
        prediction: torch.Tensor,
        actual: torch.Tensor,
        info: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Compute skill score reward.

        Skill score = 1 - (MSE_model / MSE_baseline)
        Higher is better.
        """
        model_mse = torch.mean((prediction - actual) ** 2, dim=-1)

        # Get baseline prediction
        if self.baseline == "persistence" and info is not None:
            # Persistence: today's weather is tomorrow's
            baseline_pred = info.get("previous_day_actual", actual)
        elif self.baseline == "forecast" and info is not None:
            # Use the original forecast as baseline
            baseline_pred = info.get("original_forecast", actual)
        else:
            # Fallback: use mean of actual
            baseline_pred = actual.mean(dim=-1, keepdim=True).expand_as(actual)

        baseline_mse = torch.mean((baseline_pred - actual) ** 2, dim=-1)

        # Skill score (with epsilon for numerical stability)
        eps = 1e-6
        skill_score = 1 - (model_mse / (baseline_mse + eps))

        return self.scale * skill_score


class ThresholdedReward(RewardFunction):
    """
    Reward that only cares if prediction is within acceptable threshold.

    Binary: 1 if within threshold, 0 otherwise.
    Can be used for "success rate" metrics.
    """

    def __init__(self, thresholds: Dict[int, float] = None, default_threshold: float = 2.0):
        """
        Args:
            thresholds: Map of variable index to acceptable error threshold
            default_threshold: Default threshold if not specified
        """
        self.thresholds = thresholds or {}
        self.default_threshold = default_threshold

    def compute(
        self,
        prediction: torch.Tensor,
        actual: torch.Tensor,
        info: Optional[Dict] = None,
    ) -> torch.Tensor:
        """Compute binary reward based on threshold."""
        error = torch.abs(prediction - actual)

        # Check each variable against its threshold
        within_threshold = torch.ones(len(prediction), device=prediction.device)

        for var_idx in range(prediction.shape[1]):
            threshold = self.thresholds.get(var_idx, self.default_threshold)
            within_threshold &= error[:, var_idx] < threshold

        return within_threshold.float()


class CombinedReward(RewardFunction):
    """
    Combine multiple reward functions with weights.

    Allows fine-tuned reward shaping.
    """

    def __init__(self, reward_functions: list[tuple[float, RewardFunction]]):
        """
        Args:
            reward_functions: List of (weight, reward_function) tuples
        """
        self.reward_functions = reward_functions

    def compute(
        self,
        prediction: torch.Tensor,
        actual: torch.Tensor,
        info: Optional[Dict] = None,
    ) -> torch.Tensor:
        """Compute combined weighted reward."""
        total_reward = torch.zeros(len(prediction), device=prediction.device)

        for weight, reward_fn in self.reward_functions:
            total_reward += weight * reward_fn.compute(prediction, actual, info)

        return total_reward
