"""
Metrics for evaluating weather prediction quality.

Provides various metrics beyond the reward signal for
model evaluation and comparison.
"""

import torch
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass, field


@dataclass
class PredictionMetrics:
    """Container for prediction metrics."""

    # Error metrics
    mse: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0

    # Correlation
    correlation: float = 0.0

    # Per-variable metrics
    per_variable_mse: Dict[str, float] = field(default_factory=dict)

    # Extreme event metrics
    extreme_hit_rate: float = 0.0
    extreme_false_alarm_rate: float = 0.0

    # Skill scores
    skill_vs_persistence: float = 0.0
    skill_vs_forecast: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "mse": self.mse,
            "mae": self.mae,
            "rmse": self.rmse,
            "correlation": self.correlation,
            "extreme_hit_rate": self.extreme_hit_rate,
            "extreme_false_alarm_rate": self.extreme_false_alarm_rate,
            "skill_vs_persistence": self.skill_vs_persistence,
            "skill_vs_forecast": self.skill_vs_forecast,
            **{f"var_{k}_mse": v for k, v in self.per_variable_mse.items()},
        }

    def __str__(self) -> str:
        """String representation."""
        parts = [f"MSE={self.mse:.4f}", f"MAE={self.mae:.4f}", f"RMSE={self.rmse:.4f}"]
        if self.correlation > 0:
            parts.append(f"r={self.correlation:.4f}")
        return ", ".join(parts)


def compute_metrics(
    predictions: torch.Tensor,
    actuals: torch.Tensor,
    variable_names: Optional[List[str]] = None,
    original_forecasts: Optional[torch.Tensor] = None,
    previous_actuals: Optional[torch.Tensor] = None,
    extreme_threshold: float = 2.0,
) -> PredictionMetrics:
    """
    Compute comprehensive prediction metrics.

    Args:
        predictions: (N, num_vars) model predictions
        actuals: (N, num_vars) ground truth
        variable_names: Names for each variable
        original_forecasts: (N, num_vars) original NWP forecasts
        previous_actuals: (N, num_vars) previous day's actuals (for persistence baseline)
        extreme_threshold: Threshold for defining extreme events

    Returns:
        PredictionMetrics object
    """
    # Convert to numpy for easier computation
    if isinstance(predictions, torch.Tensor):
        preds = predictions.detach().cpu().numpy()
        acts = actuals.detach().cpu().numpy()
    else:
        preds = predictions
        acts = actuals

    if original_forecasts is not None:
        orig = original_forecasts.detach().cpu().numpy() if isinstance(original_forecasts, torch.Tensor) else original_forecasts
    else:
        orig = None

    if previous_actuals is not None:
        prev = previous_actuals.detach().cpu().numpy() if isinstance(previous_actuals, torch.Tensor) else previous_actuals
    else:
        prev = None

    metrics = PredictionMetrics()

    # Basic error metrics
    errors = preds - acts
    metrics.mse = float(np.mean(errors ** 2))
    metrics.mae = float(np.mean(np.abs(errors)))
    metrics.rmse = float(np.sqrt(metrics.mse))

    # Correlation (flatten for overall correlation)
    if preds.size > 0:
        metrics.correlation = float(np.corrcoef(preds.flatten(), acts.flatten())[0, 1])

    # Per-variable MSE
    if variable_names:
        for i, name in enumerate(variable_names):
            if i < preds.shape[1]:
                metrics.per_variable_mse[name] = float(np.mean((preds[:, i] - acts[:, i]) ** 2))

    # Extreme event metrics
    # Define extreme as > threshold standard deviations from mean
    mean = np.mean(acts, axis=0)
    std = np.std(acts, axis=0) + 1e-6
    is_extreme = np.any(np.abs(acts - mean) / std > extreme_threshold, axis=1)

    if is_extreme.sum() > 0:
        # Hit rate: did we predict extreme when it was extreme?
        pred_extreme = np.any(np.abs(preds - mean) / std > extreme_threshold, axis=1)
        hits = np.sum(is_extreme & pred_extreme)
        metrics.extreme_hit_rate = float(hits / is_extreme.sum())

        # False alarm rate: predicted extreme but wasn't
        false_alarms = np.sum(~is_extreme & pred_extreme)
        metrics.extreme_false_alarm_rate = float(false_alarms / (~is_extreme).sum())

    # Skill scores
    model_mse = metrics.mse

    if prev is not None:
        persistence_mse = float(np.mean((prev - acts) ** 2))
        if persistence_mse > 1e-6:
            metrics.skill_vs_persistence = 1 - (model_mse / persistence_mse)

    if orig is not None:
        forecast_mse = float(np.mean((orig - acts) ** 2))
        if forecast_mse > 1e-6:
            metrics.skill_vs_forecast = 1 - (model_mse / forecast_mse)

    return metrics


class MetricsTracker:
    """
    Track metrics over time for analysis.

    Useful for monitoring learning progress and detecting issues.
    """

    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: Rolling window size for smoothed metrics
        """
        self.window_size = window_size
        self.history: Dict[str, List[float]] = {}

    def update(self, metrics: PredictionMetrics, step: int):
        """Update with new metrics."""
        metrics_dict = metrics.to_dict()
        for key, value in metrics_dict.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append((step, value))

    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get the most recent value for a metric."""
        if metric_name in self.history and self.history[metric_name]:
            return self.history[metric_name][-1][1]
        return None

    def get_average(self, metric_name: str, last_n: int = None) -> Optional[float]:
        """Get average of recent values."""
        if metric_name not in self.history or not self.history[metric_name]:
            return None

        values = [v for _, v in self.history[metric_name]]
        if last_n:
            values = values[-last_n:]

        return float(np.mean(values))

    def get_smoothed(self, metric_name: str) -> Optional[tuple[List[int], List[float]]]:
        """Get smoothed values over the rolling window."""
        if metric_name not in self.history or not self.history[metric_name]:
            return None, None

        data = self.history[metric_name]
        steps = [s for s, _ in data]
        values = [v for _, v in data]

        # Simple moving average
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - self.window_size)
            smoothed.append(np.mean(values[start:i+1]))

        return steps, smoothed

    def summary(self) -> Dict[str, float]:
        """Get summary of all metrics."""
        summary = {}
        for key in self.history:
            latest = self.get_latest(key)
            if latest is not None:
                summary[f"{key}_latest"] = latest
            average = self.get_average(key)
            if average is not None:
                summary[f"{key}_avg"] = average
        return summary
