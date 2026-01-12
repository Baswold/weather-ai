"""
RL Trainer for weather prediction.

Implements the main training loop for sequential temporal RL.
The trainer experiences weather history chronologically, learning
to improve predictions as it goes through time.
"""

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from ..models import ActorCritic, TransformerConfig
from ..data import WeatherDataLoader, WeatherBatch, PrioritizedReplayBuffer
from .rewards import RewardFunction, MSEReward, WeightedReward
from .metrics import PredictionMetrics, compute_metrics, MetricsTracker


@dataclass
class TrainerConfig:
    """Configuration for the RL trainer."""

    # Model config
    model_config: TransformerConfig = field(default_factory=lambda: TransformerConfig(
        d_model=256,
        nhead=8,
        num_layers=6,
        num_weather_vars=6,
        window_size=7,
    ))

    # Training hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 40
    replay_buffer_size: int = 100000
    replay_start_size: int = 1000

    # RL-specific
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005  # Soft update coefficient for target networks
    policy_noise: float = 0.1  # TD3 noise
    noise_clip: float = 0.3  # TD3 noise clipping
    policy_freq: int = 2  # TD3 policy update frequency

    # Exploration
    exploration_noise: float = 0.1
    exploration_anneal: float = 0.995  # Decay per epoch

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    log_interval: int = 10  # Days between logs
    checkpoint_interval: int = 365  # Days between checkpoints
    checkpoint_dir: str = "checkpoints"

    # Reward
    use_weighted_reward: bool = True

    def __post_init__(self):
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"


class RLTrainer:
    """
    Reinforcement Learning trainer for weather prediction.

    Training procedure:
    1. For each day in chronological order:
       a. Collect experiences from all locations
       b. Add to replay buffer
       c. Update model using sampled batch
    2. Track metrics and log progress
    3. Periodically save checkpoints
    """

    def __init__(
        self,
        model: ActorCritic,
        reward_function: Optional[RewardFunction] = None,
        config: Optional[TrainerConfig] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: The actor-critic model
            reward_function: Reward function to use
            config: Training configuration
        """
        self.config = config or TrainerConfig()
        self.device = torch.device(self.config.device)

        # Move model to device
        self.model = model.to(self.device)
        self.model.train()

        # Target networks (for TD3)
        self.target_model = ActorCritic(
            self.config.model_config,
            use_twin_critic=True,
        ).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        # Reward function
        if reward_function is None:
            if self.config.use_weighted_reward:
                self.reward_function = WeightedReward(
                    num_vars=self.config.model_config.num_weather_vars
                )
            else:
                self.reward_function = MSEReward()
        else:
            self.reward_function = reward_function

        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.model.actor.parameters(),
            lr=self.config.learning_rate,
        )
        self.critic_optimizer = optim.Adam(
            self.model.critic.parameters(),
            lr=self.config.learning_rate,
        )

        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.config.replay_buffer_size
        )

        # Training state
        self.current_step = 0
        self.current_exploration_noise = self.config.exploration_noise
        self.metrics_tracker = MetricsTracker()

        # Create checkpoint directory
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"Trainer initialized on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def train(
        self,
        data_loader: WeatherDataLoader,
        num_epochs: int = 1,
        start_date: Optional[datetime] = None,
    ) -> Dict[str, List[float]]:
        """
        Main training loop through weather history.

        Args:
            data_loader: Provides sequential weather data
            num_epochs: Number of passes through history
            start_date: Resume from this date

        Returns:
            Training history (rewards, metrics, etc.)
        """
        history = {
            "dates": [],
            "rewards": [],
            "actor_losses": [],
            "critic_losses": [],
        }

        for epoch in range(num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

            # Determine start point
            day_iterator = list(data_loader)
            if start_date and epoch == 0:
                day_iterator = [(d, b) for d, b in day_iterator if d >= start_date]

            epoch_rewards = []

            for date, batch in tqdm(day_iterator, desc=f"Epoch {epoch + 1}"):
                # Train on this day's batch
                metrics = self.train_day(batch)

                # Track history
                history["dates"].append(date)
                history["rewards"].append(metrics["reward"])
                history["actor_losses"].append(metrics.get("actor_loss", 0))
                history["critic_losses"].append(metrics.get("critic_loss", 0))
                epoch_rewards.append(metrics["reward"])

                # Logging
                if self.current_step % self.config.log_interval == 0:
                    self._log_progress(date, metrics)

                # Checkpointing
                if self.current_step % self.config.checkpoint_interval == 0:
                    self.save_checkpoint(f"checkpoint_epoch{epoch}_step{self.current_step}")

                self.current_step += 1

            # End of epoch
            avg_reward = np.mean(epoch_rewards)
            print(f"\nEpoch {epoch + 1} complete. Average reward: {avg_reward:.4f}")

            # Anneal exploration
            self.current_exploration_noise *= self.config.exploration_anneal

        return history

    def train_day(self, batch: WeatherBatch) -> Dict[str, float]:
        """
        Train on a single day's batch from all locations.

        Args:
            batch: WeatherBatch with data from multiple locations

        Returns:
            Training metrics for this day
        """
        # Convert to tensors
        history = torch.from_numpy(batch.history).float().to(self.device)
        current = torch.from_numpy(batch.current_forecast).float().to(self.device)
        target = torch.from_numpy(batch.target).float().to(self.device)

        batch_size = history.shape[0]

        # Get predictions (actions) from actor
        with torch.no_grad():
            # Deterministic actions for evaluation
            predictions = self.model.get_action(
                history, current, deterministic=True
            )

        # Compute rewards
        rewards = self.reward_function.compute(predictions, target)
        avg_reward = rewards.mean().item()

        # Add to replay buffer
        for i in range(batch_size):
            self.replay_buffer.push(
                state=torch.cat([history[i].flatten(), current[i].flatten()]).cpu().numpy(),
                action=predictions[i].cpu().numpy(),
                reward=rewards[i].item(),
            )

        # Update model if we have enough samples
        actor_loss = 0
        critic_loss = 0

        if self.replay_buffer.is_ready(self.config.batch_size):
            actor_loss, critic_loss = self._update()

        return {
            "reward": avg_reward,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
        }

    def _update(self) -> tuple[float, float]:
        """
        Perform gradient update using sampled batch.

        Returns:
            (actor_loss, critic_loss)
        """
        # Sample from replay buffer
        transitions, indices, weights = self.replay_buffer.sample(self.config.batch_size)

        # Convert to tensors
        states = torch.from_numpy(
            np.stack([t.state for t in transitions])
        ).float().to(self.device)
        actions = torch.from_numpy(
            np.stack([t.action for t in transitions])
        ).float().to(self.device)
        rewards = torch.from_numpy(
            np.array([t.reward for t in transitions])
        ).float().to(self.device)

        # Reshape states (need to recover history and current_forecast)
        # State was flattened: history (window * vars * 2) + current (vars)
        window_size = self.config.model_config.window_size
        num_vars = self.config.model_config.num_weather_vars
        history_dim = window_size * num_vars * 2

        history = states[:, :history_dim].view(-1, window_size, num_vars * 2)
        current = states[:, history_dim:]

        # Reconstruct current forecast dimension (assume [forecast, actual] -> just use first half)
        current_forecast = current[:, :num_vars]

        # Critic update
        self.critic_optimizer.zero_grad()

        # Current Q-values
        current_q = self.model.get_q_value(history, current_forecast, actions)

        # Target Q-values (using target network)
        with torch.no_grad():
            # Target actions (with noise for TD3)
            target_actions = self.target_model.get_action(
                history, current_forecast,
                deterministic=False,
                noise_scale=self.config.policy_noise,
            )
            target_actions = target_actions.clamp(-self.config.noise_clip, self.config.noise_clip)

            # Target Q
            target_q = self.target_model.get_target_q(history, current_forecast, target_actions)
            target_q_value = rewards.unsqueeze(1) + self.config.gamma * target_q

        # Critic loss
        critic_loss = nn.MSELoss()(current_q, target_q_value)
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update (delayed for TD3)
        actor_loss = 0
        if self.current_step % self.config.policy_freq == 0:
            self.actor_optimizer.zero_grad()

            # Actor wants to maximize Q
            new_actions = self.model.get_action(history, current_forecast, deterministic=True)
            actor_loss = -self.model.get_q_value(history, current_forecast, new_actions).mean()

            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update target networks
            self._soft_update()

        # Update priorities (for prioritized replay)
        with torch.no_grad():
            td_errors = (current_q - target_q_value).squeeze().cpu().numpy()
            self.replay_buffer.update_priorities(indices, td_errors)

        return actor_loss.item(), critic_loss.item()

    def _soft_update(self):
        """Soft update target networks: target = tau * model + (1-tau) * target"""
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )

    def _log_progress(self, date: datetime, metrics: Dict[str, float]):
        """Log training progress."""
        print(
            f"Day {date.strftime('%Y-%m-%d')}: "
            f"Reward={metrics['reward']:.4f}, "
            f"Actor Loss={metrics.get('actor_loss', 0):.4f}, "
            f"Critic Loss={metrics.get('critic_loss', 0):.4f}"
        )

    def evaluate(
        self,
        data_loader: WeatherDataLoader,
        num_days: int = 30,
    ) -> PredictionMetrics:
        """
        Evaluate model on validation data.

        Args:
            data_loader: Validation data loader
            num_days: Number of days to evaluate

        Returns:
            Aggregated prediction metrics
        """
        self.model.eval()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for i, (date, batch) in enumerate(data_loader):
                if i >= num_days:
                    break

                history = torch.from_numpy(batch.history).float().to(self.device)
                current = torch.from_numpy(batch.current_forecast).float().to(self.device)
                target = batch.target

                predictions = self.model(history, current)

                all_predictions.append(predictions.cpu())
                all_targets.append(torch.from_numpy(target))

        # Combine and compute metrics
        combined_preds = torch.cat(all_predictions, dim=0)
        combined_targets = torch.cat(all_targets, dim=0)

        metrics = compute_metrics(combined_preds, combined_targets)

        self.model.train()
        return metrics

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        path = self.checkpoint_dir / f"{name}.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "target_model_state_dict": self.target_model.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "current_step": self.current_step,
            "exploration_noise": self.current_exploration_noise,
            "config": self.config,
        }, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.current_step = checkpoint["current_step"]
        self.current_exploration_noise = checkpoint["exploration_noise"]
        print(f"Checkpoint loaded: {path}")


class LightweightTrainer:
    """
    Simplified trainer for low-resource environments.

    Uses a simple supervised learning approach as a baseline
    before attempting full RL training.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.current_step = 0

    def train(
        self,
        data_loader: WeatherDataLoader,
        num_epochs: int = 1,
    ) -> Dict[str, List[float]]:
        """Simple supervised training loop."""
        history = {"losses": []}

        self.model.train()

        for epoch in range(num_epochs):
            epoch_losses = []

            for date, batch in tqdm(data_loader, desc=f"Epoch {epoch + 1}"):
                # Convert to tensors
                history_tensor = torch.from_numpy(batch.history).float().to(self.device)
                current = torch.from_numpy(batch.current_forecast).float().to(self.device)
                target = torch.from_numpy(batch.target).float().to(self.device)

                # Forward pass
                prediction = self.model(history_tensor, current)
                loss = nn.MSELoss()(prediction, target)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())
                self.current_step += 1

            avg_loss = np.mean(epoch_losses)
            history["losses"].extend(epoch_losses)
            print(f"Epoch {epoch + 1} complete. Average loss: {avg_loss:.4f}")

        return history
