"""
Actor-Critic architecture for RL-based weather prediction.

Uses DDPG-style continuous action space RL where:
- Actor: Predicts weather values (continuous action)
- Critic: Estimates value of state-action pairs

This allows training with exploration and temporal difference learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import WeatherTransformer, TransformerConfig


class Actor(WeatherTransformer):
    """
    Actor network that outputs weather predictions.

    In RL terms, the "action" is the predicted weather values.
    The actor learns to map state -> optimal prediction.
    """

    def __init__(self, config: TransformerConfig, action_scale: float = 1.0):
        super().__init__(config)
        self.action_scale = action_scale

        # Override prediction head to have bounded output
        # Using tanh for bounded predictions, then scale
        self.prediction_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, config.output_dim),
            nn.Tanh(),  # Bound to [-1, 1]
        )

    def forward(self, history: torch.Tensor, current_forecast: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with scaled output.

        Returns predictions scaled to reasonable weather ranges.
        """
        # Get base prediction in [-1, 1]
        prediction = super().forward(history, current_forecast)

        # Scale to actual weather value ranges
        # Note: You may want variable-specific scaling
        return prediction * self.action_scale

    def get_action(
        self,
        history: torch.Tensor,
        current_forecast: torch.Tensor,
        add_noise: bool = False,
        noise_scale: float = 0.1,
    ) -> torch.Tensor:
        """
        Get action with optional exploration noise.

        Args:
            history: Historical weather context
            current_forecast: Current day's forecast
            add_noise: Whether to add exploration noise
            noise_scale: Scale of Gaussian noise

        Returns:
            Weather prediction (with optional noise)
        """
        action = self.forward(history, current_forecast)

        if add_noise:
            noise = torch.randn_like(action) * noise_scale
            action = action + noise

        return action


class Critic(nn.Module):
    """
    Critic network that estimates Q(s, a).

    Takes both state (history + forecast) and action (prediction)
    and outputs the estimated value of that action.
    """

    def __init__(
        self,
        config: TransformerConfig,
        action_dim: int = None,
    ):
        super().__init__()

        action_dim = action_dim or config.num_weather_vars
        self.d_model = config.d_model

        # State encoder (simplified from full transformer)
        self.state_encoder = nn.Sequential(
            nn.Linear(config.num_weather_vars * 2 * config.window_size + config.num_weather_vars, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
        )

        # Q network: processes state and action together
        self.q_network = nn.Sequential(
            nn.Linear(config.d_model + action_dim, config.dim_feedforward),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1),
        )

    def forward(
        self,
        history: torch.Tensor,
        current_forecast: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Q-value estimation.

        Args:
            history: (batch, window_size, vars*2)
            current_forecast: (batch, vars)
            action: (batch, vars) - the weather prediction

        Returns:
            Q-values: (batch, 1)
        """
        batch_size = history.shape[0]

        # Flatten state
        history_flat = history.reshape(batch_size, -1)
        state = torch.cat([history_flat, current_forecast], dim=-1)

        # Encode state
        state_embed = self.state_encoder(state)

        # Concatenate state and action for Q-value
        sa_pair = torch.cat([state_embed, action], dim=-1)
        q_value = self.q_network(sa_pair)

        return q_value


class TwinCritic(nn.Module):
    """
    Twin Q-networks (TD3 style) to reduce overestimation bias.

    Uses two critic networks and takes the minimum for target computation.
    """

    def __init__(self, config: TransformerConfig, action_dim: int = None):
        super().__init__()

        action_dim = action_dim or config.num_weather_vars

        # Two separate critic networks
        self.critic1 = Critic(config, action_dim)
        self.critic2 = Critic(config, action_dim)

    def forward(
        self,
        history: torch.Tensor,
        current_forecast: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Q-values from both critics."""
        q1 = self.critic1(history, current_forecast, action)
        q2 = self.critic2(history, current_forecast, action)
        return q1, q2

    def q1(self, history: torch.Tensor, current_forecast: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return only Q1 (used during training)."""
        return self.critic1(history, current_forecast, action)


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic model for weather prediction RL.

    This follows the DDPG/TD3 architecture:
    - Actor: Makes weather predictions
    - Critic: Evaluates prediction quality

    Training objective:
    - Actor: Maximize Q-value (make better predictions)
    - Critic: Minimize TD error (accurately value predictions)
    """

    def __init__(
        self,
        config: TransformerConfig,
        use_twin_critic: bool = True,
        action_scale: float = 10.0,  # Scale for temperature predictions
    ):
        super().__init__()

        self.config = config
        self.use_twin_critic = use_twin_critic

        # Actor network
        self.actor = Actor(config, action_scale=action_scale)

        # Critic network(s)
        if use_twin_critic:
            self.critic = TwinCritic(config)
        else:
            self.critic = Critic(config)

    def forward(self, history: torch.Tensor, current_forecast: torch.Tensor) -> torch.Tensor:
        """
        Forward through actor (prediction).

        During training, use get_action for exploration.
        """
        return self.actor(history, current_forecast)

    def get_action(
        self,
        history: torch.Tensor,
        current_forecast: torch.Tensor,
        deterministic: bool = True,
        noise_scale: float = 0.1,
    ) -> torch.Tensor:
        """Get action with optional exploration noise."""
        return self.actor.get_action(history, current_forecast, add_noise=not deterministic, noise_scale=noise_scale)

    def get_q_value(
        self,
        history: torch.Tensor,
        current_forecast: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Get Q-value from critic(s)."""
        if self.use_twin_critic:
            q1, q2 = self.critic(history, current_forecast, action)
            return q1  # Return Q1 for training
        else:
            return self.critic(history, current_forecast, action)

    def get_target_q(
        self,
        history: torch.Tensor,
        current_forecast: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get target Q-value (minimum of twin critics for TD3).

        Used for computing target values during training.
        """
        if self.use_twin_critic:
            q1, q2 = self.critic(history, current_forecast, action)
            return torch.min(q1, q2)
        else:
            return self.critic(history, current_forecast, action)


class EnsembleActorCritic(nn.Module):
    """
    Ensemble of actor-critic models for uncertainty estimation.

    Maintains multiple models and can output:
    - Mean prediction
    - Uncertainty (variance across ensemble)
    - Full distribution
    """

    def __init__(
        self,
        config: TransformerConfig,
        ensemble_size: int = 5,
        action_scale: float = 10.0,
    ):
        super().__init__()

        self.ensemble_size = ensemble_size

        # Create ensemble
        self.models = nn.ModuleList([
            ActorCritic(config, use_twin_critic=False, action_scale=action_scale)
            for _ in range(ensemble_size)
        ])

    def forward(self, history: torch.Tensor, current_forecast: torch.Tensor) -> dict:
        """
        Forward through ensemble.

        Returns dict with:
        - mean: Mean prediction
        - std: Standard deviation across ensemble
        - predictions: Individual predictions
        """
        predictions = []

        for model in self.models:
            pred = model(history, current_forecast)
            predictions.append(pred)

        stacked = torch.stack(predictions, dim=0)  # (ensemble, batch, vars)

        return {
            "mean": stacked.mean(dim=0),
            "std": stacked.std(dim=0),
            "predictions": stacked,
        }

    def get_action(
        self,
        history: torch.Tensor,
        current_forecast: torch.Tensor,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Get action from ensemble (mean prediction)."""
        outputs = self.forward(history, current_forecast)
        return outputs["mean"] if deterministic else outputs["mean"] + torch.randn_like(outputs["mean"]) * outputs["std"]


def test_actor_critic():
    """Test the actor-critic model."""
    config = TransformerConfig(
        d_model=64,
        nhead=2,
        num_layers=2,
        num_weather_vars=6,
        window_size=7,
        dim_feedforward=256,
    )

    model = ActorCritic(config, use_twin_critic=True)
    print(f"Actor parameters: {sum(p.numel() for p in model.actor.parameters()):,}")
    print(f"Critic parameters: {sum(p.numel() for p in model.critic.parameters()):,}")

    # Test forward pass
    batch_size = 4
    history = torch.randn(batch_size, 7, 12)
    current = torch.randn(batch_size, 6)

    # Actor forward
    action = model.get_action(history, current, deterministic=True)
    print(f"Action shape: {action.shape}")

    # Critic forward
    q_value = model.get_q_value(history, current, action)
    print(f"Q-value shape: {q_value.shape}")

    return model


if __name__ == "__main__":
    test_actor_critic()
