"""
Transformer-based weather prediction model.

This model learns to predict tomorrow's weather from:
1. Historical window of [forecast, actual] pairs
2. Current day's forecast

Architecture:
- Input projection: Weather variables -> d_model
- Positional encoding: Temporal position information
- Transformer encoder: Process historical context
- Prediction head: Generate next day prediction
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


@dataclass
class TransformerConfig:
    """Configuration for the WeatherTransformer."""

    # Model dimensions
    d_model: int = 256  # Embedding dimension
    nhead: int = 8  # Number of attention heads
    num_layers: int = 6  # Number of transformer layers
    dim_feedforward: int = 1024  # Feedforward dimension

    # Input/output
    num_weather_vars: int = 6  # Number of weather variables
    window_size: int = 7  # Historical window size

    # Regularization
    dropout: float = 0.1
    activation: str = "relu"

    # Output
    output_dim: Optional[int] = None  # Default: same as input

    def __post_init__(self):
        if self.output_dim is None:
            self.output_dim = self.num_weather_vars


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal sequences.

    Based on: "Attention Is All You Need" (Vaswani et al., 2017)
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix (will be computed on first forward pass)
        self.register_buffer("pe", None)

    def _ensure_pe(self, seq_len: int):
        """Ensure positional encoding is large enough for the sequence."""
        if self.pe is None or self.pe.size(1) < seq_len:
            # Create positional encoding matrix with enough room
            max_len = max(self.max_len, seq_len)
            pe = torch.zeros(max_len, self.d_model, device=self.pe.device if self.pe is not None else 'cpu')
            position = torch.arange(0, max_len, dtype=torch.float, device=pe.device).unsqueeze(1)

            # Calculate the divisors
            div_term = torch.exp(
                torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)
            ).to(device=pe.device)

            # Apply sin to even positions, cos to odd positions
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            (batch_size, seq_len, d_model) with positional encoding added
        """
        self._ensure_pe(x.size(1))
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class WeatherTransformer(nn.Module):
    """
    Transformer-based weather prediction model.

    Input:
        - history: (batch, window_size, num_vars * 2) - historical [forecast, actual] pairs
        - current_forecast: (batch, num_vars) - current day's forecast

    Output:
        - prediction: (batch, num_vars) - predicted weather for tomorrow
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Input dimensions
        self.history_input_dim = config.num_weather_vars * 2  # forecast + actual
        self.forecast_input_dim = config.num_weather_vars
        self.d_model = config.d_model

        # Input projections
        self.history_projection = nn.Linear(
            self.history_input_dim, config.d_model
        )
        self.forecast_projection = nn.Linear(
            self.forecast_input_dim, config.d_model
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            config.d_model, config.dropout, max_len=config.window_size + 1
        )

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=True,  # Pre-LN architecture for stability
        )
        self.transformer = TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        # Layer normalization before output
        self.layer_norm = nn.LayerNorm(config.d_model)

        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.Dropout(config.dropout),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, config.output_dim),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        history: torch.Tensor,
        current_forecast: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            history: (batch_size, window_size, num_vars * 2)
            current_forecast: (batch_size, num_vars)

        Returns:
            predictions: (batch_size, num_vars)
        """
        batch_size = history.shape[0]

        # Project inputs
        history_embed = self.history_projection(history)  # (B, W, D)
        forecast_embed = self.forecast_projection(current_forecast)  # (B, D)

        # Add forecast as final token in sequence
        forecast_embed = forecast_embed.unsqueeze(1)  # (B, 1, D)
        sequence = torch.cat([history_embed, forecast_embed], dim=1)  # (B, W+1, D)

        # Add positional encoding
        sequence = self.pos_encoding(sequence)

        # Apply transformer
        transformed = self.transformer(sequence)  # (B, W+1, D)

        # Use the final token (current forecast position) for prediction
        final_token = transformed[:, -1, :]  # (B, D)
        final_token = self.layer_norm(final_token)

        # Generate prediction
        prediction = self.prediction_head(final_token)  # (B, output_dim)

        return prediction

    def get_attention_weights(self, history: torch.Tensor, current_forecast: torch.Tensor):
        """
        Extract attention weights for interpretability.

        Returns a list of attention matrices per layer.
        """
        # This would require modifying the transformer to return attention
        # For now, return None
        return None


class LightweightWeatherModel(nn.Module):
    """
    A lighter-weight model for testing on limited hardware.

    Uses a simple LSTM + MLP instead of full transformer.
    """

    def __init__(
        self,
        num_weather_vars: int = 6,
        window_size: int = 7,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_weather_vars = num_weather_vars
        self.window_size = window_size
        self.hidden_dim = hidden_dim

        # LSTM for processing history
        self.lstm = nn.LSTM(
            input_size=num_weather_vars * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0,
        )

        # Combine LSTM output with current forecast
        self.fc_combine = nn.Linear(hidden_dim + num_weather_vars, hidden_dim)

        # Output head
        self.fc_out = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_weather_vars),
        )

    def forward(self, history: torch.Tensor, current_forecast: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history: (batch, window_size, num_vars * 2)
            current_forecast: (batch, num_vars)
        """
        # Process history through LSTM
        lstm_out, (h_n, c_n) = self.lstm(history)
        # Use final hidden state
        final_hidden = h_n[-1]  # (batch, hidden_dim)

        # Combine with current forecast
        combined = torch.cat([final_hidden, current_forecast], dim=-1)
        x = self.fc_combine(combined)

        # Generate prediction
        prediction = self.fc_out(x)

        return prediction


class MultiStepWeatherModel(WeatherTransformer):
    """
    Extension of WeatherTransformer that predicts multiple days ahead.

    Instead of single output, outputs predictions for next N days.
    Uses autoregressive decoding for longer horizons.
    """

    def __init__(self, config: TransformerConfig, horizon: int = 3):
        super().__init__(config)
        self.horizon = horizon

        # Modify output head to predict multiple days
        self.prediction_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.Dropout(config.dropout),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, config.output_dim * horizon),
        )

    def forward(
        self,
        history: torch.Tensor,
        current_forecast: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict next `horizon` days.

        Returns:
            (batch_size, horizon, num_vars)
        """
        batch_size = history.shape[0]

        # Get embeddings
        history_embed = self.history_projection(history)
        forecast_embed = self.forecast_projection(current_forecast).unsqueeze(1)

        # Process through transformer
        sequence = torch.cat([history_embed, forecast_embed], dim=1)
        sequence = self.pos_encoding(sequence)
        transformed = self.transformer(sequence)
        final_token = self.layer_norm(transformed[:, -1, :])

        # Predict all horizons at once
        predictions = self.prediction_head(final_token)
        predictions = predictions.view(batch_size, self.horizon, -1)

        return predictions


def test_model():
    """Test the model with dummy data."""
    config = TransformerConfig(
        d_model=128,
        nhead=4,
        num_layers=3,
        num_weather_vars=6,
        window_size=7,
    )

    model = WeatherTransformer(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 4
    history = torch.randn(batch_size, 7, 12)  # 6 vars * 2
    current = torch.randn(batch_size, 6)

    output = model(history, current)
    print(f"Output shape: {output.shape}")  # Should be (4, 6)

    return model


if __name__ == "__main__":
    test_model()
