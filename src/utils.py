"""
Utility functions for RL Weather.
"""

import random
import numpy as np
import torch
from typing import Optional, Dict, Any
from pathlib import Path


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Get model size in megabytes."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def print_system_info():
    """Print system and PyTorch info."""
    import platform
    import psutil

    print("=" * 50)
    print("System Information")
    print("=" * 50)
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")

    # CUDA info
    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA: Not available")

    # RAM info
    ram = psutil.virtual_memory()
    print(f"RAM Total: {ram.total / (1024**3):.1f} GB")
    print(f"RAM Available: {ram.available / (1024**3):.1f} GB")
    print("=" * 50)


def check_memory_requirements(config: Any) -> bool:
    """
    Check if system has enough memory for training.

    Returns True if safe to proceed, False otherwise.
    """
    import psutil

    ram_available_gb = psutil.virtual_memory().available / (1024 ** 3)

    # Rough estimates based on model size
    model_params = (
        config.model.d_model ** 2 * config.model.num_layers * 4 +
        config.model.d_model * config.model.dim_feedforward * config.model.num_layers * 2
    ) / 1e9  # Very rough estimate in GB

    # Buffer data
    buffer_gb = config.training.replay_buffer_size * 32 * 4 / (1024 ** 3)

    total_needed_gb = model_params + buffer_gb + 1  # +1GB for overhead

    print(f"Estimated memory needed: {total_needed_gb:.1f} GB")
    print(f"Available RAM: {ram_available_gb:.1f} GB")

    if ram_available_gb < total_needed_gb * 1.5:
        print("WARNING: System may not have enough memory!")
        print("Consider using get_low_memory_config()")
        return False

    return True


class EarlyStopping:
    """Early stopping utility."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """Check if should stop early."""
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop


class ProgressBar:
    """Simple progress bar for training."""

    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.description = description
        self.current = 0
        self.last_print = 0

    def update(self, n: int = 1):
        """Update progress."""
        self.current += n
        if self.current - self.last_print >= max(1, self.total // 100) or self.current >= self.total:
            pct = 100 * self.current / self.total
            print(f"\r{self.description}: {pct:.0f}%", end="", flush=True)
            self.last_print = self.current

    def close(self):
        """Finish progress bar."""
        print()  # New line


def save_config(config: Any, path: str):
    """Save configuration to file."""
    import json
    from dataclasses import asdict

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclass to dict
    if hasattr(config, '__dataclass_fields__'):
        config_dict = asdict(config)
    else:
        config_dict = config.__dict__

    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)

    print(f"Config saved to {path}")


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from file."""
    import json

    with open(path, 'r') as f:
        return json.load(f)
