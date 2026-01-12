"""
Memory monitoring and optimization utilities.

Provides real-time memory tracking and optimization recommendations
for large-scale training on systems with 24GB+ RAM.
"""

import gc
import os
import psutil
import threading
import time
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class MemoryStats:
    """Container for memory statistics."""

    # System memory
    total_ram_gb: float = 0.0
    available_ram_gb: float = 0.0
    used_ram_gb: float = 0.0
    ram_percent: float = 0.0

    # Process memory
    process_ram_mb: float = 0.0
    process_ram_gb: float = 0.0

    # GPU memory (if available)
    gpu_total_gb: float = 0.0
    gpu_used_gb: float = 0.0
    gpu_free_gb: float = 0.0

    # Component breakdown (estimated)
    model_mb: float = 0.0
    replay_buffer_mb: float = 0.0
    data_cache_mb: float = 0.0
    overhead_mb: float = 0.0

    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def safe_to_continue(self) -> bool:
        """Check if there's enough memory to continue training."""
        return self.available_ram_gb > 2.0  # Keep 2GB buffer

    @property
    def memory_pressure(self) -> str:
        """Get memory pressure level."""
        if self.ram_percent < 70:
            return "low"
        elif self.ram_percent < 85:
            return "medium"
        elif self.ram_percent < 95:
            return "high"
        else:
            return "critical"


class MemoryMonitor:
    """
    Real-time memory monitoring for training.

    Tracks RAM and GPU usage during training, provides warnings,
    and can trigger actions when memory thresholds are exceeded.
    """

    def __init__(
        self,
        sample_interval: float = 1.0,
        warning_threshold: float = 0.85,
        critical_threshold: float = 0.95,
        enable_gc: bool = True,
    ):
        """
        Args:
            sample_interval: Seconds between memory samples
            warning_threshold: RAM usage fraction for warnings
            critical_threshold: RAM usage fraction for critical actions
            enable_gc: Whether to run garbage collection on critical
        """
        self.sample_interval = sample_interval
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.enable_gc = enable_gc

        self.process = psutil.Process(os.getpid())
        self.monitoring = False
        self.thread: Optional[threading.Thread] = None

        # Stats history
        self.history: list[MemoryStats] = []
        self.max_history = 1000

        # Callbacks
        self.on_warning: Optional[Callable[[MemoryStats], None]] = None
        self.on_critical: Optional[Callable[[MemoryStats], None]] = None

    def get_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        stats = MemoryStats()

        # System RAM
        ram = psutil.virtual_memory()
        stats.total_ram_gb = ram.total / (1024 ** 3)
        stats.available_ram_gb = ram.available / (1024 ** 3)
        stats.used_ram_gb = ram.used / (1024 ** 3)
        stats.ram_percent = ram.percent

        # Process RAM
        process_mem = self.process.memory_info()
        stats.process_ram_mb = process_mem.rss / (1024 ** 2)
        stats.process_ram_gb = stats.process_ram_mb / 1024

        # GPU memory (if available)
        try:
            import torch
            if torch.cuda.is_available():
                stats.gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                stats.gpu_used_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
                stats.gpu_free_gb = stats.gpu_total_gb - stats.gpu_used_gb
        except ImportError:
            pass

        return stats

    def start(self):
        """Start background monitoring."""
        if self.monitoring:
            return

        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop background monitoring."""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=5)
            self.thread = None

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            stats = self.get_stats()

            # Store in history
            self.history.append(stats)
            if len(self.history) > self.max_history:
                self.history.pop(0)

            # Check thresholds
            usage_fraction = stats.ram_percent / 100

            if usage_fraction >= self.critical_threshold:
                if self.enable_gc:
                    gc.collect()
                if self.on_critical:
                    self.on_critical(stats)
            elif usage_fraction >= self.warning_threshold:
                if self.on_warning:
                    self.on_warning(stats)

            time.sleep(self.sample_interval)

    def get_peak_memory(self) -> float:
        """Get peak memory usage during monitoring (GB)."""
        if not self.history:
            return 0.0
        return max(s.process_ram_gb for s in self.history)

    def get_average_memory(self) -> float:
        """Get average memory usage during monitoring (GB)."""
        if not self.history:
            return 0.0
        return sum(s.process_ram_gb for s in self.history) / len(self.history)

    def print_summary(self):
        """Print memory usage summary."""
        if not self.history:
            print("No memory data collected.")
            return

        current = self.history[-1]
        peak = self.get_peak_memory()
        avg = self.get_average_memory()

        print("\n" + "=" * 50)
        print("Memory Summary")
        print("=" * 50)
        print(f"Current: {current.process_ram_gb:.2f} GB")
        print(f"Peak:    {peak:.2f} GB")
        print(f"Average: {avg:.2f} GB")
        print(f"System:  {current.ram_percent:.1f}% used ({current.used_ram_gb:.1f} / {current.total_ram_gb:.1f} GB)")
        if current.gpu_total_gb > 0:
            print(f"GPU:     {current.gpu_used_gb:.2f} / {current.gpu_total_gb:.2f} GB")
        print("=" * 50)


def estimate_memory_usage(config: Any) -> Dict[str, float]:
    """
    Estimate memory usage for a given configuration.

    Returns breakdown in GB.
    """
    from ..models import TransformerConfig

    # Model memory
    model_config = TransformerConfig(
        d_model=config.model.d_model,
        nhead=config.model.nhead,
        num_layers=config.model.num_layers,
        dim_feedforward=config.model.dim_feedforward,
        num_weather_vars=config.model.num_weather_vars,
        window_size=config.model.window_size,
    )

    # Rough parameter count estimate
    # Transformer params ~ d_model^2 * num_layers * 12
    params_estimate = (
        model_config.d_model ** 2 * model_config.num_layers * 12 +
        model_config.dim_feedforward * model_config.d_model * model_config.num_layers * 2
    )
    model_memory_gb = (params_estimate * 4) / (1024 ** 3)  # 4 bytes per param (float32)

    # Replay buffer memory
    # Each transition: state (~1KB) + action (~100B) + overhead
    transition_size_bytes = 1500  # Conservative estimate
    replay_buffer_gb = (
        config.training.replay_buffer_size * transition_size_bytes
    ) / (1024 ** 3)

    # Data cache memory
    # 9 years * 365 days * locations * variables * 4 bytes
    num_locations = len(config.data.locations)
    num_days = (datetime.strptime(config.data.end_date, "%Y-%m-%d") -
                datetime.strptime(config.data.start_date, "%Y-%m-%d")).days
    num_vars = config.model.num_weather_vars

    data_size_bytes = num_days * num_locations * num_vars * 8 * 3  # forecast, actual, target
    data_cache_gb = data_size_bytes / (1024 ** 3)

    # Training overhead (activations, gradients, optimizer states)
    # Adam maintains 2 states per parameter
    overhead_gb = model_memory_gb * 3  # Model + gradients + optimizer states

    # Batch processing overhead
    batch_overhead_gb = (config.training.batch_size * model_config.window_size *
                         config.model.num_weather_vars * 2 * 4) / (1024 ** 3)

    total_gb = model_memory_gb + replay_buffer_gb + data_cache_gb + overhead_gb + batch_overhead_gb

    return {
        "model_gb": model_memory_gb,
        "replay_buffer_gb": replay_buffer_gb,
        "data_cache_gb": data_cache_gb,
        "overhead_gb": overhead_gb + batch_overhead_gb,
        "total_gb": total_gb,
        "parameters_estimate": params_estimate,
    }


def print_memory_estimation(config: Any):
    """Print detailed memory estimation for a config."""
    print("\n" + "=" * 60)
    print("Memory Estimation")
    print("=" * 60)

    usage = estimate_memory_usage(config)

    print(f"\nModel:")
    print(f"  Parameters:  ~{usage['parameters_estimate']:,}")
    print(f"  Memory:      {usage['model_gb']:.2f} GB")

    print(f"\nReplay Buffer:")
    print(f"  Size:        {config.training.replay_buffer_size:,} transitions")
    print(f"  Memory:      {usage['replay_buffer_gb']:.2f} GB")

    print(f"\nData Cache:")
    print(f"  Locations:   {len(config.data.locations)}")
    print(f"  Memory:      {usage['data_cache_gb']:.2f} GB")

    print(f"\nTraining Overhead:")
    print(f"  Memory:      {usage['overhead_gb']:.2f} GB")

    print(f"\n{'=' * 40}")
    print(f"Total Estimated: {usage['total_gb']:.2f} GB")
    print(f"{'=' * 60}")

    # Check against system memory
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    print(f"\nSystem RAM: {ram_gb:.1f} GB")

    if usage['total_gb'] > ram_gb * 0.8:
        print("⚠️  WARNING: Estimated usage exceeds 80% of system RAM!")
        print("   Consider using a smaller configuration.")
    elif usage['total_gb'] > ram_gb * 0.6:
        print("⚠️  Estimated usage is 60-80% of system RAM.")
        print("   Close other applications before training.")
    else:
        print("✓ Estimated memory usage is within safe limits.")


def optimize_memory_settings(config: Any, target_gb: float = 24.0) -> Any:
    """
    Automatically tune configuration to target memory usage.

    Adjusts replay buffer size and batch size to fit within target.
    """
    current_usage = estimate_memory_usage(config)['total_gb']

    if current_usage <= target_gb:
        print(f"✓ Current config uses ~{current_usage:.1f} GB (within {target_gb} GB target)")
        return config

    print(f"\nTuning config for {target_gb} GB target (current: ~{current_usage:.1f} GB)...")

    # Scale down replay buffer proportionally
    scale_factor = target_gb / current_usage

    # Apply scaling to replay buffer (main memory consumer)
    config.training.replay_buffer_size = int(
        config.training.replay_buffer_size * scale_factor * 0.9
    )

    # Recalculate
    new_usage = estimate_memory_usage(config)['total_gb']
    print(f"  Adjusted replay_buffer_size to {config.training.replay_buffer_size:,}")
    print(f"  New estimated usage: ~{new_usage:.1f} GB")

    return config


class MemoryEfficientDataLoader:
    """
    Wrapper for data loading with memory management.

    Automatically unloads data between batches and can
    stream from disk instead of keeping everything in RAM.
    """

    def __init__(self, base_loader, unload_between_batches: bool = True):
        """
        Args:
            base_loader: The underlying WeatherDataLoader
            unload_between_batches: Whether to free memory between batches
        """
        self.base_loader = base_loader
        self.unload_between = unload_between_batches

    def __iter__(self):
        """Iterate with memory cleanup between batches."""
        for date, batch in self.base_loader:
            yield date, batch

            if self.unload_between:
                # Force garbage collection
                gc.collect()

    def __len__(self):
        return len(self.base_loader)


def clear_gpu_memory():
    """Clear GPU memory cache."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def get_memory_limit_gb() -> float:
    """Get the target memory limit based on system RAM."""
    total_gb = psutil.virtual_memory().total / (1024 ** 3)

    # Use 75% of total RAM to leave room for OS
    return total_gb * 0.75


def auto_config(target_gb: Optional[float] = None) -> Any:
    """
    Automatically create a config optimized for available memory.

    Args:
        target_gb: Target memory usage (default: 75% of system RAM)
    """
    from configs import Config, get_24gb_config, get_low_memory_config

    if target_gb is None:
        target_gb = get_memory_limit_gb()

    total_gb = psutil.virtual_memory().total / (1024 ** 3)

    if target_gb >= 20:
        config = get_24gb_config()
        print(f"Using 24GB config (target: {target_gb:.0f} GB)")
    elif target_gb >= 8:
        config = Config()
        config.training.replay_buffer_size = 2_000_000
        config.training.batch_size = 50
        config.model.d_model = 256
        config.model.num_layers = 6
        print(f"Using medium config (target: {target_gb:.0f} GB)")
    else:
        config = get_low_memory_config()
        print(f"Using low-memory config (target: {target_gb:.0f} GB)")

    # Tune to exact target
    config = optimize_memory_settings(config, target_gb)

    return config


if __name__ == "__main__":
    # Test memory monitoring
    print("Testing MemoryMonitor...")

    monitor = MemoryMonitor()

    # Print current stats
    stats = monitor.get_stats()
    print(f"\nCurrent memory usage:")
    print(f"  Process: {stats.process_ram_gb:.2f} GB")
    print(f"  System:  {stats.ram_percent:.1f}%")

    # Test memory estimation
    from configs import get_24gb_config
    config = get_24gb_config()
    print_memory_estimation(config)
