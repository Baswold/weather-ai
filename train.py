#!/usr/bin/env python3
"""
Main training script for RL Weather.

Usage:
    python train.py --config low_memory --epochs 1
    python train.py --config production --epochs 3

The script:
1. Loads configuration
2. Prepares data (downloads from Open-Meteo if needed)
3. Creates the model
4. Trains through weather history chronologically
5. Saves checkpoints and logs
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from src.data import WeatherDataLoader, ChunkedWeatherDataLoader, OpenMeteoClient
from src.models import ActorCritic, TransformerConfig
from src.rl import RLTrainer, TrainerConfig, WeightedReward
from src.utils import (
    set_seed, get_device, print_system_info, check_memory_requirements,
    save_config, format_time,
)
from src.memory import MemoryMonitor, print_memory_estimation
from configs import (
    Config, get_default_config, get_low_memory_config, get_production_config,
    get_24gb_config, get_extended_locations_config, get_historical_config,
    get_climate_config, auto_config
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RL Weather prediction model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --config 24gb --epochs 3
  python train.py --config extended --epochs 5 --device cuda
  python train.py --auto --target-gb 24

Config presets:
  low_memory    3 locations, 1 year, ~500MB RAM (testing)
  default       10 locations, 5 years, ~4GB RAM
  production    20 locations, 2010-2024, ~8GB RAM
  24gb          20 locations, 2016-2024, ~24GB RAM (recommended for 32GB systems)
  extended      100+ locations, 2016-2024, ~24GB RAM
  historical    20 locations, 1950-2024, ~32GB+ RAM (75 years of data!)
  climate       20 locations, 1970-2024, ~32GB+ RAM (55 years, climate analysis)
        """
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        choices=["default", "low_memory", "production", "24gb", "extended", "historical", "climate"],
        default="low_memory",
        help="Configuration preset to use",
    )

    parser.add_argument(
        "--auto", "-a",
        action="store_true",
        help="Automatically tune config for available memory",
    )

    parser.add_argument(
        "--target-gb",
        type=float,
        default=None,
        help="Target memory usage in GB (for --auto)",
    )

    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )

    parser.add_argument(
        "--device", "-d",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for training",
    )

    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable data caching",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually download data or train (for testing setup)",
    )

    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Just test data pipeline and model, don't train",
    )

    parser.add_argument(
        "--chunked",
        action="store_true",
        help="Use chunked data loading to prevent storage buildup",
    )

    parser.add_argument(
        "--chunk-years",
        type=int,
        default=1,
        help="Number of years to load in each chunk (default: 1)",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Load configuration
    if args.auto:
        print("Auto-tuning configuration for available memory...")
        config = auto_config(target_gb=args.target_gb)
    else:
        config_map = {
            "default": get_default_config,
            "low_memory": get_low_memory_config,
            "production": get_production_config,
            "24gb": get_24gb_config,
            "extended": get_extended_locations_config,
            "historical": get_historical_config,
            "climate": get_climate_config,
        }
        config = config_map[args.config]()

    # Override with command line args
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if args.seed is not None:
        config.seed = args.seed
    if args.device != "auto":
        config.training.device = args.device
    if args.no_cache:
        config.data.cache_data = False
    if args.dry_run:
        config.dry_run = True

    # Set seed
    set_seed(config.seed)

    # Print system info
    print_system_info()

    # Print memory estimation
    print_memory_estimation(config)

    print("\n" + "=" * 50)
    print("RL Weather Training")
    print("=" * 50)
    print(f"Config: {args.config if not args.auto else 'auto'}")
    print(f"Locations: {len(config.data.locations)}")
    print(f"Training period: {config.data.start_date} to {config.data.end_date}")
    print(f"Epochs: {config.training.num_epochs}")
    print(f"Device: {config.training.device}")

    # Check memory
    if not check_memory_requirements(config) and args.config not in ["low_memory", "24gb", "extended"]:
        print("\nWARNING: Recommended to use --config low_memory")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != "y":
            print("Exiting.")
            return

    # Save config
    save_config(config, f"{config.training.checkpoint_dir}/config.json")

    if args.dry_run:
        print("\nDry run mode - skipping data download and training")
        return

    # Start memory monitoring
    monitor = MemoryMonitor(sample_interval=5.0, warning_threshold=0.90)
    monitor.start()

    def on_memory_warning(stats):
        print(f"\n⚠️  Memory warning: {stats.ram_percent:.1f}% RAM used")

    def on_memory_critical(stats):
        print(f"\n🚨 CRITICAL: {stats.ram_percent:.1f}% RAM used!")
        print("Consider reducing replay_buffer_size or batch_size")

    monitor.on_warning = on_memory_warning
    monitor.on_critical = on_memory_critical

    try:
        # Prepare data
        print("\n" + "=" * 50)
        print("Preparing Data")
        print("=" * 50)

        # Use chunked loading if requested
        use_chunked = args.chunked
        if use_chunked:
            print(f"Using CHUNKED data loading (chunk_years={args.chunk_years})")
            print("Data will be automatically cleaned up after each chunk to prevent storage buildup")
            chunked_loader = ChunkedWeatherDataLoader(
                locations=config.data.locations,
                start_date=config.data.start_date,
                end_date=config.data.end_date,
                window_size=config.data.window_size,
                chunk_years=args.chunk_years,
                data_dir=config.data.data_dir,
                cleanup_after_chunk=True,
            )
            data_loader = None  # Will use chunked_loader instead
        else:
            print("Using standard data loading (all data loaded at once)")
            print("WARNING: This may consume a lot of memory and disk space!")
            data_loader = WeatherDataLoader(
                locations=config.data.locations,
                start_date=config.data.start_date,
                end_date=config.data.end_date,
                window_size=config.data.window_size,
                data_dir=config.data.data_dir,
                cache=config.data.cache_data,
            )
            chunked_loader = None

        # Create model
        print("\n" + "=" * 50)
        print("Creating Model")
        print("=" * 50)

        model_config = TransformerConfig(
            d_model=config.model.d_model,
            nhead=config.model.nhead,
            num_layers=config.model.num_layers,
            dim_feedforward=config.model.dim_feedforward,
            num_weather_vars=config.model.num_weather_vars,
            window_size=config.model.window_size,
            dropout=config.model.dropout,
        )

        model = ActorCritic(model_config, use_twin_critic=True)

        from src.utils import count_parameters, get_model_size_mb
        print(f"Model parameters: {count_parameters(model):,}")
        print(f"Model size: {get_model_size_mb(model):.1f} MB")

        if args.test_only:
            print("\nTest mode - running quick forward pass...")
            if use_chunked:
                # Get first chunk and first batch
                first_chunk = next(iter(chunked_loader))
                batch = next(iter(first_chunk))[1]
            else:
                batch = next(iter(data_loader))[1]

            history = torch.from_numpy(batch.history).float()
            current = torch.from_numpy(batch.current_forecast).float()

            device = get_device(config.training.device)
            model = model.to(device)
            history = history.to(device)
            current = current.to(device)

            with torch.no_grad():
                output = model.get_action(history, current, deterministic=True)

            print(f"Input shapes: history={history.shape}, current={current.shape}")
            print(f"Output shape: {output.shape}")
            print("\nTest passed!")
            return

        # Create trainer
        print("\n" + "=" * 50)
        print("Creating Trainer")
        print("=" * 50)

        reward_fn = WeightedReward(
            variable_weights=config.reward.variable_weights,
            num_vars=config.model.num_weather_vars,
        )

        trainer_config = TrainerConfig(
            model_config=model_config,
            learning_rate=config.training.learning_rate,
            batch_size=config.training.batch_size,
            replay_buffer_size=config.training.replay_buffer_size,
            replay_start_size=config.training.replay_start_size,
            gamma=config.training.gamma,
            tau=config.training.tau,
            policy_noise=config.training.policy_noise,
            policy_freq=config.training.policy_freq,
            exploration_noise=config.training.exploration_noise,
            exploration_anneal=config.training.exploration_anneal,
            device=config.training.device,
            checkpoint_dir=config.training.checkpoint_dir,
            checkpoint_interval=config.training.checkpoint_interval,
            log_interval=config.training.log_interval,
        )

        trainer = RLTrainer(model, reward_fn, trainer_config)

        # Train
        print("\n" + "=" * 50)
        print("Starting Training")
        print("=" * 50)

        import time
        start_time = time.time()

        if use_chunked:
            # Use chunked training
            history = trainer.train_chunked(
                chunked_loader=chunked_loader,
                num_epochs=config.training.num_epochs,
            )
        else:
            # Use standard training
            history = trainer.train(
                data_loader=data_loader,
                num_epochs=config.training.num_epochs,
            )

        elapsed = time.time() - start_time
        print(f"\nTraining complete in {format_time(elapsed)}")

        # Save final model
        print("\nSaving final model...")
        trainer.save_checkpoint("final")

        # Print summary
        print("\n" + "=" * 50)
        print("Training Summary")
        print("=" * 50)
        print(f"Total steps: {trainer.current_step}")
        if history["rewards"]:
            print(f"Final reward: {history['rewards'][-1]:.4f}")
            print(f"Average reward: {sum(history['rewards']) / len(history['rewards']):.4f}")
            print(f"Best reward: {max(history['rewards']):.4f}")

    finally:
        # Stop monitoring and print summary
        monitor.stop()
        monitor.print_summary()


if __name__ == "__main__":
    main()
