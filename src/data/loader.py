"""
Weather data loader for sequential temporal RL training.

The loader creates training examples in the format:
- State: Historical window of [forecast, actual] pairs + current forecast
- Target: Next day's actual weather

This ensures temporal causality - the model never sees future data
when predicting the past.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Iterator
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path

from .openmeteo import OpenMeteoClient


@dataclass
class WeatherBatch:
    """A batch of weather data for training."""
    # History: (batch_size, window_size, num_vars * 2)
    # The *2 is for [forecast, actual] pairs
    history: np.ndarray

    # Current forecast: (batch_size, num_vars)
    current_forecast: np.ndarray

    # Target (next day actual): (batch_size, num_vars)
    target: np.ndarray

    # Metadata
    locations: List[str]
    dates: List[datetime]

    def to_tuple(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert to tuple for model input."""
        return self.history, self.current_forecast, self.target


class WeatherDataLoader:
    """
    Data loader for sequential temporal RL training.

    The key innovation: data is experienced in chronological order,
    allowing the model to "live through" weather history.

    Example:
        loader = WeatherDataLoader(
            locations=[{"name": "NYC", "lat": 40.71, "lon": -74.01}],
            start_date="2020-01-01",
            end_date="2024-12-31",
            window_size=7
        )

        # Iterate through all days in order
        for date, batch in loader:
            # Train on this day's data
            prediction = model(batch.history, batch.current_forecast)
            reward = compute_reward(prediction, batch.target)
    """

    # Default weather variables to use
    DEFAULT_VARIABLES = [
        "temperature_2m_min",
        "temperature_2m_max",
        "temperature_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_max",
        "wind_speed_10m_mean",
    ]

    def __init__(
        self,
        locations: List[Dict[str, float]],
        start_date: str,
        end_date: str,
        window_size: int = 7,
        variables: Optional[List[str]] = None,
        data_dir: Optional[str] = None,
        cache: bool = True,
    ):
        """
        Initialize the data loader.

        Args:
            locations: List of dicts with 'name', 'latitude', 'longitude'
            start_date: Training start date (YYYY-MM-DD)
            end_date: Training end date (YYYY-MM-DD)
            window_size: Number of historical days to include as context
            variables: List of weather variables to use
            data_dir: Directory to cache downloaded data
            cache: Whether to cache downloaded data
        """
        self.locations = locations
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.window_size = window_size
        self.variables = variables or self.DEFAULT_VARIABLES.copy()
        self.data_dir = Path(data_dir) if data_dir else Path("data/weather")
        self.cache = cache

        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize API client
        self.client = OpenMeteoClient()

        # Data cache: {location_name: DataFrame}
        self._data_cache: Dict[str, pd.DataFrame] = {}

        # Load all data
        self._load_data()

        # Create date range for sequential training
        self.date_range = pd.date_range(
            start=self.start_date + timedelta(days=window_size),
            end=self.end_date - timedelta(days=1),
            freq="D"
        )

        print(f"Loaded data for {len(self.locations)} locations")
        print(f"Training period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Total training days: {len(self.date_range)}")

    def _load_data(self):
        """Load weather data for all locations."""
        for loc in self.locations:
            name = loc["name"]
            cache_file = self.data_dir / f"{name}_{self.start_date.date()}_{self.end_date.date()}.parquet"

            # Try to load from cache
            if self.cache and cache_file.exists():
                print(f"Loading cached data for {name}")
                self._data_cache[name] = pd.read_parquet(cache_file)
                continue

            # Download data
            print(f"Downloading data for {name}...")
            df = self.client.get_forecast_verification(
                latitude=loc["latitude"],
                longitude=loc["longitude"],
                start_date=self.start_date.strftime("%Y-%m-%d"),
                end_date=self.end_date.strftime("%Y-%m-%d"),
            )

            # Select requested variables (if available)
            available_vars = [v for v in self.variables if v in df.columns]
            if available_vars:
                df = df[["date"] + available_vars]

            # Cache to disk
            if self.cache:
                df.to_parquet(cache_file, index=False)

            self._data_cache[name] = df

    def get_day_window(
        self,
        location: str,
        date: datetime,
    ) -> Optional[WeatherBatch]:
        """
        Get a single day's data with historical context window.

        Args:
            location: Location name
            date: Current date (will predict for date + 1)

        Returns:
            WeatherBatch with history, current forecast, and target
        """
        df = self._data_cache.get(location)
        if df is None:
            return None

        df_date = pd.to_datetime(date)
        df["date"] = pd.to_datetime(df["date"])

        # Find the row for current date
        current_idx = df[df["date"] == df_date].index
        if len(current_idx) == 0:
            return None
        current_idx = current_idx[0]

        # Check we have enough history and a target
        if current_idx < self.window_size or current_idx >= len(df) - 1:
            return None

        # Extract historical window
        history_start = current_idx - self.window_size
        history_data = df.iloc[history_start:current_idx]

        # Get current and target
        current_data = df.iloc[current_idx]
        target_data = df.iloc[current_idx + 1]

        # Build arrays
        history_array = history_data[self.variables].values
        current_array = current_data[self.variables].values
        target_array = target_data[self.variables].values

        # For history, we need [forecast, actual] pairs
        # Since we only have actuals, duplicate the data for now
        # In a real implementation, you'd have separate forecast columns
        history_pairs = np.stack([history_array, history_array], axis=-1)
        history_pairs = history_pairs.reshape(self.window_size, -1)

        return WeatherBatch(
            history=history_pairs[np.newaxis, ...],  # Add batch dim
            current_forecast=current_array[np.newaxis, ...],
            target=target_array[np.newaxis, ...],
            locations=[location],
            dates=[date],
        )

    def get_batch_for_date(
        self,
        date: datetime,
        locations: Optional[List[str]] = None,
    ) -> Optional[WeatherBatch]:
        """
        Get a batch of data from all locations for a specific date.

        This is the key method for sequential temporal RL:
        On each calendar day, we get predictions from all locations
        and train on them together.

        Args:
            date: The calendar date to get data for
            locations: List of location names (default: all)

        Returns:
            WeatherBatch with data from all locations
        """
        if locations is None:
            locations = [loc["name"] for loc in self.locations]

        batches = []
        batch_locations = []
        batch_dates = []

        for loc_name in locations:
            batch = self.get_day_window(loc_name, date)
            if batch is not None:
                batches.append(batch)
                batch_locations.append(loc_name)
                batch_dates.append(date)

        if not batches:
            return None

        # Concatenate all location batches
        combined = WeatherBatch(
            history=np.concatenate([b.history for b in batches], axis=0),
            current_forecast=np.concatenate([b.current_forecast for b in batches], axis=0),
            target=np.concatenate([b.target for b in batches], axis=0),
            locations=batch_locations,
            dates=batch_dates,
        )

        return combined

    def __iter__(self) -> Iterator[Tuple[datetime, WeatherBatch]]:
        """
        Iterate through all training days in chronological order.

        This is the main interface for sequential temporal RL training.

        Yields:
            (date, batch) tuples for each day in the training period
        """
        for date in self.date_range:
            batch = self.get_batch_for_date(date)
            if batch is not None:
                yield date, batch

    def __len__(self) -> int:
        """Return the number of training days."""
        return len(self.date_range)

    def split_locations(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split locations into train/val/test sets.

        Args:
            train_ratio: Fraction of locations for training
            val_ratio: Fraction of locations for validation

        Returns:
            (train_locations, val_locations, test_locations)
        """
        n = len(self.locations)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        # Shuffle by climate zone for better distribution
        import random
        locations_shuffled = self.locations.copy()
        random.shuffle(locations_shuffled)

        train = locations_shuffled[:n_train]
        val = locations_shuffled[n_train:n_train + n_val]
        test = locations_shuffled[n_train + n_val:]

        return train, val, test

    def get_statistics(self) -> Dict[str, np.ndarray]:
        """
        Compute normalization statistics across all data.

        Returns:
            Dict with 'mean' and 'std' arrays for each variable
        """
        all_data = []

        for df in self._data_cache.values():
            if all(c in df.columns for c in self.variables):
                all_data.append(df[self.variables].values)

        if not all_data:
            return {"mean": np.zeros(len(self.variables)), "std": np.ones(len(self.variables))}

        combined = np.vstack(all_data)
        return {
            "mean": combined.mean(axis=0),
            "std": combined.std(axis=0),
        }

    def cleanup_data(self, delete_cache_files: bool = True):
        """
        Clean up data to free memory and optionally delete cache files.

        This should be called after training on a chunk of data to prevent
        storage buildup.

        Args:
            delete_cache_files: If True, delete the parquet cache files from disk
        """
        # Clear in-memory cache
        num_locations = len(self._data_cache)
        self._data_cache.clear()
        print(f"Cleared data cache for {num_locations} locations from memory")

        # Delete cache files if requested
        if delete_cache_files and self.data_dir.exists():
            cache_files = list(self.data_dir.glob("*.parquet"))
            deleted_count = 0
            total_size = 0

            for cache_file in cache_files:
                try:
                    # Check if this file belongs to our date range
                    if f"{self.start_date.date()}_{self.end_date.date()}" in cache_file.name:
                        file_size = cache_file.stat().st_size
                        cache_file.unlink()
                        deleted_count += 1
                        total_size += file_size
                except Exception as e:
                    print(f"Warning: Failed to delete {cache_file}: {e}")

            if deleted_count > 0:
                size_mb = total_size / (1024 * 1024)
                print(f"Deleted {deleted_count} cache files, freed {size_mb:.2f} MB of disk space")


class ChunkedWeatherDataLoader:
    """
    Weather data loader that loads data in chunks to prevent memory/storage buildup.

    This loader processes data in time chunks (e.g., year by year), automatically
    cleaning up after each chunk to free memory and disk space.

    Example:
        loader = ChunkedWeatherDataLoader(
            locations=[{"name": "NYC", "lat": 40.71, "lon": -74.01}],
            start_date="2015-01-01",
            end_date="2024-12-31",
            chunk_years=1,  # Process 1 year at a time
            cleanup_after_chunk=True,  # Delete data after processing
        )

        for chunk_loader in loader:
            # Train on this chunk
            for date, batch in chunk_loader:
                train(batch)
            # Data is automatically cleaned up after this iteration
    """

    def __init__(
        self,
        locations: List[Dict[str, float]],
        start_date: str,
        end_date: str,
        window_size: int = 7,
        chunk_years: int = 1,
        variables: Optional[List[str]] = None,
        data_dir: Optional[str] = None,
        cleanup_after_chunk: bool = True,
    ):
        """
        Initialize the chunked data loader.

        Args:
            locations: List of dicts with 'name', 'latitude', 'longitude'
            start_date: Training start date (YYYY-MM-DD)
            end_date: Training end date (YYYY-MM-DD)
            window_size: Number of historical days to include as context
            chunk_years: Number of years to load in each chunk
            variables: List of weather variables to use
            data_dir: Directory to cache downloaded data
            cleanup_after_chunk: Whether to cleanup data after each chunk
        """
        self.locations = locations
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.window_size = window_size
        self.chunk_years = chunk_years
        self.variables = variables
        self.data_dir = data_dir
        self.cleanup_after_chunk = cleanup_after_chunk

        # Create time chunks
        self.chunks = self._create_time_chunks()
        print(f"Created {len(self.chunks)} time chunks for chunked loading")
        print(f"Cleanup after each chunk: {cleanup_after_chunk}")

    def _create_time_chunks(self) -> List[Tuple[datetime, datetime]]:
        """
        Create time chunks for sequential loading.

        Returns:
            List of (start_date, end_date) tuples for each chunk
        """
        chunks = []
        current_start = self.start_date

        while current_start < self.end_date:
            # Calculate chunk end date
            chunk_end = current_start + pd.DateOffset(years=self.chunk_years)
            # Don't exceed overall end date
            chunk_end = min(chunk_end, self.end_date)

            chunks.append((current_start, chunk_end))
            current_start = chunk_end

        return chunks

    def __iter__(self):
        """
        Iterate through time chunks, yielding a WeatherDataLoader for each.

        Each loader is automatically cleaned up after iteration if cleanup_after_chunk is True.
        """
        for i, (chunk_start, chunk_end) in enumerate(self.chunks):
            print(f"\n{'='*60}")
            print(f"Loading chunk {i+1}/{len(self.chunks)}: {chunk_start.date()} to {chunk_end.date()}")
            print(f"{'='*60}")

            # Create loader for this chunk
            loader = WeatherDataLoader(
                locations=self.locations,
                start_date=chunk_start.strftime("%Y-%m-%d"),
                end_date=chunk_end.strftime("%Y-%m-%d"),
                window_size=self.window_size,
                variables=self.variables,
                data_dir=self.data_dir,
                cache=True,  # Always cache for chunked loading
            )

            # Yield the loader
            yield loader

            # Cleanup after chunk if enabled
            if self.cleanup_after_chunk:
                print(f"\nCleaning up chunk {i+1}/{len(self.chunks)}...")
                loader.cleanup_data(delete_cache_files=True)
                # Force garbage collection
                import gc
                gc.collect()

    def __len__(self) -> int:
        """Return the number of chunks."""
        return len(self.chunks)


class ContinualDataLoader:
    """
    Data loader for continual learning in production.

    Unlike WeatherDataLoader which iterates through historical data,
    this loader fetches the latest data and continues learning.
    """

    def __init__(
        self,
        locations: List[Dict[str, float]],
        window_size: int = 7,
        data_dir: str = "data/production",
    ):
        """
        Initialize the continual learning data loader.

        Args:
            locations: List of location dicts
            window_size: Historical window size
            data_dir: Directory to store live data
        """
        self.locations = locations
        self.window_size = window_size
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.client = OpenMeteoClient()

    def get_today_forecasts(self) -> Dict[str, np.ndarray]:
        """
        Get today's forecasts for all locations.

        Returns:
            Dict mapping location names to forecast arrays
        """
        forecasts = {}
        today = datetime.now().strftime("%Y-%m-%d")

        for loc in self.locations:
            try:
                df = self.client.get_forecast_verification(
                    latitude=loc["latitude"],
                    longitude=loc["longitude"],
                    start_date=today,
                    end_date=today,
                )
                if not df.empty:
                    forecasts[loc["name"]] = df.iloc[-1].values
            except Exception as e:
                print(f"Failed to get forecast for {loc['name']}: {e}")

        return forecasts

    def update_with_actuals(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Update with yesterday's actual weather.

        Returns:
            Dict mapping locations to (forecast, actual) pairs
        """
        pairs = {}
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        for loc in self.locations:
            try:
                df = self.client.get_forecast_verification(
                    latitude=loc["latitude"],
                    longitude=loc["longitude"],
                    start_date=yesterday,
                    end_date=yesterday,
                )
                if not df.empty:
                    # Store for learning
                    self._store_observation(loc["name"], df)
                    pairs[loc["name"]] = (df.iloc[-1].values, df.iloc[-1].values)
            except Exception as e:
                print(f"Failed to update {loc['name']}: {e}")

        return pairs

    def _store_observation(self, location: str, data: pd.DataFrame):
        """Store observation to local file for accumulation."""
        file = self.data_dir / f"{location}.parquet"
        if file.exists():
            existing = pd.read_parquet(file)
            combined = pd.concat([existing, data], ignore_index=True)
        else:
            combined = data

        combined.to_parquet(file, index=False)
