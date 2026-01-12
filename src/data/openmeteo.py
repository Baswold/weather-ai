"""
Open-Meteo API client for historical weather forecasts and observations.

API Documentation: https://open-meteo.com/en/docs/historical-weather-api

The Historical Weather API (Archive API) provides:
- ERA5 reanalysis data from 1940 onwards (80+ years of data)
- ERA5-Land data from 1950 onwards (higher resolution)
- Actual observations (what actually happened)
- Global coverage at 10-25km resolution
- Various weather variables: temperature, precipitation, wind, etc.

Example usage:
    client = OpenMeteoClient()
    data = client.get_forecast_and_actual(
        latitude=40.71,
        longitude=-74.01,
        start_date="2020-01-01",
        end_date="2020-01-31"
    )
"""

import datetime
from typing import Optional, List, Dict, Any
import requests
import numpy as np
import pandas as pd


class OpenMeteoClient:
    """
    Client for Open-Meteo Archive API (Historical Weather).

    Provides free access to:
    - ERA5 reanalysis data from 1940 onwards (80+ years)
    - ERA5-Land data from 1950 onwards (higher resolution)
    - Actual observations for comparison
    - No API key required
    - Global coverage at 10-25km resolution
    """

    BASE_URL = "https://archive-api.open-meteo.com/v1"

    # Available weather variables
    VARIABLES = {
        "temperature_2m": "Air temperature at 2 meters above ground (°C)",
        "relative_humidity_2m": "Relative humidity at 2 meters (%)",
        "precipitation": "Total precipitation (mm)",
        "snowfall": "Snowfall amount (cm)",
        "surface_pressure": "Surface pressure (hPa)",
        "cloud_cover": "Total cloud cover (%)",
        "wind_speed_10m": "Wind speed at 10 meters (km/h)",
        "wind_direction_10m": "Wind direction at 10 meters (degrees)",
    }

    # Forecast-specific variables (what was predicted)
    FORECAST_VARIABLES = {
        f"{k}_forecast": f"Forecasted {v}" for k, v in VARIABLES.items()
    }

    def __init__(self, timeout: int = 30):
        """
        Initialize the Open-Meteo client.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "RL-Weather/1.0"})

    def get_forecast_and_actual(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        variables: Optional[List[str]] = None,
        forecast_horizon: int = 24,
    ) -> pd.DataFrame:
        """
        Fetch both forecast and actual observations for a location and date range.

        Args:
            latitude: Location latitude
            longitude: Location longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            variables: List of weather variables to fetch (default: all)
            forecast_horizon: Hours ahead for forecast (default: 24h = next day)

        Returns:
            DataFrame with columns for both forecast and actual values
        """
        if variables is None:
            variables = list(self.VARIABLES.keys())

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(variables),
            "timezone": "auto",
        }

        # Fetch the data
        response = self.session.get(
            f"{self.BASE_URL}/archive",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        # Parse into DataFrame
        df = self._parse_response(data, forecast_horizon)

        return df

    def get_forecast_verification(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        variables: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Fetch forecast-actual pairs for verification.

        This is the key function for RL training - it gets pairs of:
        - What was forecasted for day D
        - What actually happened on day D

        Args:
            latitude: Location latitude
            longitude: Location longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            variables: List of weather variables

        Returns:
            DataFrame with forecast and actual columns for each variable
        """
        if variables is None:
            variables = ["temperature_2m", "precipitation", "wind_speed_10m"]

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(variables),
            "timezone": "auto",
            "forecast_days": 1,  # Only need next day forecast
        }

        response = self.session.get(
            f"{self.BASE_URL}/archive",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        return self._parse_verification_data(data, variables)

    def _parse_response(self, data: Dict[str, Any], horizon: int) -> pd.DataFrame:
        """Parse API response into a DataFrame."""
        hourly = data.get("hourly", {})
        if not hourly:
            raise ValueError("No hourly data in response")

        df = pd.DataFrame({k: v for k, v in hourly.items() if k != "time"})
        df["time"] = pd.to_datetime(hourly["time"])
        df = df.set_index("time")

        return df

    def _parse_verification_data(
        self, data: Dict[str, Any], variables: List[str]
    ) -> pd.DataFrame:
        """
        Parse forecast verification data into paired format.

        Returns a DataFrame where each row represents a day with:
        - Forecast columns: temp_forecast_max, temp_forecast_min, etc.
        - Actual columns: temp_actual_max, temp_actual_min, etc.
        """
        hourly = data.get("hourly", {})
        if not hourly:
            raise ValueError("No hourly data in response")

        df = pd.DataFrame({k: v for k, v in hourly.items() if k != "time"})
        df["time"] = pd.to_datetime(hourly["time"])

        # Aggregate to daily values
        df["date"] = df["time"].dt.date
        daily = df.groupby("date").agg({
            "temperature_2m": ["min", "max", "mean"],
            "precipitation": "sum",
            "wind_speed_10m": ["max", "mean"],
        })
        daily.columns = ["_".join(col).strip() for col in daily.columns.values]
        daily = daily.reset_index()
        daily["date"] = pd.to_datetime(daily["date"])

        return daily

    def get_multiple_locations(
        self,
        locations: List[Dict[str, float]],
        start_date: str,
        end_date: str,
        variables: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple locations.

        Args:
            locations: List of dicts with 'name', 'latitude', 'longitude'
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            variables: List of weather variables

        Returns:
            Dict mapping location names to DataFrames
        """
        results = {}
        for loc in locations:
            name = loc.get("name", f"{loc['latitude']},{loc['longitude']}")
            try:
                df = self.get_forecast_verification(
                    latitude=loc["latitude"],
                    longitude=loc["longitude"],
                    start_date=start_date,
                    end_date=end_date,
                    variables=variables,
                )
                results[name] = df
            except Exception as e:
                print(f"Warning: Failed to fetch data for {name}: {e}")

        return results

    @staticmethod
    def get_sample_locations() -> List[Dict[str, float]]:
        """
        Get a diverse set of sample locations for initial testing.

        Returns locations across different climate zones:
        - Tropical, Desert, Temperate, Continental, Polar, Coastal, Mountain
        """
        return [
            {"name": "New_York", "latitude": 40.71, "longitude": -74.01, "climate": "continental"},
            {"name": "London", "latitude": 51.51, "longitude": -0.13, "climate": "temperate"},
            {"name": "Tokyo", "latitude": 35.68, "longitude": 139.77, "climate": "subtropical"},
            {"name": "Sydney", "latitude": -33.87, "longitude": 151.21, "climate": "subtropical"},
            {"name": "Mumbai", "latitude": 19.08, "longitude": 72.88, "climate": "tropical"},
            {"name": "Dubai", "latitude": 25.20, "longitude": 55.27, "climate": "desert"},
            {"name": "Cairo", "latitude": 30.04, "longitude": 31.24, "climate": "desert"},
            {"name": "Singapore", "latitude": 1.35, "longitude": 103.82, "climate": "tropical"},
            {"name": "Reykjavik", "latitude": 64.15, "longitude": -21.95, "climate": "polar"},
            {"name": "Denver", "latitude": 39.74, "longitude": -104.99, "climate": "mountain"},
            {"name": "Los_Angeles", "latitude": 34.05, "longitude": -118.24, "climate": "mediterranean"},
            {"name": "Miami", "latitude": 25.76, "longitude": -80.19, "climate": "tropical"},
            {"name": "Seattle", "latitude": 47.61, "longitude": -122.33, "climate": "temperate"},
            {"name": "Moscow", "latitude": 55.76, "longitude": 37.62, "climate": "continental"},
            {"name": "Sao_Paulo", "latitude": -23.55, "longitude": -46.64, "climate": "subtropical"},
            {"name": "Cape_Town", "latitude": -33.93, "longitude": 18.42, "climate": "mediterranean"},
            {"name": "Beijing", "latitude": 39.90, "longitude": 116.41, "climate": "continental"},
            {"name": "Mumbai", "latitude": 19.08, "longitude": 72.88, "climate": "tropical"},
            {"name": "Honolulu", "latitude": 21.31, "longitude": -157.86, "climate": "tropical"},
            {"name": "Anchorage", "latitude": 61.22, "longitude": -149.90, "climate": "polar"},
        ]


def test_connection():
    """Test the API connection with a simple request."""
    client = OpenMeteoClient()

    # Test with a small date range
    df = client.get_forecast_verification(
        latitude=40.71,
        longitude=-74.01,
        start_date="2024-01-01",
        end_date="2024-01-07",
    )

    print("API Connection Test:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\n{df.head()}")

    return df


if __name__ == "__main__":
    test_connection()
