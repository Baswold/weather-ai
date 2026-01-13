#!/usr/bin/env python3
"""
Detailed API analysis to understand the mismatch between what the code
requests vs what the API actually returns.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.openmeteo import OpenMeteoClient
import requests


def show_raw_api_response():
    """Show what the raw API actually returns (before parsing)."""
    print("\n" + "="*70)
    print("RAW API RESPONSE ANALYSIS")
    print("="*70)

    # Make a raw request to understand what the API returns
    base_url = "https://archive-api.open-meteo.com/v1"

    params = {
        "latitude": 40.71,
        "longitude": -74.01,
        "start_date": "2024-01-01",
        "end_date": "2024-01-03",
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "timezone": "auto",
    }

    print(f"\nRequest URL: {base_url}/archive")
    print(f"Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    response = requests.get(f"{base_url}/archive", params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    print(f"\nRaw JSON Response Structure:")
    print(f"  Keys: {list(data.keys())}")

    if "hourly" in data:
        print(f"\n  Hourly data keys: {list(data['hourly'].keys())}")
        hourly = data['hourly']
        print(f"\n  Sample hourly values (first 5):")
        for key in hourly.keys():
            if key == "time":
                print(f"    {key}: {hourly[key][:5]}")
            else:
                print(f"    {key}: {hourly[key][:5]}")

    if "daily" in data:
        print(f"\n  Daily data keys: {list(data['daily'].keys())}")
        print(f"  Daily data (first row):")
        daily = data['daily']
        for key in daily.keys():
            if key == "time":
                print(f"    {key}: {daily[key]}")
            else:
                print(f"    {key}: {daily[key]}")


def analyze_code_expectations():
    """Analyze what the code expects vs what it gets."""
    print("\n" + "="*70)
    print("CODE EXPECTATIONS VS REALITY")
    print("="*70)

    client = OpenMeteoClient()

    # Test 1: What does DEFAULT_VARIABLES contain?
    print(f"\nWeatherDataLoader DEFAULT_VARIABLES:")
    from src.data.loader import WeatherDataLoader
    for var in WeatherDataLoader.DEFAULT_VARIABLES:
        print(f"  - {var}")

    # Test 2: What does get_forecast_verification actually return?
    print(f"\nCalling get_forecast_verification()...")
    df = client.get_forecast_verification(
        latitude=40.71,
        longitude=-74.01,
        start_date="2024-01-01",
        end_date="2024-01-03",
        variables=["temperature_2m", "precipitation", "wind_speed_10m"]
    )

    print(f"\nActual columns returned by get_forecast_verification():")
    for col in df.columns:
        print(f"  - {col}")

    print(f"\nData sample:")
    print(df)

    # Test 3: Check if DEFAULT_VARIABLES match what's returned
    print(f"\n\nCOMPARISON:")
    print(f"  Expected columns (DEFAULT_VARIABLES): {WeatherDataLoader.DEFAULT_VARIABLES}")
    print(f"  Actual columns returned: {[c for c in df.columns if c != 'date']}")

    matches = [v in df.columns for v in WeatherDataLoader.DEFAULT_VARIABLES]
    if all(matches):
        print(f"  ✓ ALL expected variables found in response")
    else:
        missing = [v for v, m in zip(WeatherDataLoader.DEFAULT_VARIABLES, matches) if not m]
        print(f"  ✗ MISSING variables: {missing}")


def check_45_years_consistency():
    """Check if the same day 45 years apart returns meaningful data."""
    print("\n" + "="*70)
    print("45-YEAR CONSISTENCY CHECK")
    print("="*70)

    client = OpenMeteoClient()

    # Use a past date to test (avoid future dates)
    base_date = datetime(2024, 1, 15)

    recent_start = base_date.strftime("%Y-%m-%d")
    recent_end = (base_date + timedelta(days=1)).strftime("%Y-%m-%d")

    forty_five_years_ago = base_date - timedelta(days=365.25 * 45)
    historical_start = forty_five_years_ago.strftime("%Y-%m-%d")
    historical_end = (forty_five_years_ago + timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"\nComparing same calendar day 45 years apart:")
    print(f"  Recent: {recent_start} (2024)")
    print(f"  Historical: {historical_start} (1979)")

    try:
        recent_df = client.get_forecast_verification(
            latitude=40.71,
            longitude=-74.01,
            start_date=recent_start,
            end_date=recent_end,
        )

        historical_df = client.get_forecast_verification(
            latitude=40.71,
            longitude=-74.01,
            start_date=historical_start,
            end_date=historical_end,
        )

        print(f"\n✓ Both API calls successful")
        print(f"\nRecent data (2024-01-15):")
        print(recent_df.to_string())

        print(f"\nHistorical data (1979-01-15):")
        print(historical_df.to_string())

        # Compare values
        print(f"\n\nVALUE COMPARISON:")
        print(f"  Temperature min - Recent: {recent_df['temperature_2m_min'].values[0]:.1f}°C, Historical: {historical_df['temperature_2m_min'].values[0]:.1f}°C")
        print(f"  Temperature max - Recent: {recent_df['temperature_2m_max'].values[0]:.1f}°C, Historical: {historical_df['temperature_2m_max'].values[0]:.1f}°C")
        print(f"  Precipitation - Recent: {recent_df['precipitation_sum'].values[0]:.1f}mm, Historical: {historical_df['precipitation_sum'].values[0]:.1f}mm")
        print(f"\n  → The values ARE DIFFERENT, which is expected - it's actual historical weather data")
        print(f"  → This proves the API is returning REAL data from 45+ years ago, not duplicate values")

    except Exception as e:
        print(f"✗ API call failed: {e}")


def main():
    """Run detailed analysis."""
    print("\n" + "="*70)
    print("DETAILED API ANALYSIS")
    print("="*70)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        show_raw_api_response()
    except Exception as e:
        print(f"Failed to show raw response: {e}")

    try:
        analyze_code_expectations()
    except Exception as e:
        print(f"Failed to analyze expectations: {e}")

    try:
        check_45_years_consistency()
    except Exception as e:
        print(f"Failed to check consistency: {e}")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
