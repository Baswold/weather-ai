#!/usr/bin/env python3
"""
Simple test script for the Open-Meteo API.

This script tests the data connection without running the full training.
Useful for verifying that the API works and understanding the data format.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.openmeteo import OpenMeteoClient


def test_basic_request():
    """Test a basic API request."""
    print("Testing Open-Meteo API connection...")
    print("=" * 50)

    client = OpenMeteoClient()

    # Test with New York City
    print("\nFetching data for New York City (Jan 1-7, 2024)...")
    df = client.get_forecast_verification(
        latitude=40.71,
        longitude=-74.01,
        start_date="2024-01-01",
        end_date="2024-01-07",
    )

    print(f"\nSuccess! Retrieved {len(df)} days of data")
    print(f"Columns: {list(df.columns)}")
    print("\nSample data:")
    print(df)

    return df


def test_multiple_locations():
    """Test fetching data for multiple locations."""
    print("\n" + "=" * 50)
    print("Testing multiple locations...")

    client = OpenMeteoClient()
    locations = client.get_sample_locations()[:5]

    print(f"\nFetching data for {len(locations)} locations...")

    results = client.get_multiple_locations(
        locations=locations,
        start_date="2024-01-01",
        end_date="2024-01-03",
    )

    print(f"\nSuccessfully fetched {len(results)} locations:")
    for name, df in results.items():
        print(f"  - {name}: {len(df)} days")


def show_available_variables():
    """Show available weather variables."""
    print("\n" + "=" * 50)
    print("Available Weather Variables")
    print("=" * 50)

    client = OpenMeteoClient()
    for var, desc in client.VARIABLES.items():
        print(f"  {var}: {desc}")


if __name__ == "__main__":
    show_available_variables()
    test_basic_request()
    test_multiple_locations()

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
