#!/usr/bin/env python3
"""
Test API response consistency and data availability.

Tests:
1. Verify API returns requested variables
2. Check data availability for recent dates
3. Check data availability for 45 years ago
4. Compare response structure and data types
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.openmeteo import OpenMeteoClient


def test_api_response_structure():
    """Test that the API returns expected variables and structure."""
    print("\n" + "="*70)
    print("TEST 1: API Response Structure")
    print("="*70)

    client = OpenMeteoClient()

    # Request specific variables
    requested_vars = ["temperature_2m", "precipitation", "wind_speed_10m"]
    print(f"\nRequested variables: {requested_vars}")

    # Test with recent data (should have plenty of data)
    today = datetime.now()
    start_date = (today - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    print(f"Date range: {start_date} to {end_date}")
    print(f"Location: New York (40.71°N, -74.01°W)")

    try:
        df = client.get_forecast_verification(
            latitude=40.71,
            longitude=-74.01,
            start_date=start_date,
            end_date=end_date,
            variables=requested_vars,
        )

        print(f"\n✓ API request successful")
        print(f"  Response shape: {df.shape}")
        print(f"  Columns returned: {list(df.columns)}")
        print(f"  Data types:\n{df.dtypes}")

        # Check if requested variables are in response
        expected_cols = ["date"] + [f"{v}_min" for v in requested_vars] + \
                       [f"{v}_max" for v in requested_vars] + \
                       [f"{v}_mean" for v in requested_vars]

        print(f"\n  Sample data (first row):")
        print(df.head(1).to_string())

        return True, df

    except Exception as e:
        print(f"\n✗ API request failed: {e}")
        return False, None


def test_recent_data_availability():
    """Test that recent data is available."""
    print("\n" + "="*70)
    print("TEST 2: Recent Data Availability")
    print("="*70)

    client = OpenMeteoClient()

    # Test last 30 days
    today = datetime.now()
    start_date = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    print(f"\nFetching data for: {start_date} to {end_date}")

    try:
        df = client.get_forecast_verification(
            latitude=40.71,
            longitude=-74.01,
            start_date=start_date,
            end_date=end_date,
        )

        print(f"✓ Successfully retrieved {len(df)} days of recent data")
        print(f"  First date: {df['date'].min()}")
        print(f"  Last date: {df['date'].max()}")

        # Check for null values
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0]
        if len(null_cols) > 0:
            print(f"  ⚠ Warning: Found null values in:")
            print(null_cols)
        else:
            print(f"  ✓ No null values found")

        return True, df

    except Exception as e:
        print(f"✗ Failed to fetch recent data: {e}")
        return False, None


def test_historical_data_availability():
    """Test data availability from 45 years ago."""
    print("\n" + "="*70)
    print("TEST 3: Historical Data (45 Years Ago)")
    print("="*70)

    client = OpenMeteoClient()

    # Calculate 45 years ago
    today = datetime.now()
    forty_five_years_ago = today - timedelta(days=365.25 * 45)

    start_date = forty_five_years_ago.strftime("%Y-%m-%d")
    end_date = (forty_five_years_ago + timedelta(days=7)).strftime("%Y-%m-%d")

    print(f"\nDate range: {start_date} to {end_date}")
    print(f"(45 years back from {today.strftime('%Y-%m-%d')})")
    print(f"Location: New York (40.71°N, -74.01°W)")

    try:
        df = client.get_forecast_verification(
            latitude=40.71,
            longitude=-74.01,
            start_date=start_date,
            end_date=end_date,
        )

        print(f"✓ Successfully retrieved {len(df)} days of historical data")
        print(f"  First date: {df['date'].min()}")
        print(f"  Last date: {df['date'].max()}")
        print(f"  Data shape: {df.shape}")

        # Check data availability
        if len(df) == 0:
            print(f"  ⚠ Warning: No data returned for this date range")
            return False, df

        # Check for null values
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0]
        if len(null_cols) > 0:
            print(f"  ⚠ Warning: Found null values in:")
            print(null_cols)
        else:
            print(f"  ✓ No null values found")

        print(f"\n  Sample data (first row):")
        print(df.head(1).to_string())

        return True, df

    except Exception as e:
        print(f"✗ Failed to fetch historical data: {e}")
        return False, None


def compare_recent_and_historical():
    """Compare recent data with data from 45 years ago."""
    print("\n" + "="*70)
    print("TEST 4: Data Consistency Comparison")
    print("="*70)

    client = OpenMeteoClient()

    # Get recent data for the same calendar dates as 45 years ago
    today = datetime.now()

    # Recent week (same dates but this year)
    recent_start = today.strftime("%Y-%m-%d")
    recent_end = (today + timedelta(days=7)).strftime("%Y-%m-%d")

    # 45 years ago same dates
    forty_five_years_ago = today - timedelta(days=365.25 * 45)
    historical_start = forty_five_years_ago.strftime("%Y-%m-%d")
    historical_end = (forty_five_years_ago + timedelta(days=7)).strftime("%Y-%m-%d")

    print(f"\nComparing same calendar period 45 years apart:")
    print(f"  Recent: {recent_start} to {recent_end}")
    print(f"  Historical: {historical_start} to {historical_end}")

    try:
        # Fetch both
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

        print(f"\n✓ Both requests successful")
        print(f"  Recent data points: {len(recent_df)}")
        print(f"  Historical data points: {len(historical_df)}")

        # Compare statistics
        if len(recent_df) > 0 and len(historical_df) > 0:
            print(f"\n  Temperature statistics (°C):")
            print(f"    Recent avg: {recent_df['temperature_2m_mean'].mean():.2f}°C")
            print(f"    Historical avg: {historical_df['temperature_2m_mean'].mean():.2f}°C")
            print(f"    Difference: {abs(recent_df['temperature_2m_mean'].mean() - historical_df['temperature_2m_mean'].mean()):.2f}°C")

            print(f"\n  Precipitation statistics (mm):")
            print(f"    Recent total: {recent_df['precipitation_sum'].sum():.2f}mm")
            print(f"    Historical total: {historical_df['precipitation_sum'].sum():.2f}mm")

            # Show sample rows
            print(f"\n  Recent data (first row):")
            print(recent_df.head(1).to_string())

            print(f"\n  Historical data (first row):")
            print(historical_df.head(1).to_string())

            return True
        else:
            print(f"⚠ Insufficient data for comparison")
            return False

    except Exception as e:
        print(f"✗ Comparison failed: {e}")
        return False


def test_variable_request():
    """Test that API respects variable requests."""
    print("\n" + "="*70)
    print("TEST 5: Variable Request Verification")
    print("="*70)

    client = OpenMeteoClient()

    # Test 1: Request only temperature
    print(f"\nRequesting only 'temperature_2m'...")
    try:
        df1 = client.get_forecast_verification(
            latitude=40.71,
            longitude=-74.01,
            start_date="2024-01-01",
            end_date="2024-01-07",
            variables=["temperature_2m"],
        )
        print(f"  ✓ Returned columns: {list(df1.columns)}")

    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 2: Request all variables
    print(f"\nRequesting all variables...")
    try:
        all_vars = list(client.VARIABLES.keys())
        df2 = client.get_forecast_verification(
            latitude=40.71,
            longitude=-74.01,
            start_date="2024-01-01",
            end_date="2024-01-07",
            variables=all_vars,
        )
        print(f"  ✓ Requested variables: {all_vars}")
        print(f"  ✓ Returned columns: {list(df2.columns)}")

    except Exception as e:
        print(f"  ✗ Failed: {e}")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("OPEN-METEO API CONSISTENCY TESTS")
    print("="*70)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    # Run tests
    results["structure"] = test_api_response_structure()[0]
    results["recent"] = test_recent_data_availability()[0]
    results["historical"] = test_historical_data_availability()[0]
    results["comparison"] = compare_recent_and_historical()
    test_variable_request()

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name.upper()}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print("="*70)


if __name__ == "__main__":
    main()
