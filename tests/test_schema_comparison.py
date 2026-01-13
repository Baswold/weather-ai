#!/usr/bin/env python3
"""
Test if the schema (column structure) is identical between recent and historical data.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.openmeteo import OpenMeteoClient


def compare_schemas():
    """Compare the exact schema between recent and 45-year-old data."""
    print("\n" + "="*70)
    print("SCHEMA COMPARISON: RECENT vs 45 YEARS AGO")
    print("="*70)

    client = OpenMeteoClient()

    # Recent data (2024)
    recent_df = client.get_forecast_verification(
        latitude=40.71,
        longitude=-74.01,
        start_date="2024-01-15",
        end_date="2024-01-20",
    )

    # Historical data (1979 - 45 years ago)
    historical_df = client.get_forecast_verification(
        latitude=40.71,
        longitude=-74.01,
        start_date="1979-01-15",
        end_date="1979-01-20",
    )

    print("\n" + "="*70)
    print("RECENT DATA (2024)")
    print("="*70)
    print(f"\nColumns: {list(recent_df.columns)}")
    print(f"Data types:\n{recent_df.dtypes}")
    print(f"\nShape: {recent_df.shape}")
    print(f"\nFirst row:")
    print(recent_df.head(1))

    print("\n" + "="*70)
    print("HISTORICAL DATA (1979)")
    print("="*70)
    print(f"\nColumns: {list(historical_df.columns)}")
    print(f"Data types:\n{historical_df.dtypes}")
    print(f"\nShape: {historical_df.shape}")
    print(f"\nFirst row:")
    print(historical_df.head(1))

    print("\n" + "="*70)
    print("SCHEMA COMPARISON RESULTS")
    print("="*70)

    # Compare column names
    recent_cols = set(recent_df.columns)
    historical_cols = set(historical_df.columns)

    print(f"\nColumns in RECENT (2024): {sorted(recent_cols)}")
    print(f"Columns in HISTORICAL (1979): {sorted(historical_cols)}")

    if recent_cols == historical_cols:
        print(f"\n✓ COLUMN NAMES MATCH")
    else:
        print(f"\n✗ COLUMN NAMES DIFFER")
        only_recent = recent_cols - historical_cols
        only_historical = historical_cols - recent_cols
        if only_recent:
            print(f"  Only in recent: {only_recent}")
        if only_historical:
            print(f"  Only in historical: {only_historical}")

    # Compare data types
    print(f"\nData Type Comparison:")
    for col in sorted(recent_cols & historical_cols):
        recent_dtype = recent_df[col].dtype
        historical_dtype = historical_df[col].dtype
        match = "✓" if recent_dtype == historical_dtype else "✗"
        print(f"  {col}: {match} Recent={recent_dtype}, Historical={historical_dtype}")

    # Check if dtypes all match
    if recent_df.dtypes.equals(historical_df[recent_df.columns].dtypes):
        print(f"\n✓ ALL DATA TYPES MATCH")
    else:
        print(f"\n✗ SOME DATA TYPES DIFFER")

    # Overall result
    print(f"\n" + "="*70)
    if recent_cols == historical_cols and recent_df.dtypes.equals(historical_df[recent_df.columns].dtypes):
        print("✓ SCHEMAS ARE IDENTICAL")
        print("  Same columns, same data types")
        print("  → Code can process both with no changes needed")
    else:
        print("✗ SCHEMAS ARE DIFFERENT")
        print("  → May need conditional handling or schema conversion")
    print("="*70)


if __name__ == "__main__":
    compare_schemas()
