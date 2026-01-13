#!/usr/bin/env python3
"""
Test script to verify chunked data loading and cleanup.

This script tests that:
1. Data is loaded in chunks
2. Memory is freed after each chunk
3. Cache files are deleted after each chunk
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data import ChunkedWeatherDataLoader
import os
import psutil


def get_directory_size(path):
    """Calculate total size of all files in directory."""
    total = 0
    if Path(path).exists():
        for file in Path(path).glob("**/*"):
            if file.is_file():
                total += file.stat().st_size
    return total / (1024 * 1024)  # Convert to MB


def main():
    print("Testing Chunked Data Loader with Cleanup")
    print("=" * 60)

    # Test configuration: small dataset
    locations = [
        {"name": "Test_NYC", "latitude": 40.71, "longitude": -74.01},
        {"name": "Test_LA", "latitude": 34.05, "longitude": -118.24},
    ]

    # Test with 2 years, split into 1-year chunks
    start_date = "2023-01-01"
    end_date = "2024-12-31"
    chunk_years = 1

    data_dir = "data/test_chunked"
    os.makedirs(data_dir, exist_ok=True)

    print(f"\nTest configuration:")
    print(f"  Locations: {len(locations)}")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Chunk size: {chunk_years} year(s)")
    print(f"  Data directory: {data_dir}")

    # Get initial memory and disk usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
    initial_disk = get_directory_size(data_dir)

    print(f"\nInitial state:")
    print(f"  Memory usage: {initial_memory:.2f} MB")
    print(f"  Disk usage: {initial_disk:.2f} MB")

    # Create chunked loader
    chunked_loader = ChunkedWeatherDataLoader(
        locations=locations,
        start_date=start_date,
        end_date=end_date,
        chunk_years=chunk_years,
        data_dir=data_dir,
        cleanup_after_chunk=True,
    )

    print(f"\nIterating through {len(chunked_loader)} chunks...")

    chunk_stats = []

    for chunk_idx, chunk_loader in enumerate(chunked_loader):
        print(f"\n{'='*60}")
        print(f"Processing chunk {chunk_idx + 1}/{len(chunked_loader)}")

        # Count days in this chunk
        num_days = len(list(chunk_loader))
        print(f"  Days in chunk: {num_days}")

        # Measure memory and disk after loading
        after_load_memory = process.memory_info().rss / (1024 * 1024)
        after_load_disk = get_directory_size(data_dir)

        print(f"  After loading:")
        print(f"    Memory: {after_load_memory:.2f} MB (+{after_load_memory - initial_memory:.2f} MB)")
        print(f"    Disk: {after_load_disk:.2f} MB (+{after_load_disk - initial_disk:.2f} MB)")

        chunk_stats.append({
            "chunk": chunk_idx + 1,
            "days": num_days,
            "memory_before_cleanup": after_load_memory,
            "disk_before_cleanup": after_load_disk,
        })

        # Cleanup happens automatically here when we exit the loop iteration

        # Measure after cleanup
        import time
        time.sleep(0.5)  # Give time for cleanup to complete

        after_cleanup_memory = process.memory_info().rss / (1024 * 1024)
        after_cleanup_disk = get_directory_size(data_dir)

        print(f"  After cleanup:")
        print(f"    Memory: {after_cleanup_memory:.2f} MB (freed {after_load_memory - after_cleanup_memory:.2f} MB)")
        print(f"    Disk: {after_cleanup_disk:.2f} MB (freed {after_load_disk - after_cleanup_disk:.2f} MB)")

        chunk_stats[-1]["memory_after_cleanup"] = after_cleanup_memory
        chunk_stats[-1]["disk_after_cleanup"] = after_cleanup_disk

    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    final_memory = process.memory_info().rss / (1024 * 1024)
    final_disk = get_directory_size(data_dir)

    print(f"\nFinal state:")
    print(f"  Memory usage: {final_memory:.2f} MB")
    print(f"  Disk usage: {final_disk:.2f} MB")
    print(f"\nTotal changes:")
    print(f"  Memory delta: {final_memory - initial_memory:+.2f} MB")
    print(f"  Disk delta: {final_disk - initial_disk:+.2f} MB")

    print(f"\nPer-chunk statistics:")
    for stat in chunk_stats:
        memory_freed = stat["memory_before_cleanup"] - stat.get("memory_after_cleanup", 0)
        disk_freed = stat["disk_before_cleanup"] - stat.get("disk_after_cleanup", 0)
        print(f"  Chunk {stat['chunk']}: {stat['days']} days, freed {memory_freed:.2f} MB RAM, {disk_freed:.2f} MB disk")

    # Verify cleanup worked
    print(f"\n{'='*60}")
    if final_disk < 5:  # Less than 5MB left
        print("✓ SUCCESS: Cleanup is working! Disk usage is minimal.")
    else:
        print("✗ WARNING: Some cache files may not have been cleaned up.")

    if final_memory < initial_memory + 100:  # Less than 100MB increase
        print("✓ SUCCESS: Memory cleanup is working!")
    else:
        print("✗ WARNING: Memory usage increased significantly.")

    print(f"{'='*60}")

    # Cleanup test directory
    import shutil
    if Path(data_dir).exists():
        shutil.rmtree(data_dir)
        print(f"\nCleaned up test directory: {data_dir}")


if __name__ == "__main__":
    main()
