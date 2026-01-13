# Test Suite - RL Weather

All tests for the RL Weather project.

## Quick Test

Run all pre-flight checks:
```bash
python tests/verify_setup.py
```

This verifies:
- ✓ All dependencies installed
- ✓ PyTorch and GPU detection
- ✓ Data pipeline works
- ✓ Model architecture correct
- ✓ Forward pass successful
- ✓ Trainer initialization

## Individual Tests

### 1. verify_setup.py
**Purpose:** Pre-flight checklist before training

**What it tests:**
- Dependencies (torch, pandas, numpy, requests, psutil, tqdm)
- PyTorch installation and GPU detection
- System memory availability
- Data pipeline (fetch real weather data)
- WeatherDataLoader functionality
- Model architecture
- Forward pass with real data
- Trainer initialization

**Usage:**
```bash
python tests/verify_setup.py
```

**Expected output:** "✓ ALL PRE-FLIGHT CHECKS PASSED"

---

### 2. test_api_consistency.py
**Purpose:** Verify API returns correct values and proper data aggregation

**What it tests:**
- API response structure (correct columns)
- Recent data availability (last 30 days)
- Historical data availability (45 years ago)
- Data consistency between recent and historical
- Variable request handling
- Null value checks

**Usage:**
```bash
python tests/test_api_consistency.py
```

**Expected findings:**
- ✓ API response successful
- ✓ All requested variables returned
- ✓ Recent data: 31 days available, no nulls
- ✓ Historical data: 45 years available, no nulls
- ✓ Data consistency validated

---

### 3. test_api_details.py
**Purpose:** Deep dive into API structure and data flow

**What it tests:**
- Raw API response structure
- JSON keys and data types
- Code expectations vs API reality
- Hourly to daily aggregation
- Temperature consistency across time periods

**Usage:**
```bash
python tests/test_api_details.py
```

**What it shows:**
```
Raw JSON Response Structure:
  Keys: ['latitude', 'longitude', 'hourly', 'timezone', ...]

Hourly data keys: ['time', 'temperature_2m', 'precipitation', 'wind_speed_10m']

Sample values:
  time: ['2024-01-01T00:00', '2024-01-01T01:00', ...]
  temperature_2m: [1.8, 2.8, 2.9, ...]

Code aggregation:
  Converts 24 hourly values → 1 daily aggregate (min, max, mean, sum, etc.)
```

---

### 4. test_schema_comparison.py
**Purpose:** Validate schema consistency between recent and 45-year-old data

**What it tests:**
- Column names match
- Data types match
- Structure consistency
- Null value handling

**Usage:**
```bash
python tests/test_schema_comparison.py
```

**Expected output:**
```
✓ SCHEMAS ARE IDENTICAL
  Same columns, same data types
  → Code can process both with no changes needed
```

---

### 5. test_api.py
**Purpose:** Basic API connection test

**What it tests:**
- Simple API request
- Response parsing

**Usage:**
```bash
python tests/test_api.py
```

---

### 6. test_chunked_loader.py
**Purpose:** Test chunked data loading for large datasets

**What it tests:**
- ChunkedWeatherDataLoader functionality
- Time chunk creation
- Sequential chunk iteration
- Cleanup between chunks
- Memory efficiency

**Usage:**
```bash
python tests/test_chunked_loader.py
```

---

## Running Tests

### Before Training (Required)
```bash
python tests/verify_setup.py
```

### Verify API Works
```bash
python tests/test_api_consistency.py
```

### Validate Data Structure
```bash
python tests/test_schema_comparison.py
```

### Deep Dive Analysis
```bash
python tests/test_api_details.py
```

### All Tests
```bash
for test in tests/verify_setup.py tests/test_*.py; do
    echo "Running $test..."
    python "$test"
    echo ""
done
```

## Test Results Summary

| Test | Status | Purpose |
|------|--------|---------|
| verify_setup.py | ✓ PASS | Pre-flight checklist |
| test_api_consistency.py | ✓ PASS | API response validation |
| test_api_details.py | ✓ PASS | API structure analysis |
| test_schema_comparison.py | ✓ PASS | Schema validation |
| test_api.py | ✓ PASS | Basic API test |
| test_chunked_loader.py | ✓ PASS | Chunked loading |

## Troubleshooting

### ImportError: No module named 'torch'
```bash
pip install -r requirements.txt
```

### ConnectionError: API unreachable
- Check internet connection
- Wait a minute (rate limit)
- Try again

### Memory errors during tests
- Run on a machine with more RAM
- Or wait for previous jobs to complete

### CUDA/GPU errors
- Install CUDA-compatible PyTorch
- Or use CPU (slower but works)
- See: https://pytorch.org/get-started/locally/

## Adding New Tests

1. Create new file: `tests/test_name.py`
2. Import required modules
3. Add docstrings
4. Print clear output
5. Exit with status 0 if pass, 1 if fail

Example:
```python
#!/usr/bin/env python3
"""Test description."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("Testing...")
# Your test code
print("✓ Test passed!")
```

---

**Status: All tests pass! Ready to train! 🚀**
