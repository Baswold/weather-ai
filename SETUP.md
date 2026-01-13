# Setup Guide: What You Need Before Training

This document outlines what's needed before you can run `train.py` successfully.

## ✓ Already Done

- [x] Dependencies listed in `requirements.txt`
- [x] Data pipeline tested and verified with real API data
- [x] Model architecture available
- [x] API returns correct data for both recent and 45-year-old dates
- [x] Schema is identical between recent and historical data

## 🚀 What to Do Now

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `torch >= 2.0.0` - Deep learning framework
- `pandas >= 2.0.0` - Data handling
- `numpy >= 1.24.0` - Numerical computing
- `requests >= 2.31.0` - API calls (for Open-Meteo)
- `psutil >= 5.9.0` - Memory monitoring
- `tqdm >= 4.66.0` - Progress bars
- `pyarrow >= 14.0.0` - Fast data serialization

### 2. Verify Everything Works

Run the pre-flight checklist:

```bash
python tests/verify_setup.py
```

This will test:
- ✓ All dependencies are installed
- ✓ PyTorch and GPU detection
- ✓ System memory detection
- ✓ Data pipeline (fetch real weather data)
- ✓ Model architecture
- ✓ Forward pass with real data
- ✓ Trainer initialization

**You must pass all checks before proceeding to training.**

### 3. Choose a Training Config

Pick one based on your available memory:

| Config | Memory | Duration | Locations | Details |
|--------|--------|----------|-----------|---------|
| `low_memory` | ~500 MB | 10 mins | 10 (sequential) | All locations, processed one-at-a-time. Slower but learns from all climates with minimal memory. |
| `default` | ~4 GB | 30 mins | 10 (parallel) | All locations, batched together. Faster, good for testing. |
| `production` | ~8 GB | 2+ hours | 20 (parallel) | 15 years data, 20 locations. Full learning on standard hardware. |
| `24gb` | ~24 GB | 6+ hours | 20 (parallel) | Full recent history, large model, big replay buffer. |
| `historical` | ~32 GB | 12+ hours | 20 (parallel) | 75 years of data (1950-2024). Requires powerful hardware. |
| `climate` | ~32 GB | 12+ hours | 20 (parallel) | 55 years climate analysis (1970-2024). Requires powerful hardware. |

### 4. Run Training

**Option A: Let the script ask for your memory**

```bash
python train.py
```

When you run with the default `low_memory` config, the script will ask:
```
Would you like to auto-configure based on your available memory?
(y/N): y

How much RAM can you dedicate to training (in GB)? [default: 12]: 16
```

**Option B: Specify a config directly**

```bash
# Test run with 500 MB memory usage
python train.py --config low_memory --epochs 1

# Full training with 8 GB memory
python train.py --config production --epochs 3

# Maximum utilization of 24 GB
python train.py --config 24gb --epochs 5

# 75 years of historical data (requires 32 GB)
python train.py --config historical --epochs 2 --chunked
```

**Option C: Auto-detect and configure**

```bash
# Use 75% of available memory (recommended)
python train.py --auto

# Use specific amount (e.g., 16 GB)
python train.py --auto --target-gb 16
```

## 📊 Memory Breakdown

### `low_memory` (~500 MB)
- 10 locations (NYC, London, Tokyo, Sydney, Mumbai, Dubai, Singapore, Reykjavik, Denver, LA)
- 1 year of data (2023)
- Full model (128 hidden dims - same as `default`)
- **Sequential processing** - batch_size=1 (one location at a time)
- Replay buffer: 5,000 transitions
- **Unique feature:** Same locations as `default`, but slower and lower memory!

### `production` (~8 GB)
- 20 locations
- 15 years of data (2010-2024)
- Medium model (256 hidden dims)
- Replay buffer: 100,000 transitions

### `24gb` (~24 GB)
- 20 locations
- 9 years of recent data (2016-2024)
- Large model (512 hidden dims)
- Replay buffer: 12,000,000 transitions

## 🎯 First Time Running?

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify setup**
   ```bash
   python tests/verify_setup.py
   ```

3. **Start with low memory config**
   ```bash
   python train.py --config low_memory --epochs 1
   ```

   This will:
   - Download ~50 MB of data
   - Take ~5 minutes
   - Train on 3 locations, 1 year
   - Create a checkpoint you can inspect

4. **Once verified, try auto-config**
   ```bash
   python train.py --auto
   ```

   This will ask how much memory you have and configure everything automatically.

## 🔧 Command-Line Options

```
python train.py [OPTIONS]

Options:
  --config, -c {low_memory|default|production|24gb|extended|historical|climate}
      Configuration preset (default: low_memory)

  --auto, -a
      Auto-tune config based on available memory

  --target-gb FLOAT
      Target memory usage in GB (for --auto)

  --epochs, -e INT
      Number of epochs to train (overrides config)

  --device {auto|cpu|cuda}
      Device to use (default: auto-detect)

  --seed, -s INT
      Random seed for reproducibility

  --no-cache
      Disable data caching

  --chunked
      Use chunked loading (for very large datasets)

  --chunk-years INT
      Years to load per chunk (default: 1)

  --test-only
      Just test pipeline and model, don't train

  --dry-run
      Don't download data or train

Examples:
  python train.py --config low_memory --epochs 1
  python train.py --auto --target-gb 24
  python train.py --config 24gb --epochs 3
  python train.py --config production --device cuda
```

## ⚠️ Common Issues

### "Not enough memory"
- Use a smaller config: `--config low_memory`
- Or specify smaller target: `--auto --target-gb 4`
- Or enable chunked loading: `--chunked --chunk-years 1`

### "No module named X"
- Install dependencies: `pip install -r requirements.txt`
- Or run verification: `python tests/verify_setup.py`

### "CUDA out of memory"
- Switch to CPU: `--device cpu`
- Use smaller config: `--config low_memory`
- The code primarily uses system RAM, not GPU VRAM

### "Data download fails"
- Check internet connection
- API may be rate-limited (wait a few minutes)
- Or use cached data from previous run

## 📈 Understanding Memory Usage

The training pipeline uses memory in several ways:

1. **Replay Buffer** (~50-80% of total)
   - Stores past experiences for replay
   - Larger = better learning, but uses more memory
   - Largest component

2. **Data Cache** (~10-30% of total)
   - Parquet files loaded into memory
   - Can be disabled with `--no-cache`

3. **Model + Gradients** (~5-15% of total)
   - Model weights and activation gradients
   - Larger models use more memory

4. **Training Overhead** (~5-10% of total)
   - Temporary tensors, intermediate computations

**Memory monitoring:**
- Script continuously monitors memory usage
- Warnings at 90% usage
- Critical alert at 100% usage
- If memory maxes out, batch size is reduced automatically

## 🎓 What Happens When You Train

1. **Data Download Phase**
   - Fetches weather data from Open-Meteo API
   - Creates parquet cache files
   - ~50MB per year per location

2. **Data Loading Phase**
   - Loads cached data into memory
   - Creates training batches
   - Displays memory usage

3. **Training Phase**
   - Chronological iteration through weather history
   - For each day:
     - Make prediction (action)
     - Compare to actual (reward)
     - Update model (learning)
     - Store experience (replay buffer)

4. **Checkpoint Saving**
   - Saves model weights periodically
   - Can resume from checkpoint

5. **Summary**
   - Prints final metrics
   - Shows training statistics

## 🚀 Next Steps After Successful Training

Once `verify_setup.py` passes, you can:

1. **Run a quick test:**
   ```bash
   python train.py --config low_memory --epochs 1
   ```

2. **Scale up:**
   ```bash
   python train.py --config production --epochs 3
   ```

3. **Use all your memory:**
   ```bash
   python train.py --auto
   ```

4. **Long-term learning (75 years):**
   ```bash
   python train.py --config historical --epochs 2 --chunked
   ```

5. **View results:**
   - Checkpoints saved in `checkpoints/` directory
   - Metrics logged to console and files
   - Can load and evaluate trained models

---

**Status: Ready to Train! 🎉**

After passing `verify_setup.py`, you have everything needed to train the RL weather model.
