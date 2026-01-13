# ✅ Before You Run Training

## Summary: What Needs to Be Done

You can now run training! But first, follow these 3 steps:

### 1. Install Dependencies (5 mins)
```bash
pip install -r requirements.txt
```

### 2. Verify Setup (2 mins)
```bash
python tests/verify_setup.py
```

This checks:
- ✓ All dependencies installed
- ✓ Data pipeline works (fetches real weather data)
- ✓ Model architecture works
- ✓ GPU/device detection
- ✓ Memory detection

**Must pass all checks before proceeding.**

### 3. Run Training

Choose one:

**Option A: Let script ask for memory (Interactive)**
```bash
python train.py
```
The script will ask: "How much RAM can you dedicate to training (in GB)?"

**Option B: Auto-detect (Recommended)**
```bash
python train.py --auto
```

**Option C: Use preset config**
```bash
python train.py --config low_memory --epochs 1    # ~500MB, 5 mins
python train.py --config production --epochs 3    # ~8GB, 2 hours
python train.py --config 24gb --epochs 5          # ~24GB, 6+ hours
```

---

## What's Already Done ✅

- [x] API integration verified to work
- [x] Data pipeline tested end-to-end
- [x] Schema validated for recent AND 45-year-old data
- [x] Model architecture ready
- [x] Trainer system initialized
- [x] Memory monitoring in place
- [x] Multiple config presets created
- [x] Chunked loading available for large datasets
- [x] vRAM input feature added to `train.py`

---

## What's NOT Done (Not Required to Start)

The TODO.md has many items, but these are **optional enhancements**, not blockers:

❌ *Not required:*
- Baseline training run metrics tracking (can add later)
- Experiment visualization (can add later)
- Multi-location training optimization (works but could be faster)
- Advanced RL techniques like prioritized replay (can add later)
- Historical forecast data integration (using reanalysis instead)
- Distributed training (single GPU enough for now)

✅ *Already sufficient:*
- Core RL training loop works
- Data pipeline functional
- Model can train

---

## Your System Status

**RAM:** 8.0 GB total (1.3 GB available)
- Recommendation: Use `--config low_memory` or `--auto --target-gb 4`

**GPU:** Not available
- Training will use CPU (slower but works)
- Add GPU later for faster training

**API:** ✓ Working (tested with real data)
- Confirmed returns same schema for recent and historical
- Confirmed returns authentic different values 45 years apart

---

## Quick Start (3 Steps)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify everything works
python tests/verify_setup.py

# 3. Start training
python train.py --auto
```

That's it! The script will ask for memory amount and configure automatically.

---

## If Something Fails

**Error: "No module named X"**
- Run: `pip install -r requirements.txt`

**Error: "Not enough memory"**
- Use: `python train.py --config low_memory`
- Or: `python train.py --auto --target-gb 2`

**Error: "Data download fails"**
- Check internet connection
- Wait a minute (API may be rate-limited)
- Cached data will be used if available

**Error: "Can't convert np.ndarray"**
- Run: `python tests/verify_setup.py` first
- This should be fixed now

---

## Understanding the Configs

| Config | RAM | Time | Locations | Processing |
|--------|-----|------|-----------|------------|
| `low_memory` | 500 MB | 10 mins | 10 cities | **Sequential** (one at a time) - Smaller memory footprint |
| `default` | 4 GB | 30 mins | 10 cities | **Parallel** (all in batch) - Faster training |
| `production` | 8 GB | 2+ hours | 20 cities | Parallel, 15 years of data |
| `24gb` | 24 GB | 6+ hours | 20 cities | Parallel, large model, 9 years |
| `historical` | 32+ GB | 12+ hours | 20 cities | Parallel, 75 years (1950-2024) |

**Key difference in `low_memory`:**
- Uses ALL 10 locations (diverse climates: NYC, London, Tokyo, Sydney, Mumbai, Dubai, Singapore, Reykjavik, Denver, LA)
- Processes **one location at a time** (batch_size=1) instead of all together
- Smaller replay buffer (5,000 vs 50,000+)
- Same model architecture - NOT downscaled
- Result: **~500MB RAM vs 4GB**, but still learns from all climates ✅

**For your 8GB system:**
- Recommended: `--config low_memory` (will use ~500MB even with all 10 locations!)
- Alternative: `--auto --target-gb 4` to use half your RAM
- Can also run: `--config default` with memory monitoring

---

## What Happens When You Run

1. **Load config** - Use the one you specified
2. **Ask for memory** - If using interactive mode
3. **Download data** - From Open-Meteo API (~50MB per location-year)
4. **Initialize model** - Create the neural network
5. **Start training** - Iterate through weather history chronologically
6. **Save checkpoints** - Model weights saved periodically
7. **Print summary** - Final metrics and statistics

Total time:
- `low_memory`: ~5 minutes
- `production`: ~2 hours
- `24gb`: ~6 hours

---

## Next Steps After Training

Once a run completes:

1. **Check results** - Look in `checkpoints/` directory
2. **Load model** - Use saved weights for inference
3. **Analyze metrics** - Review reward history
4. **Scale up** - Try larger config on next run
5. **Experiment** - Modify hyperparameters

---

## Need Help?

- **API working?** → Run `python tests/test_api_consistency.py`
- **Data format?** → Run `python tests/test_schema_comparison.py`
- **Memory issues?** → Use smaller config
- **Want details?** → Read `SETUP.md`
- **About the project?** → Read `CONCEPT.md`

---

**Status: 🟢 Ready to Train**

Your system is set up and ready. Run `python tests/verify_setup.py` to confirm, then start training!
