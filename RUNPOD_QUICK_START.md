# RunPod Quick Start Guide

## One-Command Setup

Copy and paste this into your RunPod terminal to set up and start training:

```bash
git clone https://github.com/Baswold/weather-ai.git && cd weather-ai && bash runpod_setup.sh
```

That's it! The script will:
1. ✓ Clone the repository
2. ✓ Install all dependencies
3. ✓ Detect your GPU
4. ✓ Auto-configure for your GPU memory
5. ✓ Optionally start training

## What Happens

### Automatic GPU Detection
- Detects GPU type and memory
- Shows VRAM available
- Recommends optimal config
- Auto-configures training parameters

### Configuration by GPU Memory

| GPU Memory | Recommended Config | Locations | Data Years | Est. Time |
|------------|-------------------|-----------|-----------|-----------|
| < 10 GB | `default` | 10 | 5 | 30 mins |
| 10-24 GB | `production` | 20 | 15 | 2-3 hours |
| 24+ GB | `24gb` | 20 | 9 | 6+ hours |
| 40+ GB | `24gb` | 20 | 9 | 6+ hours |

## Running Manually (If Not Auto-Starting)

After the script completes:

```bash
# Basic training (auto-configured for your GPU)
cd weather-ai
python train.py --auto --epochs 3 --chunked

# Or choose specific config
python train.py --config 24gb --epochs 5
python train.py --config production --epochs 3
```

## Monitoring

### GPU Usage
```bash
watch -n 1 'nvidia-smi'
```

### Memory Usage
```bash
watch -n 1 'free -h'
```

### Training Progress
```bash
tail -f checkpoints/*/training.log
```

## What Gets Trained

### low_memory (~500 MB, 10 mins)
- 10 locations (diverse climates)
- 1 year of data (2023)
- Sequential processing

### default (~4 GB, 30 mins)
- 10 locations
- 5 years of data (2020-2024)
- Parallel processing

### production (~8 GB, 2+ hours)
- 20 locations
- 15 years of data (2010-2024)
- Best for learning

### 24gb (~24 GB, 6+ hours)
- 20 locations
- 9 years of recent data (2016-2024)
- Large model
- Recommended for GPU with 24GB+ VRAM

## Key Features

### ✓ GPU Optimized
- Uses PyTorch with CUDA for maximum speed
- Automatically uses all available GPU VRAM
- Falls back to CPU if no GPU detected

### ✓ Smart Memory Management
- Chunked loading to prevent storage buildup
- Memory monitoring with warnings
- Auto-cleanup of intermediate data

### ✓ Data Source
- Real weather data from Open-Meteo API
- 80+ years of historical data (1940-2024)
- Global coverage (1000+ locations available)
- Multiple weather variables

### ✓ Pre-Verified
- All API tests pass
- Data schema validated
- Model architecture tested
- Ready to train immediately

## Example Commands

### Fastest Setup (Auto-configured)
```bash
python train.py --auto --epochs 3
```

### Use All GPU Memory
```bash
python train.py --config 24gb --epochs 5
```

### Production Run (Recommended)
```bash
python train.py --config production --epochs 3 --chunked
```

### Extended History (75 years)
```bash
python train.py --config historical --epochs 2 --chunked
```

### Test First (5 mins)
```bash
python train.py --config low_memory --epochs 1
```

## Troubleshooting

### Script Won't Run
```bash
chmod +x runpod_setup.sh
bash runpod_setup.sh
```

### GPU Not Detected
- Script falls back to CPU
- Training will be slower but still works

### Out of Memory
- Reduce batch size: edit config in `configs/default.py`
- Use smaller config: `--config production`
- Enable chunked loading: `--chunked`

### Data Download Fails
- Check internet connection
- API may rate-limit (wait 1 minute)
- Cached data used automatically on retry

## After Training

Model checkpoints saved in `checkpoints/` directory:
- `checkpoints/config.json` - Training configuration
- `checkpoints/model_*.pt` - Model weights at different epochs
- `checkpoints/final/` - Final trained model

## Performance Expectations

### Training Speed
- **A100 (40GB):** ~50-100k transitions/min
- **RTX 4090 (24GB):** ~30-50k transitions/min
- **RTX 3090 (24GB):** ~20-30k transitions/min
- **V100 (32GB):** ~15-25k transitions/min

### Memory Usage
- Model: ~100-500 MB (depends on config)
- Replay buffer: 50-90% of total allocation
- Data cache: 10-30% of total allocation

## Next Steps

1. **Run training:** `python train.py --auto --epochs 3`
2. **Monitor:** `watch -n 1 'nvidia-smi'`
3. **Check results:** `ls checkpoints/`
4. **Analyze:** Load model and evaluate on test data
5. **Scale up:** Try larger config for better learning

---

**Status: 🚀 Ready to Train on GPU!**

One command to rule them all:
```bash
git clone https://github.com/Baswold/weather-ai.git && cd weather-ai && bash runpod_setup.sh
```
