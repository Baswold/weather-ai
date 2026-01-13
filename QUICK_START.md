# 🚀 Quick Start - 3 Options

## Option 1: RunPod GPU (Fastest & Recommended)

### One Command
```bash
git clone https://github.com/Baswold/weather-ai.git && cd weather-ai && bash runpod_setup.sh
```

**What it does:**
1. Clones repository
2. Installs GPU-optimized dependencies
3. Auto-detects your GPU
4. Auto-configures for your hardware
5. Optionally starts training

**Expected output:**
```
GPU Detected: NVIDIA A100 (40GB)
✓ System packages updated
✓ Repository cloned
✓ Dependencies installed
✓ Pre-flight checks passed
✓ GPU configured

Ready to train!
Start training with: python train.py --auto --target-gb 32 --epochs 3 --chunked
```

**Time:** ~5-10 minutes to setup, then training starts immediately

**Cost:** $0.30-10 depending on GPU choice

---

## Option 2: Local Machine (Your Computer)

### 3 Steps

**Step 1: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 2: Verify setup**
```bash
python tests/verify_setup.py
```

**Step 3: Run training**
```bash
python train.py
```

The script will ask: "How much RAM can you dedicate to training?"

**Time:** 5 mins setup + training time (5 mins to 12+ hours depending on config)

**Configs:**
- `--config low_memory` → 500 MB, 10 mins
- `--config default` → 4 GB, 30 mins
- `--config production` → 8 GB, 2+ hours

---

## Option 3: Cloud (Colab, AWS, etc.)

### Setup
```bash
# Clone repo
!git clone https://github.com/Baswold/weather-ai.git
%cd weather-ai

# Install dependencies
!pip install -r requirements.txt

# Run tests
!python tests/verify_setup.py

# Start training
!python train.py --config production --epochs 3
```

---

## Recommendation by Hardware

### You have a GPU (RTX, A100, etc.)
👉 **Use Option 1 (RunPod)**
- One command setup
- GPU automatically detected
- Auto-optimized configuration
- Best performance

### You have a powerful laptop/desktop (16GB+ RAM)
👉 **Use Option 2 (Local)**
- No rental cost
- Full control
- Can run continuously

### You have basic laptop (8GB RAM)
👉 **Use Option 1 (RunPod)**
- Rent a GPU pod instead
- More cost-effective than waiting
- Or use `--config low_memory` locally

### You have a cloud account (AWS, GCP, etc.)
👉 **Use Option 3 (Cloud)**
- Integrate with existing setup
- Or adapt runpod_setup.sh for your platform

---

## What Gets Trained

The RL Weather model learns to predict tomorrow's weather by:

1. **Observing:** Historical weather data (1-75 years)
2. **Predicting:** Next day's temperature, precipitation, wind
3. **Learning:** Improve predictions through reinforcement learning
4. **Storing:** Successful strategies in experience replay buffer

### Model Output
- Next day's temperature (min, max, mean)
- Next day's precipitation
- Next day's wind speed (max, mean)

### Training Data
- Open-Meteo API (free, no key required)
- 10-20 diverse locations (major cities worldwide)
- 1-75 years of historical data

### Training Time
- 5 mins (quick test) to 12+ hours (full 75-year history)

---

## After Training

### 1. Check Results
```bash
ls checkpoints/
cat checkpoints/config.json
```

### 2. Download Model
- Use RunPod download UI
- Or upload to cloud
- Or keep pod running for inference

### 3. Load & Evaluate
```python
import torch
model = torch.load('checkpoints/model_final.pt')
# Use for predictions on new data
```

---

## File Structure

```
weather-ai/
├── train.py                 # Main training script
├── verify_setup.py         # Pre-flight check (moved to tests/)
├── runpod_setup.sh         # RunPod one-command setup
├── tests/                  # All test files
│   ├── verify_setup.py
│   ├── test_api_consistency.py
│   ├── test_api_details.py
│   ├── test_schema_comparison.py
│   └── README.md
├── configs/
│   └── default.py          # Training configs
├── src/
│   ├── data/               # Data loading
│   ├── models/             # Model architecture
│   ├── rl/                 # RL training
│   └── utils.py
├── SETUP.md                # Complete setup guide
├── QUICK_START.md          # This file
├── RUNPOD_QUICK_START.md   # RunPod guide
├── RUNPOD_DEPLOY.md        # Detailed RunPod docs
└── requirements.txt        # Dependencies
```

---

## Troubleshooting

### "Module not found"
```bash
pip install -r requirements.txt
```

### "Setup fails"
```bash
python tests/verify_setup.py  # Run detailed checks
```

### "Out of memory"
```bash
python train.py --config low_memory  # Use smaller config
```

### "GPU not detected"
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Falls back to CPU if false
```

### "Data download fails"
- Check internet
- Wait a minute (API rate limit)
- Cached data will be used on retry

---

## Performance Expectations

### RunPod with A100 (40GB)
- **Setup:** ~5 minutes
- **Training:** 6-12 hours
- **Speed:** 100k+ transitions/minute
- **Cost:** ~$5-10

### RunPod with RTX 4090 (24GB)
- **Setup:** ~5 minutes
- **Training:** 12-24 hours
- **Speed:** 50k+ transitions/minute
- **Cost:** ~$3-5

### Local Machine (8GB RAM)
- **Setup:** ~5 minutes
- **Training:** 10 minutes (low_memory config)
- **Speed:** ~1k transitions/minute
- **Cost:** Free (electricity only)

---

## Next Steps

### Choose Your Path:

**🚀 Fast GPU Training (RunPod)**
```bash
git clone https://github.com/Baswold/weather-ai.git && cd weather-ai && bash runpod_setup.sh
```

**💻 Local Training**
```bash
git clone https://github.com/Baswold/weather-ai.git
cd weather-ai
pip install -r requirements.txt
python tests/verify_setup.py
python train.py
```

**☁️ Cloud Training**
- Adapt runpod_setup.sh for your platform
- Or follow local setup + scale up

---

## Documentation

| Document | For |
|----------|-----|
| `QUICK_START.md` | Quick overview (this file) |
| `SETUP.md` | Complete setup guide |
| `BEFORE_RUNNING.md` | Pre-training checklist |
| `READY_TO_TRAIN.md` | Detailed summary |
| `RUNPOD_QUICK_START.md` | RunPod overview |
| `RUNPOD_DEPLOY.md` | RunPod detailed guide |
| `CONCEPT.md` | Research concept |
| `TODO.md` | Future enhancements |
| `tests/README.md` | Test descriptions |

---

## Status: 🟢 Ready to Go!

Everything is tested and verified. Pick an option above and start training!

```bash
# Fastest: One-command RunPod setup
git clone https://github.com/Baswold/weather-ai.git && cd weather-ai && bash runpod_setup.sh

# Or local training
git clone https://github.com/Baswold/weather-ai.git && cd weather-ai && python tests/verify_setup.py
```

---

**Questions? Check:**
- `SETUP.md` - Complete guide
- `tests/README.md` - Test documentation
- Issues on GitHub

**Ready to train? 🚀**
