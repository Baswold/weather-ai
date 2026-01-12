# TODO: Roadmap to Full Concept Implementation

This file tracks improvements needed to align the codebase with the full research vision in [CONCEPT.md](CONCEPT.md).

## Priority 1: Core Functionality (Before First Training Run)

- [ ] **Verify data pipeline works end-to-end**
  - [ ] Test data download for 1 location, 1 year
  - [ ] Verify [reanalysis, actual] pairs are correctly formatted
  - [ ] Test sequential iteration through dates

- [ ] **Test model architecture**
  - [ ] Verify transformer forward pass with real data
  - [ ] Check parameter count matches config expectations
  - [ ] Test gradient flow through model

- [ ] **Implement missing dependencies**
  - [ ] Add requirements.txt with all dependencies
  - [ ] Test on clean environment
  - [ ] Add installation instructions

- [ ] **Baseline training run**
  - [ ] Train `low_memory` config (3 locations, 1 year)
  - [ ] Verify training loop completes
  - [ ] Log metrics: reward, loss, prediction error
  - [ ] Save checkpoint successfully

## Priority 2: Research Infrastructure

- [ ] **Evaluation and metrics**
  - [ ] Implement comparison to naive baselines:
    - [ ] Persistence model (tomorrow = today)
    - [ ] Climatology (tomorrow = historical average for this date)
    - [ ] Repeat forecast (use reanalysis as prediction)
  - [ ] Per-variable error metrics (temp, precip, wind separately)
  - [ ] Extreme event detection and scoring
  - [ ] Skill score calculations

- [ ] **Visualization and analysis**
  - [ ] Plot reward over training time
  - [ ] Prediction error by season
  - [ ] Prediction error by location/climate type
  - [ ] Learning curve (does performance improve over decades?)
  - [ ] Attention visualization (which historical days matter most?)

- [ ] **Experiment tracking**
  - [ ] Integration with Weights & Biases or TensorBoard
  - [ ] Log hyperparameters for each run
  - [ ] Track model checkpoints with metadata
  - [ ] Reproducibility: save random seeds, data versions

## Priority 3: Scaling Up

- [ ] **Production config testing**
  - [ ] Train on 15 years (2010-2024) with 20 locations
  - [ ] Monitor memory usage throughout training
  - [ ] Estimate time for full historical run

- [ ] **Historical config (75 years)**
  - [ ] Download and cache full 1950-2024 dataset
  - [ ] Test memory requirements with actual data
  - [ ] Implement checkpointing strategy for multi-day training
  - [ ] Resume from checkpoint functionality

- [ ] **Distributed training**
  - [ ] Multi-GPU support
  - [ ] Data parallel across locations
  - [ ] Checkpoint synchronization

## Priority 4: Enhanced Data Sources

### Real Historical Forecasts (Beyond Reanalysis)

- [ ] **Research data availability**
  - [ ] NOAA NDFD archives: what years available?
  - [ ] ECMWF forecast archives: access requirements?
  - [ ] Individual weather station forecasts
  - [ ] Data format and API documentation

- [ ] **Implement historical forecast loader**
  - [ ] New data source: NOAAHistoricalForecasts class
  - [ ] Parse forecast format (may differ from reanalysis)
  - [ ] Handle missing data and quality flags
  - [ ] Merge with existing reanalysis pipeline

- [ ] **Hybrid approach**
  - [ ] Use ERA5 for 1950-1990 (reanalysis)
  - [ ] Use real forecasts for 1990-2024 (if available)
  - [ ] Document the transition point in data

### Additional Weather Variables

- [ ] **Expand beyond basic variables**
  - [ ] Dewpoint temperature
  - [ ] Cloud cover / sky condition
  - [ ] Visibility
  - [ ] Weather type (rain/snow/clear)

- [ ] **Derived variables**
  - [ ] Heat index
  - [ ] Wind chill
  - [ ] Precipitation probability (if available in forecasts)

## Priority 5: Advanced RL Techniques

- [ ] **Prioritized experience replay**
  - [ ] Implement TD-error based prioritization
  - [ ] Boost replay probability for extreme events
  - [ ] Test impact on rare event prediction

- [ ] **Reward engineering**
  - [ ] Experiment with different error metrics (MSE, MAE, Huber)
  - [ ] Variable-specific weights (optimize for what matters most)
  - [ ] Asymmetric penalties (under-predicting storms worse than over?)
  - [ ] Skill-score based rewards

- [ ] **Algorithm improvements**
  - [ ] Try SAC (Soft Actor-Critic) for better exploration
  - [ ] Implement PPO for on-policy comparison
  - [ ] Distributional RL for uncertainty estimation

- [ ] **Multi-step prediction**
  - [ ] Extend to predict 3 days ahead
  - [ ] Compound rewards for multi-day accuracy
  - [ ] Curriculum learning: 1-day → 3-day → 7-day

## Priority 6: Spatial Modeling

- [ ] **Graph neural networks**
  - [ ] Define spatial graph (locations as nodes, edges by distance)
  - [ ] Implement GNN layers for message passing
  - [ ] Test if spatial context improves predictions

- [ ] **Location embeddings**
  - [ ] Learn representations for each location
  - [ ] Test transfer: train on some locations, predict on others
  - [ ] Cluster by climate type

- [ ] **Regional models**
  - [ ] Train separate models for different climate zones
  - [ ] Compare to single global model

## Priority 7: Continual Learning Deployment

- [ ] **Production pipeline**
  - [ ] Fetch today's reanalysis/forecast automatically
  - [ ] Make predictions for tomorrow
  - [ ] Store predictions in database
  - [ ] Wait for actual weather, compute reward
  - [ ] Update model daily

- [ ] **Monitoring and alerts**
  - [ ] Track prediction error over time
  - [ ] Alert if error increases (distribution shift)
  - [ ] Periodic evaluation on held-out test set

- [ ] **Efficient adaptation**
  - [ ] LoRA-style fine-tuning for daily updates
  - [ ] Keep most parameters frozen, adapt small subset
  - [ ] Test catastrophic forgetting resistance

## Priority 8: Probabilistic Forecasting

- [ ] **Uncertainty quantification**
  - [ ] Output mean + variance for each variable
  - [ ] Ensemble predictions (multiple forward passes with dropout)
  - [ ] Calibration: are 80% confidence intervals accurate 80% of the time?

- [ ] **Distributional RL**
  - [ ] Predict full distribution, not just point estimate
  - [ ] Use quantile regression or distributional heads
  - [ ] Evaluate with proper scoring rules (CRPS)

## Priority 9: Physics-Informed Learning

- [ ] **Soft constraints**
  - [ ] Energy balance (incoming solar = outgoing thermal)
  - [ ] Temperature-dewpoint relationship (dewpoint ≤ temperature)
  - [ ] Conservation laws

- [ ] **Auxiliary losses**
  - [ ] Add physics-based regularization terms
  - [ ] Penalize physically impossible predictions
  - [ ] Test if constraints improve sample efficiency

## Priority 10: Documentation and Paper

- [ ] **Code documentation**
  - [ ] Docstrings for all classes and methods
  - [ ] Type hints throughout
  - [ ] Usage examples in docstrings

- [ ] **Research paper**
  - [ ] Introduction and motivation
  - [ ] Related work comparison
  - [ ] Method description
  - [ ] Experimental results and analysis
  - [ ] Discussion: why RL? what did we learn?

- [ ] **Reproducibility**
  - [ ] Complete installation guide
  - [ ] Step-by-step tutorial for reproducing results
  - [ ] Share pre-trained checkpoints
  - [ ] Dataset documentation

## Ideas for Future Exploration

- [ ] **Application to other forecasting domains**
  - [ ] Stock market: analyst forecasts → actual prices
  - [ ] Energy: demand forecasts → actual load
  - [ ] Traffic: ETA predictions → actual travel time

- [ ] **Meta-learning**
  - [ ] Can the model learn to "learn how to forecast"?
  - [ ] Transfer to new locations with few-shot adaptation
  - [ ] Generalize across climate types

- [ ] **Forecast combination**
  - [ ] Combine multiple forecast sources (ECMWF, GFS, HRRR)
  - [ ] Learn optimal weights for ensemble
  - [ ] Dynamic selection based on recent performance

## Research Questions to Investigate

1. **Learning dynamics**: What weather patterns emerge first? Daily cycles, then seasonal, then decadal?
2. **Transfer learning**: Does learning in tropical regions help in temperate regions?
3. **Rare events**: Can experience replay enable better extreme event prediction?
4. **Distribution shift**: How well does the model adapt to climate change signals?
5. **RL vs. supervised**: Is sequential temporal RL competitive with standard supervised learning?
6. **Data efficiency**: Does the model learn faster from forecast-actual pairs than from raw observations?

---

**Legend**:
- [ ] Not started
- [x] Completed
- [🚧] In progress
- [❓] Needs research/decision
