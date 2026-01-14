# Cross-Modal Progressive Fine-Tuning (CMPF) for Wildfire Spread Forecasting

This repository contains code to reproduce the experiments in the paper "Adapting Video Foundation Models for Spatiotemporal Wildfire Forecasting via Cross-Modal Progressive Fine-Tuning"

## Repository Structure

```
.
├── configs/                  # configs for models, datasets, training
├── data/
│   ├── WildfireSpreadTS/     # Preprocessed WTS data
│   └── NextDayWildfireSpread/# Preprocessed NDWS data
├── models/                   # Model backbones + heads
    ├── backbones.py
    ├── base.py
    ├── cmpf_model.py
    └── segmentation_head.py
├── datasets/                 # PyTorch Dataset/Dataloader wrappers
├── trainers/                 # Generic training loops for baseline & CMPF
└── utils/                    # Metrics, logging, visualization, etc.
```

## Data

### WildfireSpreadTS (Target Task Dataset)

* Multi-modal, multi-temporal dataset designed for 24‑hour‑ahead wildfire spread prediction.
* 607 fire events (2018–2021), 13,607 daily image sets.
* Each daily sample:
  * Input tensor `X_WTS ∈ R^{H×W×M_WTS}`, with M_WTS = 23 channels (fuel, topography, weather & forecasts, vegetation).
  * Target fire mask `S_WTS ∈ {0,1}^{H×W}` at 375 m resolution.
* Official split used in the paper:
  * **Train:** 2018–2020
  * **Validation/Test:** 2021

### NextDayWildfireSpread (Auxiliary Dataset)

* Large-scale wildfire dataset used for intermediate fine-tuning.
* 18,545 events across the contiguous U.S., years 2012–2020.
* Each event is a single-day pair:
  * Input `X_NDWS,t ∈ R^{H×W×M_NDWS}`, M_NDWS = 12 variables.
  * Next day target `S_NDWS,t+1 ∈ {0,1}^{H×W}` at 1 km resolution. 
