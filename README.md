# NFL Predictor

A production-style NFL forecasting engine that predicts game winners, scores, and derived quantities (spread, total) with calibrated probabilities.

## Vision

Build a serious, incremental NFL forecasting system that:

- **Predicts multiple targets:**
  - Winner (with calibrated win probabilities)
  - Score distribution (expected points, uncertainty intervals)
  - Derived quantities (spread, total, win margin distribution)

- **Designed for continuous improvement:**
  - Start with solid baselines (V0)
  - Evolve into advanced models (V1-V3: GBDT → neural nets → ensembles)
  - Robust data pipelines and feature stores

- **Teaches best practices:**
  - Structured ML experiments (tracking, configs, reproducibility)
  - Production-quality code (not "toy" classifiers)
  - Honest evaluation (time-based backtesting, calibration)

## Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd nfl-predictor

# Install package
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"
```

### Project Structure

```
nfl-predictor/
├── data/           # Data pipeline stages (raw → interim → processed)
├── models/         # Trained model artifacts
├── notebooks/      # Jupyter notebooks for exploration
├── src/            # Core package (nfl_predictor)
├── scripts/        # CLI entrypoints
├── configs/        # Hydra configuration files
├── tests/          # Unit tests
└── docs/           # Documentation
```

## Documentation

- **[CLAUDE.md](CLAUDE.md)** - AI agent operating rules and guidelines
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Full system architecture
- **[docs/SCHEMAS.md](docs/SCHEMAS.md)** - Data table schemas

## Development Roadmap

### V0: Foundation & Simple Baselines *(Current)*

**Goal:** Establish reference performance and data infrastructure.

- [ ] Ingest games data (historical results)
- [ ] Build `games_base` table (interim data)
- [ ] Implement simple rolling stats (last 3/5 games)
- [ ] Build Elo rating system
- [ ] Baseline models:
  - Home team always wins
  - Higher Elo wins
  - Logistic regression (minimal features)
- [ ] Time-based evaluation framework

**Deliverables:**
- Clean data pipeline (raw → interim → processed)
- Baseline model accuracy benchmarks
- Evaluation infrastructure

---

### V1: Rich Features + GBDT

**Goal:** Production-quality gradient boosting models.

- [ ] Implement all feature groups:
  - Team rolling stats (offensive/defensive efficiency)
  - Head-to-head history
  - Rest & travel distance
  - Venue features (dome, surface, altitude)
  - Weather conditions
  - Betting lines (spread, total)
- [ ] Train XGBoost/LightGBM models (winner + score)
- [ ] Hyperparameter tuning
- [ ] Calibration (Platt scaling)
- [ ] Feature importance analysis

**Deliverables:**
- Feature registry with 50-100 features
- Tuned GBDT models
- Calibration curves and backtest reports

---

### V2: Neural Networks + Sequence Models

**Goal:** Leverage deep learning for team state representations.

- [ ] Build team/stadium embeddings
- [ ] Implement MLP with multi-task heads (winner + margin + total)
- [ ] Sequence models (RNN/Transformer) over recent games
- [ ] Combined training (classification + regression loss)
- [ ] Experiment tracking (W&B)

**Deliverables:**
- Neural network models
- Team form embeddings
- Performance comparison to V1

---

### V3: Ensembles + Uncertainty

**Goal:** Maximize performance and quantify uncertainty.

- [ ] Ensemble GBDT + neural models (stacking/blending)
- [ ] Quantile regression for score distributions
- [ ] Monte Carlo dropout for uncertainty
- [ ] Season simulation (playoff odds, division winners)

**Deliverables:**
- Production ensemble pipeline
- Calibrated prediction intervals
- Season simulation tools

## Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10+ |
| **Data** | pandas, numpy, pyarrow |
| **ML** | scikit-learn, xgboost, pytorch |
| **Experiment Tracking** | wandb |
| **Config Management** | hydra |
| **Testing** | pytest |

## Usage

### Data Ingestion

```bash
# Download and process raw data
python scripts/build_all_data.py
```

### Training

```bash
# Train baseline model
python scripts/train_model.py training=v0

# Train GBDT model with all features
python scripts/train_model.py training=v1 model=gbdt
```

### Evaluation

```bash
# Run backtest on test seasons
python scripts/evaluate_model.py model_path=models/winner/v1_gbdt.pkl
```

### Prediction

```bash
# Predict single game
python scripts/predict_games.py \
  --home-team BUF \
  --away-team KC \
  --game-date 2024-01-21 \
  --model winner_v1_gbdt

# Predict full week
python scripts/predict_games.py \
  --season 2024 \
  --week 3 \
  --schedule data/raw/schedules_2024_week3.csv
```

## Design Principles

1. **Incremental sophistication** - Start simple, evolve to complex
2. **Separation of concerns** - Data → features → models → evaluation → serving
3. **Reproducible experiments** - Versioned datasets, config-driven training, experiment tracking
4. **Composable features** - Modular feature groups, easy to add/remove
5. **Honest evaluation** - Time-based backtesting, strong baselines, calibration checks
6. **No leakage** - Features only use pre-game data, enforced by validation

## Core Philosophy

> "There is no artificial deadline. The roadmap is incremental and extensible."

This project prioritizes:
- **Quality over speed** - Robust pipelines, not quick hacks
- **Learning over results** - Understanding what works and why
- **Extensibility** - Easy to add new features, models, data sources

## Contributing

See [CLAUDE.md](CLAUDE.md) for development guidelines and AI agent operating rules.

## License

MIT
