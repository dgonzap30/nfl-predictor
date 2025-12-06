# NFL Predictor - Architecture

## Repository Structure

```
nfl-predictor/
├── data/                      # Data pipeline stages
│   ├── raw/                   # Raw downloads (games, weather, injuries, etc.)
│   ├── interim/              # Cleaned/normalized tables
│   ├── processed/            # Feature matrices ready for modeling
│   └── artifacts/            # Derived tables, embeddings
│
├── models/                    # Trained model artifacts
│   ├── winner/               # Classification models (home win prediction)
│   ├── score/                # Regression models (margin, total)
│   └── ensembles/            # Stacking/blending models
│
├── notebooks/                # Jupyter notebooks for exploration & reports
│   ├── 01_eda_games.ipynb
│   ├── 02_feature_groups.ipynb
│   ├── 03_model_baselines.ipynb
│   ├── 04_model_advanced.ipynb
│   └── 05_evaluation_backtest.ipynb
│
├── src/nfl_predictor/        # Core package
│   ├── config/               # Hydra/YAML configs
│   ├── data/                 # Data ingestion & schemas
│   ├── features/             # Feature engineering modules
│   ├── modeling/             # Models, training, ensembling
│   ├── evaluation/           # Backtesting, reports, simulation
│   ├── serving/              # Prediction serving
│   └── utils/                # Utilities (I/O, validation, geo)
│
├── scripts/                  # CLI entrypoints
│   ├── build_all_data.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict_games.py
│
├── configs/                  # Root-level Hydra configs
│   └── config.yaml
│
├── experiments/              # Experiment logs, W&B links
├── tests/                    # Unit tests
└── docs/                     # Documentation
```

---

## Module Responsibilities

### `src/nfl_predictor/data/`

**Purpose:** Ingest raw data and define schemas.

**Key files:**
- `ingest_games.py` - Download/parse game results
- `ingest_weather.py` - Fetch weather data per game
- `ingest_stadiums.py` - Stadium metadata
- `ingest_betting_lines.py` - Vegas lines (spread, total)
- `ingest_injuries.py` - Player injury reports
- `schemas.py` - Dataclass/Pydantic schemas for all tables

**Output:** Normalized parquet files in `data/interim/`

---

### `src/nfl_predictor/features/`

**Purpose:** Transform interim data into modeling features.

**Architecture:**
- **Feature Registry** (`feature_registry.py`):
  - Central registry of all feature groups
  - Each group implements `FeatureGroup` protocol:
    ```python
    class FeatureGroup(Protocol):
        name: str
        def build_features(games_base, aux_tables, config) -> DataFrame
    ```
  - Master builder merges selected groups based on config

**Feature Groups:**
1. **Base Game Features** - Week, season, game_type, calendar flags
2. **Team Rolling Stats** - Last N games: points, EPA, win%, home/away splits
3. **Elo Rating** - Dynamic team ratings (offense/defense variants)
4. **Head-to-Head** - Historical matchup stats, rivalry indicators
5. **Divisional Context** - Same division/conference flags
6. **Rest & Travel** - Days since last game, travel distance
7. **Venue Features** - Dome, surface, altitude, true home advantage
8. **Weather Features** - Temperature, wind, precip (bucketed)
9. **Betting Features** - Spread, total, implied probabilities
10. **Injury Features** - Starters out, position-weighted importance
11. **Sequence Features** (advanced) - Team form embeddings from RNN/Transformer

**Output:** Feature matrices in `data/processed/`

---

### `src/nfl_predictor/modeling/`

**Purpose:** Training, evaluation, and ensembling.

**Key files:**
- `datasets.py` - Build train/val/test splits from processed features
- `metrics.py` - Evaluation metrics (accuracy, log loss, MAE, calibration)
- `baselines.py` - V0 baseline models (home team, Elo, Vegas)
- `gbdt_models.py` - XGBoost/LightGBM wrappers
- `nn_models.py` - PyTorch modules (MLP, RNN, Transformer)
- `training_loops.py` - Generic train/eval loops
- `ensembling.py` - Stacking, blending
- `calibration.py` - Platt scaling, isotonic regression

**Model Evolution:**
- **V0:** Simple baselines (home advantage, Elo, Vegas)
- **V1:** GBDT on handcrafted features
- **V2:** Neural networks with embeddings + multi-task heads
- **V3:** Ensembles + uncertainty quantification

---

### `src/nfl_predictor/evaluation/`

**Purpose:** Backtest models and generate reports.

**Key files:**
- `backtest.py` - Time-based train/test splits, rolling windows
- `reports.py` - Summary tables, calibration plots, stratified analysis
- `simulation.py` - Monte Carlo season simulations, playoff odds

**Metrics:**
- **Winner:** Accuracy, log loss, Brier score, calibration
- **Score:** MAE/RMSE on points, margin, total
- **Benchmarks:** Compare to Vegas, Elo, previous versions

**Stratification:**
- By season, week bucket, divisional/non-divisional, dome/outdoor, weather

---

### `src/nfl_predictor/serving/`

**Purpose:** Production prediction interface.

**Key files:**
- `predict_single.py` - Predict single game
- `predict_batch.py` - Predict full week schedule
- `precompute_context.py` - Precompute latest features before gameday

**API:**
```python
predict_single(
    home_team: str,
    away_team: str,
    game_date: datetime,
    stadium_id: str = None,
    model_config: str = "winner_v1_gbdt"
) -> PredictionResult
```

**Output:**
- Win probabilities
- Expected scores (home_points, away_points)
- Implied spread & total
- Optional: score distribution quantiles

---

## Modeling Roadmap

### V0: Foundation & Baselines
**Goal:** Establish reference performance.

**Models:**
- Home team always wins
- Higher Elo wins
- Vegas favorite (if available)
- Logistic regression on minimal features

**Features:**
- Basic game info (home/away, week, season)
- Simple rolling stats (last 3 games)
- Elo ratings

**Deliverables:**
- `data/interim/games_base.parquet`
- Baseline model accuracy benchmarks
- Time-based evaluation framework

---

### V1: Rich Features + GBDT
**Goal:** Build production-quality gradient boosting models.

**Features:**
- All 11 feature groups (except sequence embeddings)
- ~50-100 features total

**Models:**
- XGBoost/LightGBM for winner (classification)
- XGBoost/LightGBM for margin + total (regression)

**Enhancements:**
- Hyperparameter tuning (Optuna/W&B sweeps)
- Calibration (Platt scaling)
- Feature importance analysis

**Deliverables:**
- Feature importance reports
- Calibration curves
- Comparison to V0 baselines

---

### V2: Neural Networks + Sequence Models
**Goal:** Leverage deep learning for team state representations.

**Architecture:**
- Embeddings for categorical vars (teams, divisions, stadiums)
- MLP on concatenated embeddings + numerical features
- Optional: RNN/GRU/Transformer over recent game sequences

**Multi-task heads:**
- Classification: home_win probability
- Regression: margin, total_points

**Training:**
- Combined loss (BCE + MSE with weighting)
- Early stopping, learning rate schedules
- Experiment tracking (W&B)

**Deliverables:**
- Sequence feature extractors
- Multi-task neural models
- Comparison to GBDT (V1)

---

### V3: Ensembles + Uncertainty
**Goal:** Maximize performance through ensembling and probabilistic outputs.

**Ensembling:**
- Stacking: Meta-model on top of V1 + V2 predictions
- Blending: Weighted average of best models
- Include baseline signals (Elo, Vegas)

**Uncertainty:**
- Quantile regression for score distributions
- Monte Carlo dropout / ensembles
- Calibrated confidence intervals

**Simulation:**
- Sample from predicted distributions
- Full season simulations (playoff odds, division winners)

**Deliverables:**
- Production ensemble pipeline
- Uncertainty-aware predictions
- Season simulation tools

---

## Data Pipeline Flow

```
Raw Data
   ↓
[Ingestion] (src/nfl_predictor/data/ingest_*.py)
   ↓
Interim Tables (data/interim/*.parquet)
   ↓
[Feature Engineering] (src/nfl_predictor/features/)
   ↓
Processed Features (data/processed/*.parquet)
   ↓
[Training] (src/nfl_predictor/modeling/training_loops.py)
   ↓
Models (models/winner/*.pkl, models/score/*.pt)
   ↓
[Evaluation] (src/nfl_predictor/evaluation/backtest.py)
   ↓
Reports (experiments/*.csv, plots)
   ↓
[Serving] (src/nfl_predictor/serving/predict_*.py)
   ↓
Predictions
```

---

## Configuration System (Hydra)

**Structure:**
```
configs/
├── config.yaml           # Root config
├── data/
│   ├── games.yaml
│   └── weather.yaml
├── features/
│   ├── base.yaml
│   ├── rolling.yaml
│   ├── elo.yaml
│   └── all.yaml         # All feature groups
├── models/
│   ├── gbdt.yaml
│   ├── nn.yaml
│   └── ensemble.yaml
└── training/
    ├── v0.yaml
    ├── v1.yaml
    └── v2.yaml
```

**Example config:**
```yaml
# configs/training/v1.yaml
defaults:
  - /features/all
  - /models/gbdt

model:
  type: xgboost
  max_depth: 6
  learning_rate: 0.1
  n_estimators: 200

training:
  train_seasons: [2018, 2019, 2020, 2021]
  val_seasons: [2022]
  test_seasons: [2023, 2024]

wandb:
  enabled: true
  project: nfl-predictor
  run_name: v1_gbdt_winner
```

---

## Testing Strategy

1. **Unit Tests:** Individual feature builders, metric calculations
2. **Integration Tests:** Full pipeline (raw → features → predictions)
3. **Validation Tests:** Data leakage checks, schema compliance
4. **Backtests:** Historical performance on past seasons

**Key test patterns:**
```python
# Test no future leakage
def test_feature_no_leakage():
    features = build_rolling_stats(games)
    assert_no_future_leakage(features)

# Test feature determinism
def test_feature_determinism():
    f1 = build_features(games, config)
    f2 = build_features(games, config)
    assert_frame_equal(f1, f2)
```

---

## Experiment Tracking

**W&B Integration:**
- Log hyperparameters, metrics (loss curves, accuracy, MAE)
- Save model artifacts
- Track dataset versions (data/processed checksums)

**Local logging:**
- `experiments/` directory: CSV summaries, plots
- Git tags for model versions

**Reproducibility:**
- Random seeds in configs
- Environment snapshots (requirements.txt)
- Dataset versioning (DVC or git-lfs)
