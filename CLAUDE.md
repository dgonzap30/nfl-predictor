# NFL Predictor - AI Agent Context

## Project Overview

**NFL Predictor** is a production-style NFL forecasting engine that predicts:
- **Winner** (with calibrated probabilities)
- **Scores** (expected points, spread, total)
- **Distributions** (win margin, score uncertainty)

**Design Philosophy:**
- Incremental sophistication (V0 baselines → V3 ensembles)
- Separation of concerns (data → features → models → evaluation → serving)
- Reproducible experiments (configs, tracking, versioning)
- Composable features (modular feature groups)

## Tech Stack

- **Language:** Python 3.10+
- **Data:** pandas, numpy, pyarrow
- **ML:** scikit-learn, xgboost, pytorch
- **Experiment Tracking:** wandb
- **Config Management:** hydra
- **Testing:** pytest

## Documentation

**For detailed architecture and schemas, see:**
- `docs/ARCHITECTURE.md` - Full repository structure, module responsibilities, modeling roadmap
- `docs/SCHEMAS.md` - Data table schemas (games, stadiums, weather, betting, injuries)

## Agent Operating Rules

### 1. Respect the Architecture
- Add new code under appropriate modules
- Keep responsibilities narrow (no random feature logic in modeling modules)
- Use the feature registry for all feature engineering

### 2. Prevent Data Leakage
**CRITICAL:** Features must only use data from **before** the game date.

```python
# Always check for leakage
from nfl_predictor.utils.validation import assert_no_future_leakage
assert_no_future_leakage(features_df, game_date_col="game_date")
```

- Any feature builder must only use `games[games.game_date < current_game_date]`
- Add assertions where possible: `assert feature_max_date < game_date`

### 3. Prefer Configs Over Hardcoding
- Use `configs/*.yaml` for:
  - Model types & hyperparameters
  - Active feature groups
  - Train/val/test splits
- Never hardcode paths, thresholds, or hyperparameters

### 4. Keep Experiments Organized
For each new model or change:
- Add/update relevant config in `configs/`
- Log to W&B/MLflow (experiment name, metrics, artifacts)
- Update notebooks that summarize results

### 5. Iterate Incrementally
- Implement V0 → V1 → V2 → V3 in order
- Stabilize each version before moving forward
- Keep earlier pipelines intact when adding advanced features

### 6. Don't Break Public APIs
- Core prediction interfaces (`predict_single`, `predict_batch`) must maintain backward-compatible signatures
- Document breaking changes clearly

### 7. Commit Guidelines
- Use conventional commit style: `feat:`, `fix:`, `refactor:`, `docs:`
- Be explicit about schema changes and backwards compatibility

## Current Phase: V0 - Foundation & Simple Baselines

### Completed
- [x] Project scaffolding
- [x] Data schemas defined

### In Progress
- [ ] None

### Pending
- [ ] Ingest games data
- [ ] Build games_base table
- [ ] Simple rolling stats
- [ ] Elo rating system
- [ ] Logistic regression baseline
- [ ] Score regression baseline
- [ ] Time-based evaluation
