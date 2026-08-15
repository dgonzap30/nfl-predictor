# NFL Predictor

> **Status:** Forecasting research in progress. The repository is a structured experimentation scaffold; it does not currently claim validated predictive performance or production readiness.

## Research question

Can progressively stronger baselines improve NFL winner, score, spread, and total forecasts while remaining well calibrated and reproducible?

## Evaluation contract

The project is organized around a few non-negotiable checks:

- Time-based backtests instead of random train/test leakage
- Simple reference baselines before complex models
- Probability calibration, not accuracy alone
- Features available before kickoff only
- Reproducible data, configuration, and experiment records
- Separate evaluation for winner, score, spread, and total targets

## Planned progression

1. **Baselines** — home-team, Elo, and logistic-regression references
2. **Structured features** — rolling team form, rest, venue, weather, and market context
3. **Gradient boosting** — calibrated tree-based models
4. **Sequence and ensemble research** — only after simpler models establish a credible benchmark

These are research stages, not completed-performance claims.

## Install

```bash
git clone https://github.com/dgonzap30/nfl-predictor.git
cd nfl-predictor
pip install -e ".[dev]"
pytest
```

## Project map

```text
data/       data pipeline stages
models/     local model artifacts
notebooks/  exploration
src/        Python package
scripts/    command-line entry points
configs/    experiment configuration
tests/      automated checks
docs/       architecture and schema notes
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Data schemas](docs/SCHEMAS.md)
- [Development guidance](CLAUDE.md)

## License

MIT
