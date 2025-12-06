#!/usr/bin/env python3
"""Build the V0 feature matrix.

Loads games_base.parquet and builds all V0 feature groups:
- base_game_v0: Targets and calendar features
- team_rolling_v0: Rolling team statistics
- elo_v0: Elo ratings

Output: data/processed/games_features_v0.parquet
"""

import argparse
from pathlib import Path

import pandas as pd

# Import feature builder (this will auto-register all feature groups)
from nfl_predictor.features.feature_registry import build_features_v0


def main():
    parser = argparse.ArgumentParser(description="Build V0 feature matrix")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/interim/games_base.parquet"),
        help="Path to games_base.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/games_features_v0.parquet"),
        help="Path to output feature matrix",
    )
    args = parser.parse_args()

    print(f"Loading games_base from {args.input}...")
    games_base = pd.read_parquet(args.input)
    print(f"  Loaded {len(games_base):,} games")

    print("\nBuilding V0 features...")
    print("  - base_game_v0: Targets and calendar features")
    print("  - team_rolling_v0: 3-game rolling statistics")
    print("  - elo_v0: Elo ratings")

    features = build_features_v0(games_base)

    print(f"\nFeature matrix shape: {features.shape}")
    print(f"  Rows: {features.shape[0]:,}")
    print(f"  Columns: {features.shape[1]}")

    print("\nColumns:")
    for col in features.columns:
        print(f"  - {col}")

    # Show sample statistics
    print("\nSample statistics:")
    print(f"  Date range: {features['game_date'].min()} to {features['game_date'].max()}")
    print(f"  Seasons: {features['season'].min()} to {features['season'].max()}")
    print(f"  Home win rate: {features['home_win'].mean():.3f}")
    print(f"  Avg total points: {features['total_points'].mean():.1f}")

    print(f"\nSaving to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output, index=False)
    print("Done!")


if __name__ == "__main__":
    main()
