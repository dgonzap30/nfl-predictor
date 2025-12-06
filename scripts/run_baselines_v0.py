#!/usr/bin/env python3
"""Run V0 baselines and output performance metrics.

Evaluates:
- HomeTeamBaseline: Always predicts home team wins
- EloBaseline: Uses Elo ratings to predict winner

Metrics:
- Accuracy: Percentage of correct predictions
- Log Loss: Calibration metric for probabilistic predictions
- Brier Score: Alternative calibration metric

Train: 2000-2018
Test: 2019-2023
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from nfl_predictor.modeling.baselines import EloBaseline, HomeTeamBaseline
from nfl_predictor.modeling.datasets import build_winner_dataset_v0
from nfl_predictor.modeling.metrics import (
    winner_accuracy,
    winner_brier_score,
    winner_log_loss,
)


def main():
    parser = argparse.ArgumentParser(description="Run V0 baseline evaluation")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/games_features_v0.parquet"),
        help="Path to feature matrix",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/v0_baselines.json"),
        help="Path to output results JSON",
    )
    parser.add_argument(
        "--train-start",
        type=int,
        default=2000,
        help="First season for training",
    )
    parser.add_argument(
        "--train-end",
        type=int,
        default=2019,
        help="Last season for training (exclusive)",
    )
    parser.add_argument(
        "--test-start",
        type=int,
        default=2019,
        help="First season for testing",
    )
    parser.add_argument(
        "--test-end",
        type=int,
        default=2024,
        help="Last season for testing (exclusive)",
    )
    args = parser.parse_args()

    print("=== V0 Baselines ===")
    print(f"Loading features from {args.input}...")
    features = pd.read_parquet(args.input)
    print(f"  Loaded {len(features):,} games")

    # Build train/test split
    train_seasons = list(range(args.train_start, args.train_end))
    test_seasons = list(range(args.test_start, args.test_end))

    print(f"\nBuilding dataset...")
    print(f"  Train: {args.train_start}-{args.train_end - 1}")
    print(f"  Test: {args.test_start}-{args.test_end - 1}")

    X_train, y_train, X_test, y_test = build_winner_dataset_v0(
        features, train_seasons, test_seasons
    )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(X_train):,} games")
    print(f"  Test: {len(X_test):,} games")
    print(f"  Features: {len(X_train.columns)}")

    # Initialize baselines
    baselines = {
        "home_always": HomeTeamBaseline(),
        "elo": EloBaseline(),
    }

    # Run evaluation
    print("\nRunning baselines...")
    results = {}

    for name, model in baselines.items():
        # Get predictions
        y_pred_proba = model.predict_proba(X_test)
        # Extract home win probability (second column)
        y_pred_proba_home = y_pred_proba[:, 1]

        # Compute metrics
        acc = winner_accuracy(y_test.values, y_pred_proba_home)
        ll = winner_log_loss(y_test.values, y_pred_proba_home)
        bs = winner_brier_score(y_test.values, y_pred_proba_home)

        results[name] = {
            "accuracy": float(acc),
            "log_loss": float(ll),
            "brier_score": float(bs),
        }

        print(f"  {name}: acc={acc:.3f}, ll={ll:.3f}, brier={bs:.3f}")

    # Print results table
    print("\n" + "=" * 60)
    print(f"{'Model':<20} {'Accuracy':>10} {'Log Loss':>10} {'Brier':>10}")
    print("=" * 60)
    for name, metrics in results.items():
        display_name = name.replace("_", " ").title()
        print(
            f"{display_name:<20} "
            f"{metrics['accuracy']:>10.3f} "
            f"{metrics['log_loss']:>10.3f} "
            f"{metrics['brier_score']:>10.3f}"
        )
    print("=" * 60)

    # Save results
    output_data = {
        "train_seasons": train_seasons,
        "test_seasons": test_seasons,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "num_features": len(X_train.columns),
        "feature_columns": list(X_train.columns),
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
