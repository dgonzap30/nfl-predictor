#!/usr/bin/env python3
"""Train V0 winner classifier models.

Trains LogisticRegression and GBDT (HistGradientBoostingClassifier) on
engineered features to beat 62.8% Elo baseline.

Target performance:
- Accuracy: 65-70%
- Log Loss: ~0.60 or lower
- Brier Score: < 0.22
"""

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from nfl_predictor.modeling.datasets import build_winner_dataset_v0
from nfl_predictor.modeling.metrics import (
    winner_accuracy,
    winner_brier_score,
    winner_log_loss,
)


def main():
    parser = argparse.ArgumentParser(description="Train V0 winner classifier models")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/games_features_v0.parquet"),
        help="Path to feature matrix",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/winner/v0_gbdt"),
        help="Directory to save model artifacts",
    )
    parser.add_argument(
        "--experiments",
        type=Path,
        default=Path("experiments/v0_models.json"),
        help="Path to save experiment results",
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

    print("=== V0 ML Models ===")
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

    # Initialize models
    models = {
        "logreg": LogisticRegression(max_iter=1000, random_state=42),
        "gbdt": HistGradientBoostingClassifier(
            max_iter=100,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
        ),
    }

    # Train and evaluate models
    print("\nTraining models...")
    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)

        # Get predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Compute metrics
        acc = winner_accuracy(y_test.values, y_pred_proba)
        ll = winner_log_loss(y_test.values, y_pred_proba)
        bs = winner_brier_score(y_test.values, y_pred_proba)

        results[name] = {
            "accuracy": float(acc),
            "log_loss": float(ll),
            "brier_score": float(bs),
        }

        trained_models[name] = model

        print(f"    Accuracy: {acc:.3f}")
        print(f"    Log Loss: {ll:.3f}")
        print(f"    Brier:    {bs:.3f}")

    # Print comparison table
    print("\n" + "=" * 60)
    print(f"{'Model':<20} {'Accuracy':>10} {'Log Loss':>10} {'Brier':>10}")
    print("=" * 60)

    # Include Elo baseline for comparison
    print(f"{'Elo Baseline':<20} {0.628:>10.3f} {0.663:>10.3f} {0.234:>10.3f}")

    for name, metrics in results.items():
        display_name = name.replace("_", " ").title()
        print(
            f"{display_name:<20} "
            f"{metrics['accuracy']:>10.3f} "
            f"{metrics['log_loss']:>10.3f} "
            f"{metrics['brier_score']:>10.3f}"
        )
    print("=" * 60)

    # Find best model by accuracy
    best_name = max(results.keys(), key=lambda k: results[k]["accuracy"])
    best_model = trained_models[best_name]

    print(f"\nBest model: {best_name}")
    print(f"  Accuracy: {results[best_name]['accuracy']:.3f}")
    print(f"  Improvement over Elo: +{(results[best_name]['accuracy'] - 0.628) * 100:.1f} pts")

    # Save best model
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"\nSaved model to {model_path}")

    # Save metadata
    meta = {
        "model_type": best_name,
        "train_seasons": train_seasons,
        "test_seasons": test_seasons,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "num_features": len(X_train.columns),
        "feature_columns": list(X_train.columns),
        "metrics": results[best_name],
    }

    meta_path = args.output_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_path}")

    # Save experiment results
    output_data = {
        "train_seasons": train_seasons,
        "test_seasons": test_seasons,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "num_features": len(X_train.columns),
        "feature_columns": list(X_train.columns),
        "results": results,
        "best_model": best_name,
    }

    args.experiments.parent.mkdir(parents=True, exist_ok=True)
    with open(args.experiments, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved experiment results to {args.experiments}")


if __name__ == "__main__":
    main()
