"""Dataset builders for NFL prediction models."""

from typing import Tuple

import pandas as pd


def build_winner_dataset_v0(
    features: pd.DataFrame,
    train_seasons: list[int],
    test_seasons: list[int],
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Build train/test split for winner prediction.

    Creates time-based train/test split respecting temporal ordering.
    Drops rows with NaN features (e.g., Week 1 games with no rolling stats).
    Removes leakage columns (scores, derived targets, identifiers).

    Args:
        features: Feature matrix from games_features_v0.parquet
        train_seasons: Seasons to include in training (e.g., range(2000, 2019))
        test_seasons: Seasons for testing (e.g., range(2019, 2024))

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)

    Example:
        >>> features = pd.read_parquet("data/processed/games_features_v0.parquet")
        >>> X_train, y_train, X_test, y_test = build_winner_dataset_v0(
        ...     features,
        ...     train_seasons=list(range(2000, 2019)),
        ...     test_seasons=list(range(2019, 2024))
        ... )
    """
    # Filter by season
    train = features[features["season"].isin(train_seasons)].copy()
    test = features[features["season"].isin(test_seasons)].copy()

    # Drop rows with NaN in features (Week 1 games have no rolling stats)
    train = train.dropna()
    test = test.dropna()

    # Target column
    target_col = "home_win"

    # Columns to drop (not features)
    drop_cols = [
        # Identifiers
        "game_id",
        "game_date",
        "season",
        "week",
        # Team IDs
        "home_team",
        "away_team",
        # Score information (would cause leakage)
        "home_points",
        "away_points",
        # Score derivatives (would cause leakage)
        "margin",
        "total_points",
    ]

    # Feature columns (everything else except target)
    feature_cols = [c for c in features.columns if c not in drop_cols + [target_col]]

    X_train = train[feature_cols]
    y_train = train[target_col]
    X_test = test[feature_cols]
    y_test = test[target_col]

    return X_train, y_train, X_test, y_test
