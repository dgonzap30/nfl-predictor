"""Validation utilities to prevent data leakage."""

import pandas as pd


def assert_no_future_leakage(
    features_df: pd.DataFrame,
    game_date_col: str = "game_date",
    feature_date_col: str = "feature_max_date",
) -> None:
    """Assert that no features use future data.

    Args:
        features_df: DataFrame with features and metadata
        game_date_col: Column name for game date
        feature_date_col: Column name for max date used in features

    Raises:
        ValueError: If any features use data from after the game date
    """
    if feature_date_col not in features_df.columns:
        # If feature_max_date not tracked, cannot validate
        return

    # Check for violations
    violations = features_df[
        features_df[feature_date_col] >= features_df[game_date_col]
    ]

    if len(violations) > 0:
        raise ValueError(
            f"Data leakage detected: {len(violations)} rows have "
            f"features computed from future data. "
            f"Example game_ids: {violations['game_id'].head().tolist()}"
        )


def validate_schema(
    df: pd.DataFrame, required_columns: list[str], table_name: str = "table"
) -> None:
    """Validate that DataFrame has required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        table_name: Name of table for error messages

    Raises:
        ValueError: If any required columns are missing
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"{table_name} is missing required columns: {sorted(missing)}"
        )


def validate_no_nulls(
    df: pd.DataFrame, columns: list[str], table_name: str = "table"
) -> None:
    """Validate that specified columns have no null values.

    Args:
        df: DataFrame to validate
        columns: List of column names to check
        table_name: Name of table for error messages

    Raises:
        ValueError: If any columns have null values
    """
    null_counts = df[columns].isnull().sum()
    nulls = null_counts[null_counts > 0]

    if not nulls.empty:
        null_info = ", ".join([f"{col}: {count}" for col, count in nulls.items()])
        raise ValueError(f"{table_name} has null values in critical columns: {null_info}")


def validate_date_range(
    df: pd.DataFrame,
    date_col: str,
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
    table_name: str = "table",
) -> None:
    """Validate that dates are within expected range.

    Args:
        df: DataFrame to validate
        date_col: Column name with dates
        min_date: Minimum expected date
        max_date: Maximum expected date
        table_name: Name of table for error messages

    Raises:
        ValueError: If any dates are outside the expected range
    """
    dates = pd.to_datetime(df[date_col])
    out_of_range = (dates < min_date) | (dates > max_date)

    if out_of_range.any():
        raise ValueError(
            f"{table_name} has {out_of_range.sum()} dates outside "
            f"expected range [{min_date}, {max_date}]"
        )
