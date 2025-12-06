"""Data ingestion from nflverse (nflreadpy)."""

import pandas as pd
from datetime import datetime
from typing import Optional

from nfl_predictor.utils.validation import (
    validate_schema,
    validate_no_nulls,
    validate_date_range,
)


# Game type mapping from nflverse to our schema
GAME_TYPE_MAPPING = {
    "REG": "REG",
    "WC": "WC",
    "DIV": "DIV",
    "CON": "CONF",  # nflverse uses CON, we use CONF
    "SB": "SB",
}


def normalize_nflverse_schedule(df_raw) -> pd.DataFrame:
    """Normalize nflverse schedule data to GameBase schema.

    Args:
        df_raw: Raw schedule DataFrame from nflreadpy.load_schedules()
                (can be Polars or pandas DataFrame)

    Returns:
        DataFrame conforming to GameBase schema

    Raises:
        ValueError: If data validation fails
    """
    # Convert Polars to pandas if needed
    if hasattr(df_raw, "to_pandas"):
        df_raw = df_raw.to_pandas()

    # Filter out future games (no scores yet)
    df_completed = df_raw[
        df_raw["home_score"].notnull() & df_raw["away_score"].notnull()
    ].copy()

    # Parse game date from gameday and gametime
    # gametime format: "HH:MM" or similar
    df_completed["game_date"] = pd.to_datetime(
        df_completed["gameday"] + " " + df_completed["gametime"].fillna("00:00"),
        errors="coerce"
    )

    # Map game_type
    df_completed["game_type_normalized"] = df_completed["game_type"].map(GAME_TYPE_MAPPING)

    # Handle neutral site - if location is "Neutral", set neutral_site = True
    df_completed["neutral_site"] = df_completed.get("location", "").str.lower() == "neutral"

    # Create normalized DataFrame with GameBase columns
    df_normalized = pd.DataFrame({
        "game_id": df_completed["game_id"],
        "season": df_completed["season"].astype(int),
        "week": df_completed["week"].astype(int),
        "game_date": df_completed["game_date"],
        "game_type": df_completed["game_type_normalized"],
        "home_team": df_completed["home_team"],
        "away_team": df_completed["away_team"],
        "home_points": df_completed["home_score"].astype(int),
        "away_points": df_completed["away_score"].astype(int),
        "stadium_id": df_completed.get("stadium_id", None),
        "kickoff_time_local": None,  # Not typically in nflverse schedule
        "neutral_site": df_completed["neutral_site"],
    })

    # Validate schema
    required_columns = [
        "game_id", "season", "week", "game_date", "game_type",
        "home_team", "away_team", "home_points", "away_points"
    ]
    validate_schema(df_normalized, required_columns, "games_base")

    # Validate no nulls in critical fields
    validate_no_nulls(df_normalized, required_columns, "games_base")

    # Validate date range (NFL founded in 1920, max is current date)
    validate_date_range(
        df_normalized,
        "game_date",
        pd.Timestamp("1920-01-01"),
        pd.Timestamp.now(),
        "games_base"
    )

    # Validate game_type values
    valid_game_types = {"REG", "WC", "DIV", "CONF", "SB"}
    invalid_types = set(df_normalized["game_type"].unique()) - valid_game_types
    if invalid_types:
        raise ValueError(f"Invalid game_type values found: {invalid_types}")

    return df_normalized


def print_summary_stats(df: pd.DataFrame) -> None:
    """Print summary statistics for sanity checking.

    Args:
        df: Normalized games_base DataFrame
    """
    print("\n" + "="*60)
    print("GAMES_BASE SUMMARY")
    print("="*60)
    print(f"\nTotal games: {len(df):,}")
    print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    print(f"Seasons: {df['season'].min()} to {df['season'].max()}")

    print("\nGames by type:")
    print(df["game_type"].value_counts().sort_index())

    print("\nGames by season (last 5):")
    print(df["season"].value_counts().sort_index().tail())

    print("\nTeams (unique):")
    all_teams = pd.concat([df["home_team"], df["away_team"]]).unique()
    print(f"  {len(all_teams)} teams: {sorted(all_teams)}")

    print("\nNull checks:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    print("\nSample records:")
    print(df.head(3).to_string())
    print("="*60 + "\n")
