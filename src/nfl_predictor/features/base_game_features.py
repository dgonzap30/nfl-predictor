"""Base game features - targets and calendar features."""

from typing import Any, Dict

import pandas as pd


class BaseGameFeatures:
    """Feature group for base game-level features.

    Computes targets and simple calendar features from raw game data.
    No historical lookback - all features derived from current game row only.
    """

    name = "base_game_v0"

    def build_features(
        self,
        games_base: pd.DataFrame,
        aux_tables: Dict[str, pd.DataFrame],
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """Build base game features.

        Args:
            games_base: DataFrame with base game information
            aux_tables: Dictionary of auxiliary tables (not used for base features)
            config: Configuration dictionary (not used for base features)

        Returns:
            DataFrame with game_id and computed features

        Raises:
            ValueError: If required columns are missing
        """
        # Verify required columns
        required_cols = [
            "game_id",
            "season",
            "week",
            "game_date",
            "game_type",
            "home_team",
            "away_team",
            "home_points",
            "away_points",
            "neutral_site",
        ]
        missing = set(required_cols) - set(games_base.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Start with a copy to avoid modifying input
        df = games_base.copy()

        # Compute targets
        df["home_win"] = (df["home_points"] > df["away_points"]).astype(int)
        df["margin"] = df["home_points"] - df["away_points"]
        df["total_points"] = df["home_points"] + df["away_points"]

        # Calendar features
        df["month"] = pd.to_datetime(df["game_date"]).dt.month
        df["is_playoff"] = (df["game_type"] != "REG")

        # Neutral site flag
        df["is_neutral_site"] = df["neutral_site"]

        # Select output columns
        # Note: game_id, game_date, season, week are provided by build_all_features
        # We only need to return game_id (for join) + computed features
        output_cols = [
            # Key (required for merge)
            "game_id",
            # Context (pass-through)
            "home_team",
            "away_team",
            "home_points",
            "away_points",
            # Targets
            "home_win",
            "margin",
            "total_points",
            # Calendar features
            "month",
            "is_playoff",
            # Flags
            "is_neutral_site",
        ]

        return df[output_cols]
