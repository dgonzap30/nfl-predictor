"""Elo rating system - team strength ratings."""

from collections import defaultdict
from typing import Any, Dict

import pandas as pd


# Team relocations - map to canonical team ID to preserve history
TEAM_RELOCATIONS = {
    "OAK": "LV",  # Oakland Raiders → Las Vegas Raiders
    "SD": "LAC",  # San Diego Chargers → LA Chargers
    "STL": "LA",  # St. Louis Rams → LA Rams
}


def get_canonical_team(team: str) -> str:
    """Map team to canonical ID to preserve history across relocations.

    Args:
        team: Team abbreviation

    Returns:
        Canonical team ID
    """
    return TEAM_RELOCATIONS.get(team, team)


class EloRatings:
    """Feature group for Elo ratings.

    Computes Elo ratings for each team, updating after each game.
    Captures Elo BEFORE each game to prevent data leakage.
    """

    name = "elo_v0"

    def build_features(
        self,
        games_base: pd.DataFrame,
        aux_tables: Dict[str, pd.DataFrame],
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """Build Elo rating features.

        Args:
            games_base: DataFrame with base game information
            aux_tables: Dictionary of auxiliary tables (not used)
            config: Configuration dictionary with optional params:
                - K: K-factor for updates (default: 20)
                - HFA: Home field advantage in Elo points (default: 65)
                - initial_elo: Starting Elo rating (default: 1500)

        Returns:
            DataFrame with game_id and Elo features

        Raises:
            ValueError: If required columns are missing
        """
        # Verify required columns
        required_cols = [
            "game_id",
            "game_date",
            "home_team",
            "away_team",
            "home_points",
            "away_points",
        ]
        missing = set(required_cols) - set(games_base.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Get parameters from config
        K = config.get("K", 20)
        HFA = config.get("HFA", 65)
        initial_elo = config.get("initial_elo", 1500)

        # Compute Elo features
        elo_df = self._compute_elo_ratings(games_base, K, HFA, initial_elo)

        return elo_df

    def _compute_elo_ratings(
        self,
        games_base: pd.DataFrame,
        K: float,
        HFA: float,
        initial_elo: float,
    ) -> pd.DataFrame:
        """Compute Elo ratings by iterating through games chronologically.

        Args:
            games_base: Game-level DataFrame
            K: K-factor for Elo updates
            HFA: Home field advantage in Elo points
            initial_elo: Starting Elo rating for all teams

        Returns:
            DataFrame with game_id and Elo features
        """
        # Sort games chronologically
        games = games_base.sort_values("game_date").reset_index(drop=True)

        # Initialize team Elos (using canonical team IDs)
        team_elos = defaultdict(lambda: initial_elo)

        # Store results for each game
        results = []

        # Iterate through games in chronological order
        for idx, game in games.iterrows():
            # Get canonical team IDs
            home = get_canonical_team(game["home_team"])
            away = get_canonical_team(game["away_team"])

            # Capture Elo BEFORE the game (prevents data leakage)
            home_elo_before = team_elos[home]
            away_elo_before = team_elos[away]

            # Expected score for home team
            E_home = 1 / (
                1 + 10 ** (-((home_elo_before - away_elo_before + HFA) / 400))
            )

            # Actual game result
            if game["home_points"] > game["away_points"]:
                actual_result = 1.0  # Home win
            elif game["home_points"] < game["away_points"]:
                actual_result = 0.0  # Home loss
            else:
                actual_result = 0.5  # Tie

            # Update Elo ratings AFTER capturing pre-game values
            team_elos[home] += K * (actual_result - E_home)
            team_elos[away] += K * ((1 - actual_result) - (1 - E_home))

            # Store pre-game Elo features
            results.append(
                {
                    "game_id": game["game_id"],
                    "home_elo_before": home_elo_before,
                    "away_elo_before": away_elo_before,
                    "elo_diff": home_elo_before - away_elo_before,
                }
            )

        return pd.DataFrame(results)
