"""Team rolling statistics - recent performance features."""

from typing import Any, Dict

import pandas as pd


class TeamRollingStats:
    """Feature group for team rolling statistics.

    Computes rolling averages of team performance over recent games.
    Uses shift(1) to prevent data leakage - only uses games BEFORE current game.
    """

    name = "team_rolling_v0"

    def build_features(
        self,
        games_base: pd.DataFrame,
        aux_tables: Dict[str, pd.DataFrame],
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """Build team rolling statistics features.

        Args:
            games_base: DataFrame with base game information
            aux_tables: Dictionary of auxiliary tables (not used)
            config: Configuration dictionary with optional 'window' param

        Returns:
            DataFrame with game_id and rolling stats features

        Raises:
            ValueError: If required columns are missing
        """
        # Verify required columns
        required_cols = [
            "game_id",
            "game_date",
            "season",
            "home_team",
            "away_team",
            "home_points",
            "away_points",
        ]
        missing = set(required_cols) - set(games_base.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Get window size from config (default: 3)
        window = config.get("window", 3)

        # Step 1: Build long-format team-game table
        team_games = self._build_team_games_table(games_base)

        # Step 2: Compute rolling stats with shift to prevent leakage
        team_stats = self._compute_rolling_stats(team_games, window)

        # Step 3: Pivot back to game-level format
        game_features = self._pivot_to_game_level(team_stats)

        return game_features

    def _build_team_games_table(self, games_base: pd.DataFrame) -> pd.DataFrame:
        """Transform games_base into long-format team-game table.

        Each game creates 2 rows: one for home team, one for away team.

        Args:
            games_base: Game-level DataFrame

        Returns:
            Long-format DataFrame with one row per team per game
        """
        # Home team rows
        home_rows = pd.DataFrame(
            {
                "game_id": games_base["game_id"],
                "game_date": games_base["game_date"],
                "season": games_base["season"],
                "team": games_base["home_team"],
                "opponent": games_base["away_team"],
                "is_home": True,
                "points_for": games_base["home_points"],
                "points_against": games_base["away_points"],
                "won": (games_base["home_points"] > games_base["away_points"]).astype(
                    int
                ),
                "margin": games_base["home_points"] - games_base["away_points"],
            }
        )

        # Away team rows
        away_rows = pd.DataFrame(
            {
                "game_id": games_base["game_id"],
                "game_date": games_base["game_date"],
                "season": games_base["season"],
                "team": games_base["away_team"],
                "opponent": games_base["home_team"],
                "is_home": False,
                "points_for": games_base["away_points"],
                "points_against": games_base["home_points"],
                "won": (games_base["away_points"] > games_base["home_points"]).astype(
                    int
                ),
                "margin": games_base["away_points"] - games_base["home_points"],
            }
        )

        # Combine and sort by team, then date
        team_games = pd.concat([home_rows, away_rows], ignore_index=True)
        team_games = team_games.sort_values(["team", "game_date"]).reset_index(
            drop=True
        )

        return team_games

    def _compute_rolling_stats(
        self, team_games: pd.DataFrame, window: int
    ) -> pd.DataFrame:
        """Compute rolling statistics for each team.

        CRITICAL: Uses shift(1) to ensure rolling stats only use PRIOR games.

        Args:
            team_games: Long-format team-game table
            window: Rolling window size (e.g., 3 for last 3 games)

        Returns:
            team_games with rolling stats columns added
        """
        df = team_games.copy()

        # CRITICAL: shift(1) ensures we only use games BEFORE the current game
        # This prevents data leakage
        df["pts_for_last3"] = df.groupby("team")["points_for"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

        df["pts_against_last3"] = df.groupby("team")["points_against"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

        df["win_pct_last3"] = df.groupby("team")["won"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

        return df

    def _pivot_to_game_level(self, team_stats: pd.DataFrame) -> pd.DataFrame:
        """Pivot team-level stats back to game-level format.

        Joins home team and away team stats to create game-level features.

        Args:
            team_stats: Long-format table with rolling stats

        Returns:
            Game-level DataFrame with home_* and away_* features
        """
        # Select columns to include
        stat_cols = ["game_id", "team", "pts_for_last3", "pts_against_last3", "win_pct_last3"]

        # Separate home and away team stats
        home_stats = team_stats[team_stats["is_home"]][stat_cols].copy()
        away_stats = team_stats[~team_stats["is_home"]][stat_cols].copy()

        # Rename columns with home_ prefix
        home_stats = home_stats.rename(
            columns={
                "team": "home_team",
                "pts_for_last3": "home_pts_for_last3",
                "pts_against_last3": "home_pts_against_last3",
                "win_pct_last3": "home_win_pct_last3",
            }
        )

        # Rename columns with away_ prefix
        away_stats = away_stats.rename(
            columns={
                "team": "away_team",
                "pts_for_last3": "away_pts_for_last3",
                "pts_against_last3": "away_pts_against_last3",
                "win_pct_last3": "away_win_pct_last3",
            }
        )

        # Merge home and away stats
        game_features = home_stats.merge(away_stats, on="game_id", how="inner")

        # Drop team columns (redundant with games_base)
        game_features = game_features.drop(columns=["home_team", "away_team"])

        return game_features
