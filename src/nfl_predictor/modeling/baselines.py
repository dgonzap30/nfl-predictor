"""Baseline models for NFL prediction (V0)."""

from typing import Literal

import numpy as np
import pandas as pd


class HomeTeamBaseline:
    """Baseline that always predicts home team wins."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict home team always wins.

        Args:
            X: DataFrame with game features

        Returns:
            Array of 1s (home team wins)
        """
        return np.ones(len(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of home team win.

        Args:
            X: DataFrame with game features

        Returns:
            Array of [0, 1] probabilities (100% home win)
        """
        n = len(X)
        return np.column_stack([np.zeros(n), np.ones(n)])


class EloBaseline:
    """Baseline that predicts based on Elo ratings.

    Assumes X has columns: home_elo_before, away_elo_before
    """

    def __init__(self, elo_factor: float = 400.0):
        """Initialize Elo baseline.

        Args:
            elo_factor: Elo rating scale factor (default: 400)
        """
        self.elo_factor = elo_factor

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict winner based on higher Elo.

        Args:
            X: DataFrame with home_elo_before, away_elo_before

        Returns:
            Binary predictions (1 = home win)
        """
        return (X["home_elo_before"] > X["away_elo_before"]).astype(int).values

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict win probability from Elo difference.

        Uses standard Elo expected score formula:
        E = 1 / (1 + 10^(-(elo_diff)/400))

        Args:
            X: DataFrame with home_elo_before, away_elo_before

        Returns:
            Array of [p_away_win, p_home_win] probabilities
        """
        elo_diff = X["home_elo_before"] - X["away_elo_before"]
        p_home_win = 1.0 / (1.0 + 10 ** (-elo_diff / self.elo_factor))
        p_away_win = 1.0 - p_home_win

        return np.column_stack([p_away_win, p_home_win])


class VegasBaseline:
    """Baseline that uses Vegas betting lines.

    Assumes X has column: spread_closing (home team perspective)
    """

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict winner based on Vegas spread.

        Args:
            X: DataFrame with spread_closing

        Returns:
            Binary predictions (1 = home win if spread > 0)
        """
        return (X["spread_closing"] > 0).astype(int).values

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Convert spread to win probability.

        Uses empirical conversion: p = 1 / (1 + exp(-spread/3.3))

        Args:
            X: DataFrame with spread_closing

        Returns:
            Array of [p_away_win, p_home_win] probabilities
        """
        # Empirical conversion factor (~3.3 points per standard deviation)
        spread = X["spread_closing"]
        p_home_win = 1.0 / (1.0 + np.exp(-spread / 3.3))
        p_away_win = 1.0 - p_home_win

        return np.column_stack([p_away_win, p_home_win])


class MeanScoreBaseline:
    """Baseline that predicts league-average scores.

    For margin/total regression.
    """

    def __init__(self, target: Literal["margin", "total"] = "margin"):
        """Initialize mean score baseline.

        Args:
            target: Prediction target ('margin' or 'total')
        """
        self.target = target
        self.mean_value_ = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MeanScoreBaseline":
        """Fit by computing mean of target.

        Args:
            X: Features (not used)
            y: Target values

        Returns:
            Self
        """
        self.mean_value_ = y.mean()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict mean value for all games.

        Args:
            X: DataFrame with game features

        Returns:
            Array of mean predictions
        """
        if self.mean_value_ is None:
            raise ValueError("Model must be fit before predicting")

        return np.full(len(X), self.mean_value_)
