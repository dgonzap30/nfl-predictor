"""Placeholder tests for NFL Predictor.

This file ensures the test suite can run. Replace with actual tests as you build features.
"""

import pytest

from nfl_predictor import __version__


def test_version():
    """Test that version is defined."""
    assert __version__ == "0.1.0"


def test_imports():
    """Test that core modules can be imported."""
    from nfl_predictor.data import schemas
    from nfl_predictor.features import feature_registry
    from nfl_predictor.modeling import baselines, metrics
    from nfl_predictor.utils import validation

    assert schemas is not None
    assert feature_registry is not None
    assert baselines is not None
    assert metrics is not None
    assert validation is not None


class TestValidation:
    """Tests for validation utilities."""

    def test_assert_no_future_leakage_passes(self):
        """Test that validation passes when no leakage."""
        import pandas as pd

        from nfl_predictor.utils.validation import assert_no_future_leakage

        df = pd.DataFrame(
            {
                "game_id": ["2023_01_BUF_KC"],
                "game_date": pd.to_datetime(["2023-09-10"]),
                "feature_max_date": pd.to_datetime(["2023-09-09"]),
            }
        )

        # Should not raise
        assert_no_future_leakage(df)

    def test_assert_no_future_leakage_fails(self):
        """Test that validation fails when leakage detected."""
        import pandas as pd

        from nfl_predictor.utils.validation import assert_no_future_leakage

        df = pd.DataFrame(
            {
                "game_id": ["2023_01_BUF_KC"],
                "game_date": pd.to_datetime(["2023-09-10"]),
                "feature_max_date": pd.to_datetime(["2023-09-11"]),  # Future data!
            }
        )

        with pytest.raises(ValueError, match="Data leakage detected"):
            assert_no_future_leakage(df)


class TestBaselines:
    """Tests for baseline models."""

    def test_home_team_baseline(self):
        """Test HomeTeamBaseline predictions."""
        import pandas as pd

        from nfl_predictor.modeling.baselines import HomeTeamBaseline

        model = HomeTeamBaseline()
        X = pd.DataFrame({"game_id": ["g1", "g2", "g3"]})

        preds = model.predict(X)
        assert len(preds) == 3
        assert all(preds == 1)

        proba = model.predict_proba(X)
        assert proba.shape == (3, 2)
        assert all(proba[:, 1] == 1.0)

    def test_elo_baseline(self):
        """Test EloBaseline predictions."""
        import pandas as pd

        from nfl_predictor.modeling.baselines import EloBaseline

        model = EloBaseline()
        X = pd.DataFrame(
            {
                "home_elo_before": [1600, 1400, 1500],
                "away_elo_before": [1400, 1600, 1500],
            }
        )

        preds = model.predict(X)
        assert preds[0] == 1  # Home team higher Elo
        assert preds[1] == 0  # Away team higher Elo
        # preds[2] could be either when equal

        proba = model.predict_proba(X)
        assert proba.shape == (3, 2)
        assert proba[0, 1] > 0.5  # Home team favored
        assert proba[1, 1] < 0.5  # Away team favored


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_winner_accuracy(self):
        """Test winner accuracy calculation."""
        import numpy as np

        from nfl_predictor.modeling.metrics import winner_accuracy

        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])

        acc = winner_accuracy(y_true, y_pred)
        assert acc == 1.0

        y_pred_wrong = np.array([0, 1, 0, 1])
        acc_wrong = winner_accuracy(y_true, y_pred_wrong)
        assert acc_wrong == 0.0

    def test_score_hit_rate(self):
        """Test score hit rate calculation."""
        import numpy as np

        from nfl_predictor.modeling.metrics import score_hit_rate

        y_true = np.array([24, 21, 30, 17])
        y_pred = np.array([26, 20, 31, 14])  # Errors: 2, 1, 1, 3

        hit_rate = score_hit_rate(y_true, y_pred, threshold=3.0)
        assert hit_rate == 1.0  # All within 3 points

        hit_rate_strict = score_hit_rate(y_true, y_pred, threshold=1.0)
        assert hit_rate_strict == 0.5  # Only 2/4 within 1 point
