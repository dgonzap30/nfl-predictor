"""Evaluation metrics for NFL prediction models."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)


def winner_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate winner prediction accuracy.

    Args:
        y_true: True labels (0/1)
        y_pred: Predicted probabilities or binary predictions

    Returns:
        Accuracy score
    """
    if y_pred.max() <= 1.0 and y_pred.min() >= 0.0:
        # Probabilities - convert to binary
        y_pred_binary = (y_pred >= 0.5).astype(int)
    else:
        y_pred_binary = y_pred

    return accuracy_score(y_true, y_pred_binary)


def winner_log_loss(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Calculate log loss for winner predictions.

    Args:
        y_true: True labels (0/1)
        y_pred_proba: Predicted probabilities

    Returns:
        Log loss
    """
    return log_loss(y_true, y_pred_proba)


def winner_brier_score(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Calculate Brier score for winner predictions.

    Args:
        y_true: True labels (0/1)
        y_pred_proba: Predicted probabilities

    Returns:
        Brier score
    """
    return brier_score_loss(y_true, y_pred_proba)


def winner_roc_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Calculate ROC AUC for winner predictions.

    Args:
        y_true: True labels (0/1)
        y_pred_proba: Predicted probabilities

    Returns:
        ROC AUC score
    """
    return roc_auc_score(y_true, y_pred_proba)


def score_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean absolute error for score predictions.

    Args:
        y_true: True scores
        y_pred: Predicted scores

    Returns:
        MAE
    """
    return mean_absolute_error(y_true, y_pred)


def score_hit_rate(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 3.0
) -> float:
    """Calculate hit rate (percentage within threshold).

    Args:
        y_true: True scores
        y_pred: Predicted scores
        threshold: Error threshold (default: 3 points)

    Returns:
        Hit rate (0-1)
    """
    errors = np.abs(y_true - y_pred)
    return (errors <= threshold).mean()


def expected_calibration_error(
    y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10
) -> float:
    """Calculate Expected Calibration Error (ECE).

    Args:
        y_true: True labels (0/1)
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins for calibration

    Returns:
        ECE score
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Find predictions in this bin
        in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)

        if in_bin.sum() > 0:
            # Average predicted probability in bin
            avg_pred = y_pred_proba[in_bin].mean()
            # Average true frequency in bin
            avg_true = y_true[in_bin].mean()
            # Weight by bin size
            weight = in_bin.sum() / len(y_true)

            ece += weight * np.abs(avg_pred - avg_true)

    return ece
