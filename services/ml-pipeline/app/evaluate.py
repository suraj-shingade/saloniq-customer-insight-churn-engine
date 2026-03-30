"""
Model evaluation utilities for the SalonIQ Customer Insight Engine.

Provides standard metric computation for both classification and regression
models, returning results as dictionaries suitable for structured logging
and metadata persistence.
"""

import logging

import numpy as np
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

logger = logging.getLogger(__name__)


def evaluate_classifier(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate a binary classifier and return key performance metrics.

    Returns:
        Dictionary with roc_auc, pr_auc, precision, recall, and f1.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_prob)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
    }
    return metrics


def evaluate_regressor(
    model, X_test: np.ndarray, y_test: np.ndarray, metric_type: str = "general"
) -> dict:
    """
    Evaluate a regression model and return key performance metrics.

    Args:
        model: Trained regressor.
        X_test: Test feature matrix.
        y_test: True target values.
        metric_type: If "ltv", additionally computes MAPE with zero-division protection.

    Returns:
        Dictionary with mae, rmse, r_squared, and optionally mape.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "r_squared": round(float(r2_score(y_test, y_pred)), 4),
    }

    if metric_type == "ltv":
        non_zero_mask = np.abs(y_test) > 1e-8
        if np.any(non_zero_mask):
            mape = float(
                np.mean(
                    np.abs(
                        (y_test[non_zero_mask] - y_pred[non_zero_mask])
                        / y_test[non_zero_mask]
                    )
                )
                * 100.0
            )
            metrics["mape"] = round(mape, 4)
        else:
            logger.warning("All y_test values are zero; MAPE is undefined, skipping")
            metrics["mape"] = None

    return metrics
