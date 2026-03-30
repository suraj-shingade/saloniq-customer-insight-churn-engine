"""
Model training functions for the SalonIQ Customer Insight Engine.

Each function accepts pre-split training data and returns a fitted model instance.
Hyperparameters are tuned for the salon-domain synthetic dataset and should be
revisited when retraining on production-scale data.
"""

import logging

import numpy as np
from xgboost import XGBClassifier, XGBRegressor

logger = logging.getLogger(__name__)


def _compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """Compute scale_pos_weight as the ratio of negative to positive samples."""
    n_positive = int(np.sum(y_train == 1))
    n_negative = int(np.sum(y_train == 0))
    if n_positive == 0:
        logger.warning("No positive samples found; defaulting scale_pos_weight to 1.0")
        return 1.0
    return n_negative / n_positive


def train_churn_model(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    """Train a churn-prediction classifier (binary: churned within 90 days)."""
    scale_pos_weight = _compute_scale_pos_weight(y_train)
    logger.info("Churn model scale_pos_weight=%.4f", scale_pos_weight)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
    )
    model.fit(X_train, y_train)
    return model


def train_nextvisit_model(X_train: np.ndarray, y_train: np.ndarray) -> XGBRegressor:
    """Train a regressor predicting the typical inter-visit interval in days."""
    model = XGBRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_ltv_model(X_train: np.ndarray, y_train: np.ndarray) -> XGBRegressor:
    """Train a regressor predicting 12-month customer lifetime value."""
    model = XGBRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_upsell_model(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    """Train a classifier predicting likelihood of accepting premium services."""
    scale_pos_weight = _compute_scale_pos_weight(y_train)
    logger.info("Upsell model scale_pos_weight=%.4f", scale_pos_weight)

    model = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
    )
    model.fit(X_train, y_train)
    return model
