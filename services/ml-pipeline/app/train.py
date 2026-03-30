"""
ML Training Pipeline -- Main Entry Point

Orchestrates the full batch training workflow for the SalonIQ Customer Insight
Engine: connects to PostgreSQL, loads pre-computed features, engineers labels,
trains four ML models (churn, next-visit, LTV, upsell), evaluates each on a
held-out test set, persists artifacts to disk, and exits.

Usage:
    python -m app.train
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import psycopg2
from sklearn.model_selection import train_test_split

from app.config import (
    LOG_LEVEL,
    MODEL_PATH,
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
)
from app.evaluate import evaluate_classifier, evaluate_regressor
from app.models import (
    train_churn_model,
    train_ltv_model,
    train_nextvisit_model,
    train_upsell_model,
)

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "visit_count_total",
    "visit_count_90d",
    "visit_count_180d",
    "days_since_last_visit",
    "avg_inter_visit_days",
    "std_inter_visit_days",
    "inter_visit_trend",
    "visit_regularity_index",
    "overdue_ratio",
    "dow_mode",
    "hour_mode",
    "avg_ticket_value",
    "ticket_value_trend",
    "max_ticket_value",
    "service_variety_index",
    "has_color_service",
    "has_treatment_service",
    "color_frequency_ratio",
    "stylist_consistency_ratio",
    "stylist_change_count",
    "multi_stylist_flag",
    "walkin_ratio",
    "noshow_rate",
    "cancellation_rate",
    "avg_booking_lead_days",
    "product_purchase_count",
    "product_purchase_frequency",
    "product_category_count",
]

BOOLEAN_COLUMNS = ["has_color_service", "has_treatment_service", "multi_stylist_flag"]

FEATURE_QUERY = """
    SELECT cf.*, v_stats.last_visit_date, v_stats.total_visits_after
    FROM customer_features cf
    LEFT JOIN (
        SELECT customer_id, salon_id,
               MAX(visit_date) AS last_visit_date,
               COUNT(*) AS total_visits_after
        FROM visits
        WHERE status = 'COMPLETED'
        GROUP BY customer_id, salon_id
    ) v_stats
      ON cf.customer_id = v_stats.customer_id
     AND cf.salon_id    = v_stats.salon_id
"""

DB_CONNECT_MAX_RETRIES = 10
DB_CONNECT_RETRY_DELAY_SEC = 3
MIN_SAMPLE_WARNING_THRESHOLD = 50


def _configure_logging() -> None:
    """Set up structured logging with a consistent format."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )


def _connect_postgres() -> psycopg2.extensions.connection:
    """
    Establish a PostgreSQL connection with retry logic.

    Retries up to DB_CONNECT_MAX_RETRIES times with a fixed delay between
    attempts. Raises the last connection error if all retries are exhausted.
    """
    last_error = None
    for attempt in range(1, DB_CONNECT_MAX_RETRIES + 1):
        try:
            conn = psycopg2.connect(
                host=POSTGRES_HOST,
                port=POSTGRES_PORT,
                dbname=POSTGRES_DB,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD,
            )
            logger.info("Connected to PostgreSQL at %s:%s/%s (attempt %d)",
                        POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, attempt)
            return conn
        except psycopg2.OperationalError as exc:
            last_error = exc
            logger.warning("DB connection attempt %d/%d failed: %s",
                           attempt, DB_CONNECT_MAX_RETRIES, exc)
            if attempt < DB_CONNECT_MAX_RETRIES:
                time.sleep(DB_CONNECT_RETRY_DELAY_SEC)

    logger.error("Exhausted all %d DB connection attempts", DB_CONNECT_MAX_RETRIES)
    raise last_error


def _load_features(conn: psycopg2.extensions.connection) -> pd.DataFrame:
    """Load the pre-computed feature table joined with visit statistics."""
    logger.info("Loading feature data from PostgreSQL")
    df = pd.read_sql(FEATURE_QUERY, conn)
    logger.info("Loaded %d rows with %d columns", len(df), len(df.columns))
    return df


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the feature matrix: cast booleans, impute missing values,
    and drop columns not used as model inputs.
    """
    for col in BOOLEAN_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(int)

    if "primary_stylist_id" in df.columns:
        df = df.drop(columns=["primary_stylist_id"])

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            logger.warning("Expected feature column '%s' missing; filling with 0", col)
            df[col] = 0
            continue
        if df[col].dtype in ("int64", "int32", "Int64"):
            df[col] = df[col].fillna(0)
        else:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0)

    return df


def _engineer_labels(df: pd.DataFrame) -> dict:
    """
    Derive target labels for all four models.

    Returns a dictionary keyed by model name, each value being a tuple of
    (feature DataFrame subset, label Series) after filtering invalid rows.
    """
    labels = {}

    # --- Churn (binary) ---
    churn_mask = df["days_since_last_visit"].notna()
    churn_df = df.loc[churn_mask].copy()
    churn_df["churned_90d"] = (churn_df["days_since_last_visit"] > 90).astype(int)
    labels["churn"] = (churn_df[FEATURE_COLUMNS], churn_df["churned_90d"])

    # --- Next-visit (regression) ---
    nv_mask = (df["avg_inter_visit_days"].notna()) & (df["avg_inter_visit_days"] > 0)
    nv_df = df.loc[nv_mask].copy()
    labels["nextvisit"] = (nv_df[FEATURE_COLUMNS], nv_df["avg_inter_visit_days"])

    # --- LTV (regression) ---
    ltv_mask = df["visit_count_total"].notna() & (df["visit_count_total"] > 0)
    ltv_df = df.loc[ltv_mask].copy()
    ltv_df["ltv_12m"] = (
        ltv_df["avg_ticket_value"]
        * (365.0 / ltv_df["avg_inter_visit_days"].clip(lower=1))
    )
    labels["ltv"] = (ltv_df[FEATURE_COLUMNS], ltv_df["ltv_12m"])

    # --- Upsell (binary) ---
    upsell_df = df.copy()
    upsell_df["upsell_accepted"] = (
        (upsell_df["has_color_service"] == 1) | (upsell_df["has_treatment_service"] == 1)
    ).astype(int)
    labels["upsell"] = (upsell_df[FEATURE_COLUMNS], upsell_df["upsell_accepted"])

    return labels


def _train_and_evaluate_all(labels: dict) -> tuple:
    """
    Train all four models, evaluate each on a held-out test set, and return
    the trained model objects together with their evaluation metrics.

    Returns:
        (trained_models dict, metadata dict with sample counts and metrics)
    """
    model_configs = {
        "churn": {
            "train_fn": train_churn_model,
            "eval_fn": lambda m, Xt, yt: evaluate_classifier(m, Xt, yt),
            "type": "XGBClassifier",
            "file": "churn_model.joblib",
            "stratify": True,
        },
        "nextvisit": {
            "train_fn": train_nextvisit_model,
            "eval_fn": lambda m, Xt, yt: evaluate_regressor(m, Xt, yt),
            "type": "XGBRegressor",
            "file": "nextvisit_model.joblib",
            "stratify": False,
        },
        "ltv": {
            "train_fn": train_ltv_model,
            "eval_fn": lambda m, Xt, yt: evaluate_regressor(m, Xt, yt, metric_type="ltv"),
            "type": "XGBRegressor",
            "file": "ltv_model.joblib",
            "stratify": False,
        },
        "upsell": {
            "train_fn": train_upsell_model,
            "eval_fn": lambda m, Xt, yt: evaluate_classifier(m, Xt, yt),
            "type": "XGBClassifier",
            "file": "upsell_model.joblib",
            "stratify": True,
        },
    }

    trained_models = {}
    model_metadata = {}
    total_train_samples = 0
    total_test_samples = 0

    for name, cfg in model_configs.items():
        try:
            X, y = labels[name]
            n_samples = len(X)

            if n_samples < MIN_SAMPLE_WARNING_THRESHOLD:
                logger.warning(
                    "Model '%s' has only %d samples (threshold: %d); training anyway",
                    name, n_samples, MIN_SAMPLE_WARNING_THRESHOLD,
                )

            stratify_param = y if cfg["stratify"] else None

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify_param,
            )

            logger.info("Training %s on %d samples...", name, len(X_train))
            model = cfg["train_fn"](X_train.values, y_train.values)

            metrics = cfg["eval_fn"](model, X_test.values, y_test.values)
            logger.info("Model %s metrics: %s", name, metrics)

            trained_models[name] = model
            model_metadata[name] = {
                "file": cfg["file"],
                "type": cfg["type"],
                "metrics": metrics,
            }

            total_train_samples = max(total_train_samples, len(X_train))
            total_test_samples = max(total_test_samples, len(X_test))

        except Exception:
            logger.exception("Failed to train model '%s'; skipping", name)

    return trained_models, model_metadata, total_train_samples, total_test_samples


def _save_artifacts(
    trained_models: dict,
    model_metadata: dict,
    total_train_samples: int,
    total_test_samples: int,
) -> None:
    """Persist trained model binaries and a metadata manifest to MODEL_PATH."""
    os.makedirs(MODEL_PATH, exist_ok=True)

    for name, model in trained_models.items():
        artifact_path = os.path.join(MODEL_PATH, model_metadata[name]["file"])
        joblib.dump(model, artifact_path)
        logger.info("Saved %s to %s", name, artifact_path)

    manifest = {
        "version": "v1.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_samples": total_train_samples,
        "test_samples": total_test_samples,
        "models": model_metadata,
        "feature_names": FEATURE_COLUMNS,
    }

    manifest_path = os.path.join(MODEL_PATH, "model_metadata.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Saved model metadata to %s", manifest_path)


def main() -> None:
    """Execute the full training pipeline."""
    _configure_logging()
    logger.info("ML Training Pipeline starting")

    conn = _connect_postgres()
    try:
        df = _load_features(conn)
        df = _prepare_features(df)
        labels = _engineer_labels(df)

        trained_models, model_metadata, n_train, n_test = _train_and_evaluate_all(labels)

        if not trained_models:
            logger.error("No models were trained successfully; exiting with error")
            sys.exit(1)

        _save_artifacts(trained_models, model_metadata, n_train, n_test)
        logger.info("Training pipeline complete. All models saved to %s", MODEL_PATH)
    finally:
        conn.close()
        logger.info("Database connection closed")


if __name__ == "__main__":
    main()
