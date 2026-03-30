import json
import logging
import os
import time
from collections import Counter
from datetime import datetime, timezone

import joblib
import numpy as np
import psycopg2
import psycopg2.extras
import redis
from fastapi import APIRouter, HTTPException, Query

from app.config import settings
from app.schemas import (
    AtRiskCustomer,
    CaseStudyResponse,
    CustomerPrediction,
    DashboardKPIs,
    DashboardResponse,
    RiskFactor,
    SalonInfo,
    VisitRecord,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Module-level state populated during startup
# ---------------------------------------------------------------------------
_models: dict = {}
_model_metadata: dict = {}
_pg_conn = None
_redis_client = None
_predictions_ready: bool = False

RISK_LEVEL_HIERARCHY = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

SUGGESTED_ACTIONS = {
    "CRITICAL": "OFFER_PROMOTION",
    "HIGH": "SEND_FOLLOWUP",
    "MEDIUM": "BOOK_APPOINTMENT",
    "LOW": "MONITOR",
}

DASHBOARD_CACHE_TTL_SECONDS = 300
ANALYTICS_CACHE_TTL_SECONDS = 300

SEGMENT_LOYAL_REGULAR = "Loyal Regular"
SEGMENT_AT_RISK = "At Risk"
SEGMENT_NEW = "New"
SEGMENT_DECLINING = "Declining"
SEGMENT_SEASONAL = "Seasonal"
SEGMENT_SPORADIC = "Sporadic"

SEGMENT_DISPLAY_ORDER = (
    SEGMENT_LOYAL_REGULAR,
    SEGMENT_AT_RISK,
    SEGMENT_NEW,
    SEGMENT_DECLINING,
    SEGMENT_SEASONAL,
    SEGMENT_SPORADIC,
)

SEGMENT_COLORS = {
    SEGMENT_LOYAL_REGULAR: "#10B981",
    SEGMENT_AT_RISK: "#EF4444",
    SEGMENT_NEW: "#3B82F6",
    SEGMENT_DECLINING: "#F59E0B",
    SEGMENT_SEASONAL: "#8B5CF6",
    SEGMENT_SPORADIC: "#64748B",
}

_SALON_EXISTS_QUERY = "SELECT 1 FROM salons WHERE salon_id = %s"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_pg_connection():
    """Return the shared PostgreSQL connection, reconnecting if necessary."""
    global _pg_conn
    if _pg_conn is None or _pg_conn.closed:
        _pg_conn = psycopg2.connect(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            dbname=settings.POSTGRES_DB,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
        )
        _pg_conn.autocommit = True
        logger.info("PostgreSQL connection established.")
    return _pg_conn


def _get_redis_client():
    """Return the shared Redis client."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=True,
        )
        _redis_client.ping()
        logger.info("Redis connection established.")
    return _redis_client


def _classify_risk_level(probability: float) -> str:
    """Map churn probability to a risk level string."""
    if probability >= 0.85:
        return "CRITICAL"
    if probability >= 0.60:
        return "HIGH"
    if probability >= 0.30:
        return "MEDIUM"
    return "LOW"


def _compute_risk_factors(row: dict, overdue_ratio: float) -> list[dict]:
    """Derive the top-3 risk factors for a customer from feature values."""
    factors: list[tuple[str, str, float]] = []

    days_since_last_visit = float(row.get("days_since_last_visit") or 0)
    avg_inter_visit_days = float(row.get("avg_inter_visit_days") or 0)
    inter_visit_trend = float(row.get("inter_visit_trend") or 0)
    stylist_change_count = int(row.get("stylist_change_count") or 0)
    noshow_rate = float(row.get("noshow_rate") or 0)
    ticket_value_trend = float(row.get("ticket_value_trend") or 0)
    visit_count_90d = int(row.get("visit_count_90d") or 0)
    visit_count_total = int(row.get("visit_count_total") or 0)
    visit_regularity_index = float(row.get("visit_regularity_index") or 0)

    if overdue_ratio > 1.5:
        factors.append((
            "overdue_ratio",
            f"Visit overdue by {overdue_ratio:.1f}x typical interval",
            (overdue_ratio - 1) * 0.3,
        ))

    if inter_visit_trend > 0.5:
        factors.append((
            "inter_visit_trend",
            "Visit intervals increasing over recent visits",
            inter_visit_trend * 0.2,
        ))

    if stylist_change_count > 1:
        factors.append((
            "stylist_change",
            f"Changed stylist {stylist_change_count} times recently",
            stylist_change_count * 0.1,
        ))

    if days_since_last_visit > avg_inter_visit_days * 1.5 and avg_inter_visit_days > 0:
        factors.append((
            "days_overdue",
            f"No visit in {int(days_since_last_visit)} days (avg: {avg_inter_visit_days:.0f} days)",
            overdue_ratio * 0.25,
        ))

    if noshow_rate > 0.1:
        factors.append((
            "noshow",
            f"Elevated no-show rate ({noshow_rate * 100:.0f}%)",
            noshow_rate * 0.15,
        ))

    if ticket_value_trend < -100:
        factors.append((
            "spending_decline",
            "Spending declining over recent visits",
            abs(ticket_value_trend) * 0.001,
        ))

    if visit_count_90d == 0 and visit_count_total > 0:
        factors.append((
            "no_recent_visits",
            "No visits in the last 90 days",
            0.3,
        ))

    if visit_regularity_index < 0.3 and visit_count_total >= 3:
        factors.append((
            "irregular",
            "Irregular visit pattern",
            0.2,
        ))

    # Sort descending by impact, take top 3
    factors.sort(key=lambda f: f[2], reverse=True)
    top = factors[:3]

    if not top:
        top = [("general", "Within normal parameters", 0.0)]

    return [
        {"feature": f[0], "description": f[1], "impact": round(f[2], 4)}
        for f in top
    ]


def _build_feature_vector(row: dict, feature_names: list[str]) -> np.ndarray:
    """Construct the ordered feature vector from a customer_features row."""
    values = []
    for name in feature_names:
        val = row.get(name)
        if isinstance(val, bool):
            val = int(val)
        if val is None:
            val = 0
        values.append(float(val))
    arr = np.array([values])
    # Replace any remaining NaN with 0
    arr = np.nan_to_num(arr, nan=0.0)
    return arr


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

async def startup() -> None:
    """Load models, compute predictions, populate caches."""
    global _predictions_ready

    model_path = settings.MODEL_PATH
    conn = _get_pg_connection()
    rd = _get_redis_client()

    # -- Load models ----------------------------------------------------------
    model_files = {
        "churn": "churn_model.joblib",
        "nextvisit": "nextvisit_model.joblib",
        "ltv": "ltv_model.joblib",
        "upsell": "upsell_model.joblib",
    }
    for key, filename in model_files.items():
        path = os.path.join(model_path, filename)
        _models[key] = joblib.load(path)

    metadata_path = os.path.join(model_path, "model_metadata.json")
    with open(metadata_path, "r") as fh:
        _model_metadata.update(json.load(fh))

    logger.info("Loaded %d models", len(_models))

    # -- Read customer features ------------------------------------------------
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM customer_features")
        rows = cur.fetchall()

    feature_names: list[str] = _model_metadata["feature_names"]
    model_version: str = _model_metadata.get("version", "unknown")
    now_iso = datetime.now(timezone.utc).isoformat()

    logger.info("Computing predictions for %d customers...", len(rows))

    predictions_by_salon: dict[str, list[dict]] = {}

    for row in rows:
        customer_id = str(row["customer_id"])
        salon_id = str(row["salon_id"])

        features = _build_feature_vector(row, feature_names)

        # -- Run inference -----------------------------------------------------
        churn_probability = float(_models["churn"].predict_proba(features)[0][1])
        churn_probability = round(max(0.0, min(1.0, churn_probability)), 4)

        predicted_next_visit_days = int(_models["nextvisit"].predict(features)[0])
        predicted_next_visit_days = max(1, min(365, predicted_next_visit_days))

        estimated_ltv_12m = float(_models["ltv"].predict(features)[0])
        estimated_ltv_12m = round(max(0.0, min(999999.0, estimated_ltv_12m)), 2)

        upsell_propensity = float(_models["upsell"].predict_proba(features)[0][1])
        upsell_propensity = round(max(0.0, min(1.0, upsell_propensity)), 4)

        risk_level = _classify_risk_level(churn_probability)

        avg_inter_visit_days = float(row.get("avg_inter_visit_days") or 0)
        overdue_ratio = (
            float(row.get("days_since_last_visit") or 0) / avg_inter_visit_days
            if avg_inter_visit_days > 0
            else 0.0
        )

        risk_factors = _compute_risk_factors(row, overdue_ratio)

        prediction_record = {
            "customer_id": customer_id,
            "salon_id": salon_id,
            "churn_probability": churn_probability,
            "risk_level": risk_level,
            "risk_factors": json.dumps(risk_factors),
            "predicted_next_visit_days": predicted_next_visit_days,
            "estimated_ltv_12m": estimated_ltv_12m,
            "upsell_propensity": upsell_propensity,
            "model_version": model_version,
            "predicted_at": now_iso,
        }

        # -- Persist to PostgreSQL ---------------------------------------------
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO predictions (
                    customer_id, salon_id, churn_probability, risk_level,
                    risk_factors, predicted_next_visit_days, estimated_ltv_12m,
                    upsell_propensity, model_version, predicted_at
                ) VALUES (
                    %(customer_id)s, %(salon_id)s, %(churn_probability)s,
                    %(risk_level)s, %(risk_factors)s, %(predicted_next_visit_days)s,
                    %(estimated_ltv_12m)s, %(upsell_propensity)s,
                    %(model_version)s, %(predicted_at)s
                )
                ON CONFLICT (customer_id, salon_id) DO UPDATE SET
                    churn_probability = EXCLUDED.churn_probability,
                    risk_level = EXCLUDED.risk_level,
                    risk_factors = EXCLUDED.risk_factors,
                    predicted_next_visit_days = EXCLUDED.predicted_next_visit_days,
                    estimated_ltv_12m = EXCLUDED.estimated_ltv_12m,
                    upsell_propensity = EXCLUDED.upsell_propensity,
                    model_version = EXCLUDED.model_version,
                    predicted_at = EXCLUDED.predicted_at
                """,
                prediction_record,
            )

        # -- Cache in Redis ----------------------------------------------------
        redis_payload = {
            **prediction_record,
            "risk_factors": risk_factors,  # store as native structure
        }
        rd.set(
            f"predictions:{salon_id}:{customer_id}",
            json.dumps(redis_payload, default=str),
        )

        predictions_by_salon.setdefault(salon_id, []).append(redis_payload)

    logger.info("Predictions complete. %d stored.", len(rows))

    # -- Pre-compute dashboard data per salon ----------------------------------
    for salon_id, preds in predictions_by_salon.items():
        dashboard = _compute_dashboard_for_salon(conn, salon_id, preds)
        rd.set(
            f"dashboard:{salon_id}",
            json.dumps(dashboard, default=str),
            ex=DASHBOARD_CACHE_TTL_SECONDS,
        )

    _predictions_ready = True


def _compute_dashboard_for_salon(conn, salon_id: str, preds: list[dict]) -> dict:
    """Build dashboard payload for a single salon from prediction data."""
    total = len(preds)
    at_risk = [p for p in preds if p["risk_level"] in ("MEDIUM", "HIGH", "CRITICAL")]
    at_risk_count = len(at_risk)
    at_risk_percentage = round((at_risk_count / total * 100) if total > 0 else 0.0, 2)

    # Fetch supplemental metrics from customer_features
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                AVG(avg_inter_visit_days) AS avg_ivd,
                AVG(avg_ticket_value) AS avg_tv,
                COUNT(*) FILTER (WHERE days_since_last_visit > 90) AS churned_count,
                COUNT(*) AS total_count
            FROM customer_features
            WHERE salon_id = %s
            """,
            (salon_id,),
        )
        stats = cur.fetchone()

    total_from_features = int(stats["total_count"] or 0)
    churned_count = int(stats["churned_count"] or 0)
    churn_rate = round(
        (churned_count / total_from_features * 100) if total_from_features > 0 else 0.0,
        2,
    )
    retention_rate = round(100.0 - churn_rate, 2)
    avg_inter_visit_days = round(float(stats["avg_ivd"] or 0), 2)
    avg_ticket_value = round(float(stats["avg_tv"] or 0), 2)

    # Revenue in the last 30 days
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COALESCE(SUM(total_amount), 0) AS rev
            FROM visits
            WHERE salon_id = %s
              AND visit_date >= CURRENT_DATE - INTERVAL '30 days'
              AND status = 'COMPLETED'
            """,
            (salon_id,),
        )
        total_revenue_30d = round(float(cur.fetchone()[0] or 0), 2)

    # Risk distribution
    risk_dist: dict[str, int] = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for p in preds:
        risk_dist[p["risk_level"]] = risk_dist.get(p["risk_level"], 0) + 1

    # Top churn drivers -- aggregate risk factors across at-risk customers
    factor_counter: Counter = Counter()
    for p in at_risk:
        rf = p.get("risk_factors") or []
        if isinstance(rf, str):
            rf = json.loads(rf)
        for f in rf:
            factor_counter[f["feature"]] += 1

    top_churn_drivers = [
        {"factor": factor, "count": count}
        for factor, count in factor_counter.most_common(10)
    ]

    # Fetch salon name
    with conn.cursor() as cur:
        cur.execute("SELECT salon_name FROM salons WHERE salon_id = %s", (salon_id,))
        salon_row = cur.fetchone()
    salon_name = salon_row[0] if salon_row else "Unknown"

    return {
        "salon_id": salon_id,
        "salon_name": salon_name,
        "period": "current",
        "kpis": {
            "total_active_customers": total,
            "at_risk_count": at_risk_count,
            "at_risk_percentage": at_risk_percentage,
            "churn_rate_trailing_90d": churn_rate,
            "avg_inter_visit_days": avg_inter_visit_days,
            "retention_rate_trailing_90d": retention_rate,
            "avg_ticket_value": avg_ticket_value,
            "total_revenue_30d": total_revenue_30d,
        },
        "risk_distribution": risk_dist,
        "top_churn_drivers": top_churn_drivers,
    }


async def shutdown() -> None:
    """Release connections on application shutdown."""
    global _pg_conn, _redis_client
    if _pg_conn and not _pg_conn.closed:
        _pg_conn.close()
        logger.info("PostgreSQL connection closed.")
    if _redis_client:
        _redis_client.close()
        logger.info("Redis connection closed.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health")
async def health():
    """Liveness / readiness probe."""
    if _predictions_ready and _models:
        return {
            "status": "healthy",
            "model_version": _model_metadata.get("version", "unknown"),
        }
    raise HTTPException(status_code=503, detail="Service not ready")


@router.get("/v1/salons")
async def list_salons():
    """Return all salons with their customer counts."""
    conn = _get_pg_connection()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT s.salon_id, s.salon_name, s.region, s.city,
                   COUNT(c.customer_id) AS customer_count
            FROM salons s
            LEFT JOIN customers c ON s.salon_id = c.salon_id
            GROUP BY s.salon_id, s.salon_name, s.region, s.city
            ORDER BY s.salon_id
            """
        )
        rows = cur.fetchall()

    salons = [
        SalonInfo(
            salon_id=str(r["salon_id"]),
            salon_name=r["salon_name"],
            region=r["region"],
            city=r["city"],
            customer_count=int(r["customer_count"]),
        ).model_dump()
        for r in rows
    ]
    return {"salons": salons}


@router.get("/v1/salon/{salon_id}/dashboard")
async def salon_dashboard(salon_id: str):
    """Aggregated KPIs and risk distribution for a salon."""
    rd = _get_redis_client()

    # Try Redis cache first
    cached = rd.get(f"dashboard:{salon_id}")
    if cached:
        return DashboardResponse(**json.loads(cached)).model_dump()

    # Compute on demand
    conn = _get_pg_connection()

    # Verify salon exists
    with conn.cursor() as cur:
        cur.execute("SELECT salon_name FROM salons WHERE salon_id = %s", (salon_id,))
        salon_row = cur.fetchone()
    if not salon_row:
        raise HTTPException(status_code=404, detail=f"Salon {salon_id} not found")

    # Fetch predictions for this salon
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT customer_id, salon_id, churn_probability, risk_level,
                   risk_factors, predicted_next_visit_days, estimated_ltv_12m,
                   upsell_propensity, model_version, predicted_at
            FROM predictions
            WHERE salon_id = %s
            """,
            (salon_id,),
        )
        pred_rows = cur.fetchall()

    if not pred_rows:
        raise HTTPException(status_code=404, detail=f"No predictions found for salon {salon_id}")

    preds = []
    for r in pred_rows:
        rf = r["risk_factors"]
        if isinstance(rf, str):
            rf = json.loads(rf)
        preds.append({
            "customer_id": str(r["customer_id"]),
            "salon_id": str(r["salon_id"]),
            "churn_probability": float(r["churn_probability"]),
            "risk_level": r["risk_level"],
            "risk_factors": rf,
            "predicted_next_visit_days": int(r["predicted_next_visit_days"]),
            "estimated_ltv_12m": float(r["estimated_ltv_12m"]),
            "upsell_propensity": float(r["upsell_propensity"]),
            "model_version": r["model_version"],
            "predicted_at": str(r["predicted_at"]),
        })

    dashboard = _compute_dashboard_for_salon(conn, salon_id, preds)

    # Cache for subsequent requests
    rd.set(
        f"dashboard:{salon_id}",
        json.dumps(dashboard, default=str),
        ex=DASHBOARD_CACHE_TTL_SECONDS,
    )

    return DashboardResponse(**dashboard).model_dump()


@router.get("/v1/salon/{salon_id}/at-risk")
async def at_risk_customers(
    salon_id: str,
    limit: int = Query(default=50, ge=1, le=500),
    min_risk: str = Query(default="MEDIUM"),
):
    """List customers at risk of churning, ordered by churn probability."""
    risk_filters = {
        "LOW": ("LOW", "MEDIUM", "HIGH", "CRITICAL"),
        "MEDIUM": ("MEDIUM", "HIGH", "CRITICAL"),
        "HIGH": ("HIGH", "CRITICAL"),
        "CRITICAL": ("CRITICAL",),
    }

    min_risk_upper = min_risk.upper()
    if min_risk_upper not in risk_filters:
        raise HTTPException(status_code=400, detail=f"Invalid min_risk value: {min_risk}")

    allowed_levels = risk_filters[min_risk_upper]

    conn = _get_pg_connection()

    # Verify salon exists
    with conn.cursor() as cur:
        cur.execute(_SALON_EXISTS_QUERY, (salon_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail=f"Salon {salon_id} not found")

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT p.customer_id, c.customer_name, p.churn_probability,
                   p.risk_level, p.risk_factors, p.estimated_ltv_12m,
                   cf.days_since_last_visit, cf.avg_inter_visit_days
            FROM predictions p
            JOIN customers c ON p.customer_id = c.customer_id AND p.salon_id = c.salon_id
            JOIN customer_features cf ON p.customer_id = cf.customer_id AND p.salon_id = cf.salon_id
            WHERE p.salon_id = %s AND p.risk_level IN %s
            ORDER BY p.churn_probability DESC
            LIMIT %s
            """,
            (salon_id, allowed_levels, limit),
        )
        rows = cur.fetchall()

    # Total count (without limit)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) FROM predictions
            WHERE salon_id = %s AND risk_level IN %s
            """,
            (salon_id, allowed_levels),
        )
        total_count = cur.fetchone()[0]

    results = []
    for r in rows:
        rf = r["risk_factors"]
        if isinstance(rf, str):
            rf = json.loads(rf)

        primary = rf[0]["description"] if rf else "Within normal parameters"
        risk_level = r["risk_level"]

        results.append(
            AtRiskCustomer(
                customer_id=str(r["customer_id"]),
                customer_name=r["customer_name"],
                churn_probability=round(float(r["churn_probability"]), 4),
                risk_level=risk_level,
                days_since_last_visit=int(r["days_since_last_visit"] or 0),
                avg_inter_visit_days=round(float(r["avg_inter_visit_days"] or 0), 2),
                primary_risk_factor=primary,
                suggested_action=SUGGESTED_ACTIONS.get(risk_level, "MONITOR"),
                estimated_ltv_12m=round(float(r["estimated_ltv_12m"]), 2),
            ).model_dump()
        )

    return {
        "salon_id": salon_id,
        "at_risk_customers": results,
        "total_count": total_count,
    }


@router.get("/v1/salon/{salon_id}/customers")
async def list_customers(
    salon_id: str,
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
    search: str = Query(default=""),
    risk_level: str = Query(default=""),
):
    """Paginated customer list with prediction data."""
    conn = _get_pg_connection()

    # Verify salon exists
    with conn.cursor() as cur:
        cur.execute(_SALON_EXISTS_QUERY, (salon_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail=f"Salon {salon_id} not found")

    offset = (page - 1) * per_page
    conditions = ["c.salon_id = %s"]
    params: list = [salon_id]

    if search:
        conditions.append("c.customer_name ILIKE %s")
        params.append(f"%{search}%")

    if risk_level:
        conditions.append("p.risk_level = %s")
        params.append(risk_level.upper())

    where_clause = " AND ".join(conditions)

    # Total count
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT COUNT(*)
            FROM customers c
            JOIN predictions p ON c.customer_id = p.customer_id AND c.salon_id = p.salon_id
            WHERE {where_clause}
            """,
            params,
        )
        total_count = cur.fetchone()[0]

    # Paginated results
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            f"""
            SELECT c.customer_id, c.salon_id, c.customer_name,
                   p.churn_probability, p.risk_level, p.risk_factors,
                   p.predicted_next_visit_days, p.estimated_ltv_12m,
                   p.upsell_propensity, p.model_version, p.predicted_at
            FROM customers c
            JOIN customer_features cf ON c.customer_id = cf.customer_id AND c.salon_id = cf.salon_id
            JOIN predictions p ON c.customer_id = p.customer_id AND c.salon_id = p.salon_id
            WHERE {where_clause}
            ORDER BY c.customer_name
            LIMIT %s OFFSET %s
            """,
            params + [per_page, offset],
        )
        rows = cur.fetchall()

    customers = []
    for r in rows:
        rf = r["risk_factors"]
        if isinstance(rf, str):
            rf = json.loads(rf)

        customers.append(
            CustomerPrediction(
                customer_id=str(r["customer_id"]),
                salon_id=str(r["salon_id"]),
                customer_name=r["customer_name"],
                churn_probability_90d=round(float(r["churn_probability"]), 4),
                risk_level=r["risk_level"],
                risk_factors=[RiskFactor(**f) for f in rf],
                predicted_next_visit_days=int(r["predicted_next_visit_days"]),
                estimated_ltv_12m=round(float(r["estimated_ltv_12m"]), 2),
                upsell_propensity=round(float(r["upsell_propensity"]), 4),
                model_version=r["model_version"],
                predicted_at=str(r["predicted_at"]),
            ).model_dump()
        )

    return {
        "customers": customers,
        "total_count": total_count,
        "page": page,
        "per_page": per_page,
    }


@router.get("/v1/predictions/churn/{customer_id}")
async def customer_prediction(
    customer_id: str,
    salon_id: str = Query(...),
):
    """Detailed prediction for a single customer, including visit history."""
    conn = _get_pg_connection()

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT c.customer_id, c.salon_id, c.customer_name,
                   p.churn_probability, p.risk_level, p.risk_factors,
                   p.predicted_next_visit_days, p.estimated_ltv_12m,
                   p.upsell_propensity, p.model_version, p.predicted_at
            FROM customers c
            JOIN predictions p ON c.customer_id = p.customer_id AND c.salon_id = p.salon_id
            WHERE c.customer_id = %s AND c.salon_id = %s
            """,
            (customer_id, salon_id),
        )
        row = cur.fetchone()

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"Customer {customer_id} not found in salon {salon_id}",
        )

    rf = row["risk_factors"]
    if isinstance(rf, str):
        rf = json.loads(rf)

    prediction = CustomerPrediction(
        customer_id=str(row["customer_id"]),
        salon_id=str(row["salon_id"]),
        customer_name=row["customer_name"],
        churn_probability_90d=round(float(row["churn_probability"]), 4),
        risk_level=row["risk_level"],
        risk_factors=[RiskFactor(**f) for f in rf],
        predicted_next_visit_days=int(row["predicted_next_visit_days"]),
        estimated_ltv_12m=round(float(row["estimated_ltv_12m"]), 2),
        upsell_propensity=round(float(row["upsell_propensity"]), 4),
        model_version=row["model_version"],
        predicted_at=str(row["predicted_at"]),
    )

    # Fetch last 10 completed visits with services
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT v.visit_date, v.total_amount, st.stylist_name,
                   STRING_AGG(s.service_name, ', ') AS services
            FROM visits v
            LEFT JOIN stylists st ON v.stylist_id = st.stylist_id
            LEFT JOIN visit_services vs ON v.visit_id = vs.visit_id
            LEFT JOIN services s ON vs.service_id = s.service_id
            WHERE v.customer_id = %s AND v.salon_id = %s AND v.status = 'COMPLETED'
            GROUP BY v.visit_id, v.visit_date, v.total_amount, st.stylist_name
            ORDER BY v.visit_date DESC
            LIMIT 10
            """,
            (customer_id, salon_id),
        )
        visits = cur.fetchall()

    visit_history = [
        VisitRecord(
            visit_date=str(v["visit_date"]),
            services=v["services"] or "",
            amount=round(float(v["total_amount"] or 0), 2),
            stylist=v["stylist_name"] or "",
        ).model_dump()
        for v in visits
    ]

    result = prediction.model_dump()
    result["visit_history"] = visit_history
    return result


@router.get("/v1/salon/{salon_id}/case-study")
async def salon_case_study(salon_id: str):
    """Before/after case study metrics for a salon."""
    conn = _get_pg_connection()

    # Verify salon exists and get name
    with conn.cursor() as cur:
        cur.execute("SELECT salon_name FROM salons WHERE salon_id = %s", (salon_id,))
        salon_row = cur.fetchone()
    if not salon_row:
        raise HTTPException(status_code=404, detail=f"Salon {salon_id} not found")

    salon_name = salon_row[0]

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT period, metric_name, metric_value
            FROM case_study_metrics
            WHERE salon_id = %s
            ORDER BY period, metric_name
            """,
            (salon_id,),
        )
        rows = cur.fetchall()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No case study data for salon {salon_id}",
        )

    before: dict[str, float] = {}
    after: dict[str, float] = {}

    for r in rows:
        period = r["period"].upper()
        metric_name = r["metric_name"]
        metric_value = float(r["metric_value"])
        if period == "BEFORE":
            before[metric_name] = round(metric_value, 2)
        elif period == "AFTER":
            after[metric_name] = round(metric_value, 2)

    improvements = []
    for metric_name in before:
        if metric_name in after:
            before_val = before[metric_name]
            after_val = after[metric_name]
            change_pct = (
                round((after_val - before_val) / before_val * 100, 2)
                if before_val != 0
                else 0.0
            )
            improvements.append({
                "metric": metric_name,
                "before": before_val,
                "after": after_val,
                "change_percent": change_pct,
            })

    return CaseStudyResponse(
        salon_id=salon_id,
        salon_name=salon_name,
        before=before,
        after=after,
        improvements=improvements,
    ).model_dump()


# ---------------------------------------------------------------------------
# Helper -- salon existence check
# ---------------------------------------------------------------------------

def _verify_salon_exists(conn, salon_id: str) -> None:
    """Raise 404 if the given salon_id does not exist in the salons table."""
    with conn.cursor() as cur:
        cur.execute(_SALON_EXISTS_QUERY, (salon_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail=f"Salon {salon_id} not found")


# ---------------------------------------------------------------------------
# Analytics Endpoints
# ---------------------------------------------------------------------------

@router.get("/v1/analytics/revenue/{salon_id}")
async def analytics_revenue(salon_id: str):
    """Monthly revenue breakdown for the last 12 months."""
    logger.info("Revenue analytics requested for salon_id=%s", salon_id)

    conn = _get_pg_connection()
    rd = _get_redis_client()

    # Check Redis cache
    cache_key = f"analytics:revenue:{salon_id}"
    cached = rd.get(cache_key)
    if cached:
        logger.debug("Revenue analytics cache hit for salon_id=%s", salon_id)
        return json.loads(cached)

    _verify_salon_exists(conn, salon_id)

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT DATE_TRUNC('month', visit_date) AS month,
                   SUM(total_amount) AS revenue,
                   COUNT(*) AS visit_count,
                   AVG(total_amount) AS avg_ticket
            FROM visits
            WHERE salon_id = %s AND status = 'COMPLETED'
              AND visit_date >= NOW() - INTERVAL '12 months'
            GROUP BY DATE_TRUNC('month', visit_date)
            ORDER BY month
            """,
            (salon_id,),
        )
        rows = cur.fetchall()

    months = []
    total_revenue = 0.0
    total_visits = 0
    for r in rows:
        revenue = round(float(r["revenue"] or 0), 2)
        visit_count = int(r["visit_count"] or 0)
        avg_ticket = round(float(r["avg_ticket"] or 0), 2)
        month_str = r["month"].strftime("%Y-%m") if r["month"] else "unknown"

        months.append({
            "month": month_str,
            "revenue": revenue,
            "visit_count": visit_count,
            "avg_ticket": avg_ticket,
        })
        total_revenue += revenue
        total_visits += visit_count

    overall_avg_ticket = round(total_revenue / total_visits, 2) if total_visits > 0 else 0.0

    response = {
        "salon_id": salon_id,
        "months": months,
        "totals": {
            "total_revenue": round(total_revenue, 2),
            "total_visits": total_visits,
            "avg_ticket": overall_avg_ticket,
        },
    }

    rd.set(cache_key, json.dumps(response, default=str), ex=ANALYTICS_CACHE_TTL_SECONDS)
    logger.info("Revenue analytics computed for salon_id=%s, months=%d", salon_id, len(months))
    return response


@router.get("/v1/analytics/segments/{salon_id}")
async def analytics_segments(salon_id: str):
    """Customer segmentation derived from behavioral features and predictions."""
    logger.info("Segment analytics requested for salon_id=%s", salon_id)

    conn = _get_pg_connection()
    rd = _get_redis_client()

    cache_key = f"analytics:segments:{salon_id}"
    cached = rd.get(cache_key)
    if cached:
        logger.debug("Segment analytics cache hit for salon_id=%s", salon_id)
        return json.loads(cached)

    _verify_salon_exists(conn, salon_id)

    # Fetch features joined with predictions for segmentation logic
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT cf.customer_id,
                   cf.visit_count_total,
                   cf.overdue_ratio,
                   cf.visit_regularity_index,
                   cf.inter_visit_trend,
                   cf.avg_ticket_value,
                   p.risk_level
            FROM customer_features cf
            LEFT JOIN predictions p
              ON cf.customer_id = p.customer_id AND cf.salon_id = p.salon_id
            WHERE cf.salon_id = %s
            """,
            (salon_id,),
        )
        rows = cur.fetchall()

    # Classify each customer into exactly one segment with priority ordering
    segment_data: dict[str, list[float]] = {
        "Loyal Regular": [],
        "At Risk": [],
        "New": [],
        "Declining": [],
        "Seasonal": [],
        "Sporadic": [],
    }

    for r in rows:
        visit_count_total = int(r.get("visit_count_total") or 0)
        overdue_ratio = float(r.get("overdue_ratio") or 0)
        visit_regularity_index = float(r.get("visit_regularity_index") or 0)
        inter_visit_trend = float(r.get("inter_visit_trend") or 0)
        avg_ticket = float(r.get("avg_ticket_value") or 0)
        risk_level = r.get("risk_level") or "LOW"

        # Priority-based classification: first match wins
        if risk_level in ("HIGH", "CRITICAL"):
            segment_data["At Risk"].append(avg_ticket)
        elif visit_count_total >= 10 and overdue_ratio < 1.2 and visit_regularity_index > 0.5:
            segment_data["Loyal Regular"].append(avg_ticket)
        elif visit_count_total <= 3:
            segment_data["New"].append(avg_ticket)
        elif inter_visit_trend > 1.0 and visit_count_total > 5:
            segment_data["Declining"].append(avg_ticket)
        elif visit_regularity_index < 0.3 and visit_count_total >= 5:
            segment_data["Seasonal"].append(avg_ticket)
        else:
            segment_data["Sporadic"].append(avg_ticket)

    total_customers = len(rows)
    segments = []
    for name in ("Loyal Regular", "At Risk", "New", "Declining", "Seasonal", "Sporadic"):
        tickets = segment_data[name]
        count = len(tickets)
        percentage = round((count / total_customers * 100), 1) if total_customers > 0 else 0.0
        avg_ticket_value = round(sum(tickets) / count, 2) if count > 0 else 0.0
        segments.append({
            "name": name,
            "count": count,
            "percentage": percentage,
            "avg_ticket_value": avg_ticket_value,
            "color": SEGMENT_COLORS[name],
        })

    response = {
        "salon_id": salon_id,
        "total_customers": total_customers,
        "segments": segments,
    }

    rd.set(cache_key, json.dumps(response, default=str), ex=ANALYTICS_CACHE_TTL_SECONDS)
    logger.info(
        "Segment analytics computed for salon_id=%s, total_customers=%d",
        salon_id, total_customers,
    )
    return response


@router.get("/v1/analytics/services/{salon_id}")
async def analytics_services(salon_id: str):
    """Service category breakdown and top-performing services for a salon."""
    logger.info("Service analytics requested for salon_id=%s", salon_id)

    conn = _get_pg_connection()
    rd = _get_redis_client()

    cache_key = f"analytics:services:{salon_id}"
    cached = rd.get(cache_key)
    if cached:
        logger.debug("Service analytics cache hit for salon_id=%s", salon_id)
        return json.loads(cached)

    _verify_salon_exists(conn, salon_id)

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT s.category, s.service_name,
                   SUM(vs.price) AS revenue,
                   COUNT(*) AS service_count
            FROM visit_services vs
            JOIN visits v ON vs.visit_id = v.visit_id
            JOIN services s ON vs.service_id = s.service_id
            WHERE v.salon_id = %s AND v.status = 'COMPLETED'
            GROUP BY s.category, s.service_name
            ORDER BY revenue DESC
            """,
            (salon_id,),
        )
        rows = cur.fetchall()

    # Aggregate by category
    category_totals: dict[str, dict] = {}
    top_services = []
    grand_total_revenue = 0.0

    for r in rows:
        category = r["category"] or "UNKNOWN"
        revenue = round(float(r["revenue"] or 0), 2)
        count = int(r["service_count"] or 0)
        service_name = r["service_name"]

        if category not in category_totals:
            category_totals[category] = {"revenue": 0.0, "count": 0}
        category_totals[category]["revenue"] += revenue
        category_totals[category]["count"] += count
        grand_total_revenue += revenue

        top_services.append({
            "service_name": service_name,
            "category": category,
            "revenue": revenue,
            "count": count,
        })

    categories = []
    for cat, data in sorted(category_totals.items(), key=lambda x: x[1]["revenue"], reverse=True):
        cat_revenue = round(data["revenue"], 2)
        percentage = round((cat_revenue / grand_total_revenue * 100), 1) if grand_total_revenue > 0 else 0.0
        categories.append({
            "category": cat,
            "revenue": cat_revenue,
            "count": data["count"],
            "percentage": percentage,
        })

    response = {
        "salon_id": salon_id,
        "categories": categories,
        "top_services": top_services,
    }

    rd.set(cache_key, json.dumps(response, default=str), ex=ANALYTICS_CACHE_TTL_SECONDS)
    logger.info(
        "Service analytics computed for salon_id=%s, categories=%d, services=%d",
        salon_id, len(categories), len(top_services),
    )
    return response


# ---------------------------------------------------------------------------
# Stylist Performance Endpoint
# ---------------------------------------------------------------------------

@router.get("/v1/stylists/{salon_id}")
async def stylist_performance(salon_id: str):
    """Stylist performance metrics including revenue, visit counts, and retention."""
    logger.info("Stylist performance requested for salon_id=%s", salon_id)

    conn = _get_pg_connection()
    rd = _get_redis_client()

    cache_key = f"stylists:performance:{salon_id}"
    cached = rd.get(cache_key)
    if cached:
        logger.debug("Stylist performance cache hit for salon_id=%s", salon_id)
        return json.loads(cached)

    _verify_salon_exists(conn, salon_id)

    # Core stylist metrics
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT st.stylist_id, st.stylist_name, st.specialty, st.experience_years,
                   COUNT(DISTINCT v.customer_id) AS customer_count,
                   COUNT(v.visit_id) AS visit_count,
                   COALESCE(SUM(v.total_amount), 0) AS total_revenue,
                   COALESCE(AVG(v.total_amount), 0) AS avg_ticket,
                   COALESCE(AVG(v.duration_minutes), 0) AS avg_duration
            FROM stylists st
            LEFT JOIN visits v ON st.stylist_id = v.stylist_id AND v.status = 'COMPLETED'
            WHERE st.salon_id = %s AND st.is_active = TRUE
            GROUP BY st.stylist_id, st.stylist_name, st.specialty, st.experience_years
            ORDER BY total_revenue DESC
            """,
            (salon_id,),
        )
        stylist_rows = cur.fetchall()

    # Retention rate per stylist: customers NOT in HIGH/CRITICAL risk / total customers
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT v.stylist_id,
                   COUNT(DISTINCT v.customer_id) AS total_customers,
                   COUNT(DISTINCT v.customer_id)
                       FILTER (WHERE p.risk_level NOT IN ('HIGH', 'CRITICAL'))
                       AS retained_customers
            FROM visits v
            JOIN predictions p ON v.customer_id = p.customer_id AND v.salon_id = p.salon_id
            WHERE v.salon_id = %s AND v.status = 'COMPLETED'
            GROUP BY v.stylist_id
            """,
            (salon_id,),
        )
        retention_rows = cur.fetchall()

    retention_map: dict[str, float] = {}
    for rr in retention_rows:
        total_cust = int(rr["total_customers"] or 0)
        retained = int(rr["retained_customers"] or 0)
        rate = round((retained / total_cust * 100), 1) if total_cust > 0 else 0.0
        retention_map[str(rr["stylist_id"])] = rate

    stylists = []
    total_revenue_all = 0.0
    top_performer_name = ""
    top_performer_revenue = 0.0

    for sr in stylist_rows:
        stylist_id = str(sr["stylist_id"])
        total_revenue = round(float(sr["total_revenue"] or 0), 2)
        avg_ticket = round(float(sr["avg_ticket"] or 0), 2)
        avg_duration = round(float(sr["avg_duration"] or 0), 0)
        retention_rate = retention_map.get(stylist_id, 0.0)

        total_revenue_all += total_revenue
        if total_revenue > top_performer_revenue:
            top_performer_revenue = total_revenue
            top_performer_name = sr["stylist_name"]

        stylists.append({
            "stylist_id": stylist_id,
            "stylist_name": sr["stylist_name"],
            "specialty": sr["specialty"],
            "experience_years": int(sr["experience_years"] or 0),
            "customer_count": int(sr["customer_count"] or 0),
            "visit_count": int(sr["visit_count"] or 0),
            "total_revenue": total_revenue,
            "avg_ticket": avg_ticket,
            "avg_duration": int(avg_duration),
            "retention_rate": retention_rate,
        })

    total_stylists = len(stylists)
    avg_revenue_per_stylist = round(
        total_revenue_all / total_stylists, 2
    ) if total_stylists > 0 else 0.0

    response = {
        "salon_id": salon_id,
        "stylists": stylists,
        "summary": {
            "total_stylists": total_stylists,
            "avg_revenue_per_stylist": avg_revenue_per_stylist,
            "top_performer": top_performer_name if top_performer_name else "N/A",
        },
    }

    rd.set(cache_key, json.dumps(response, default=str), ex=ANALYTICS_CACHE_TTL_SECONDS)
    logger.info(
        "Stylist performance computed for salon_id=%s, stylists=%d",
        salon_id, total_stylists,
    )
    return response


# ---------------------------------------------------------------------------
# System Monitoring Endpoint
# ---------------------------------------------------------------------------

@router.get("/v1/system/status")
async def system_status():
    """Comprehensive system health and operational status."""
    logger.info("System status check requested.")

    status_response: dict = {
        "status": "operational",
        "uptime": "healthy",
        "services": {},
        "models": {},
        "data": {},
        "prediction_distribution": {},
    }

    # -- Database health -------------------------------------------------------
    db_status = {"status": "disconnected", "latency_ms": -1}
    try:
        conn = _get_pg_connection()
        start_ns = time.monotonic_ns()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        latency_ms = round((time.monotonic_ns() - start_ns) / 1_000_000, 1)
        db_status = {"status": "connected", "latency_ms": latency_ms}
    except Exception as exc:
        logger.error("Database health check failed: %s", exc)
        db_status = {"status": "error", "latency_ms": -1}
        status_response["status"] = "degraded"

    status_response["services"]["database"] = db_status

    # -- Redis health ----------------------------------------------------------
    redis_status = {"status": "disconnected", "keys": 0}
    try:
        rd = _get_redis_client()
        rd.ping()
        redis_keys = rd.dbsize()
        redis_status = {"status": "connected", "keys": redis_keys}
    except Exception as exc:
        logger.error("Redis health check failed: %s", exc)
        redis_status = {"status": "error", "keys": 0}
        status_response["status"] = "degraded"

    status_response["services"]["redis"] = redis_status

    # -- Feature store and prediction engine -----------------------------------
    try:
        conn = _get_pg_connection()

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM customer_features")
            feature_count = cur.fetchone()[0]

        with conn.cursor() as cur:
            cur.execute("SELECT MAX(computed_at) FROM customer_features")
            last_feature_update = cur.fetchone()[0]

        status_response["services"]["feature_store"] = {
            "status": "operational",
            "features_computed": feature_count,
            "last_updated": str(last_feature_update) if last_feature_update else None,
        }

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM predictions")
            prediction_count = cur.fetchone()[0]

        with conn.cursor() as cur:
            cur.execute("SELECT MAX(predicted_at) FROM predictions")
            last_prediction = cur.fetchone()[0]

        status_response["services"]["prediction_engine"] = {
            "status": "operational",
            "predictions_stored": prediction_count,
            "last_run": str(last_prediction) if last_prediction else None,
        }

        # -- Data counts -------------------------------------------------------
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM customers")
            customer_count = cur.fetchone()[0]

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM visits")
            visit_count = cur.fetchone()[0]

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(DISTINCT salon_id) FROM salons")
            salon_count = cur.fetchone()[0]

        status_response["data"] = {
            "total_customers": customer_count,
            "total_visits": visit_count,
            "total_salons": salon_count,
        }

        # -- Prediction distribution -------------------------------------------
        with conn.cursor() as cur:
            cur.execute("SELECT risk_level, COUNT(*) FROM predictions GROUP BY risk_level")
            pred_dist = dict(cur.fetchall())

        status_response["prediction_distribution"] = pred_dist

    except Exception as exc:
        logger.error("Database queries for system status failed: %s", exc)
        status_response["services"]["feature_store"] = {"status": "error"}
        status_response["services"]["prediction_engine"] = {"status": "error"}
        status_response["status"] = "degraded"

    # -- Model metadata --------------------------------------------------------
    try:
        metadata_path = os.path.join(settings.MODEL_PATH, "model_metadata.json")
        with open(metadata_path, "r") as fh:
            model_meta = json.load(fh)

        models_info = {}
        for model_key in ("churn", "nextvisit", "ltv", "upsell"):
            meta_entry = model_meta.get("models", {}).get(model_key, {})
            models_info[model_key] = {
                "type": meta_entry.get("type", "unknown"),
                "version": meta_entry.get("version", model_meta.get("version", "unknown")),
                "trained_at": meta_entry.get("trained_at", "unknown"),
                "metrics": meta_entry.get("metrics", {}),
            }

        status_response["models"] = models_info

    except FileNotFoundError:
        logger.warning("Model metadata file not found at %s", metadata_path)
        status_response["models"] = {"error": "metadata file not found"}
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse model metadata: %s", exc)
        status_response["models"] = {"error": "invalid metadata format"}

    logger.info("System status check completed, status=%s", status_response["status"])
    return status_response


# ---------------------------------------------------------------------------
# Business Insights Endpoint
# ---------------------------------------------------------------------------

INSIGHTS_CACHE_TTL_SECONDS = 300
NETWORK_AVG_TICKET_JPY = 10500


def _safe_pct_change(current: float, previous: float) -> float:
    """Calculate percentage change, returning 0.0 when previous is zero."""
    if previous == 0:
        return 0.0
    return round((current - previous) / previous * 100, 1)


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Calculate a ratio, returning 0.0 when denominator is zero."""
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 2)


def _compute_retention_trend(conn, salon_id: str) -> list[dict]:
    """Compute monthly retention rates for the last 6 months.

    Retention for a given month is defined as the percentage of customers
    who visited in the prior 3 months and also visited in the target month.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            WITH monthly_visitors AS (
                SELECT DISTINCT customer_id,
                       DATE_TRUNC('month', visit_date) AS visit_month
                FROM visits
                WHERE salon_id = %s AND status = 'COMPLETED'
                  AND visit_date >= NOW() - INTERVAL '9 months'
            )
            SELECT visit_month, COUNT(DISTINCT customer_id) AS visitor_count
            FROM monthly_visitors
            GROUP BY visit_month
            ORDER BY visit_month
            """,
            (salon_id,),
        )
        rows = cur.fetchall()

    if not rows:
        return []

    # Build a mapping: month -> set of customer_ids
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT DISTINCT customer_id,
                   DATE_TRUNC('month', visit_date) AS visit_month
            FROM visits
            WHERE salon_id = %s AND status = 'COMPLETED'
              AND visit_date >= NOW() - INTERVAL '9 months'
            """,
            (salon_id,),
        )
        detail_rows = cur.fetchall()

    month_customers: dict[str, set] = {}
    for dr in detail_rows:
        month_key = dr["visit_month"].strftime("%Y-%m")
        month_customers.setdefault(month_key, set()).add(str(dr["customer_id"]))

    sorted_months = sorted(month_customers.keys())
    trend = []

    for idx, month_key in enumerate(sorted_months):
        # Need at least one prior month to compute retention
        if idx == 0:
            continue

        prev_month_key = sorted_months[idx - 1]
        prev_customers = month_customers.get(prev_month_key, set())
        current_customers = month_customers.get(month_key, set())

        if len(prev_customers) == 0:
            rate = 0.0
        else:
            retained = prev_customers.intersection(current_customers)
            rate = round(len(retained) / len(prev_customers) * 100, 1)

        trend.append({
            "month": month_key,
            "rate": rate,
            "visitors": len(current_customers),
        })

    # Return only the last 6 months of trend data
    return trend[-6:]


def _insight_risk(at_risk_count: int, total_customers: int) -> str | None:
    """Return a risk-level insight if at-risk percentage exceeds threshold."""
    at_risk_pct = _safe_ratio(at_risk_count, total_customers) * 100
    if at_risk_pct > 20:
        return (
            f"{at_risk_count} customers ({at_risk_pct:.1f}%) are at critical risk"
            " -- immediate outreach recommended for high-LTV clients"
        )
    return None


def _insight_revenue_trend(mom_revenue_change: float) -> str | None:
    """Return a revenue-trend insight based on month-over-month change."""
    if mom_revenue_change < -5:
        return (
            f"Revenue decreased {abs(mom_revenue_change):.1f}% month-over-month"
            " -- review seasonal factors and scheduling capacity"
        )
    if mom_revenue_change > 5:
        return (
            f"Revenue increased {mom_revenue_change:.1f}% month-over-month"
            " -- strong customer engagement trend"
        )
    return None


def _insight_walkin(walkin_ratio: float) -> str | None:
    """Return a walk-in ratio insight when outside the normal band."""
    if walkin_ratio > 0.2:
        return (
            f"Walk-in rate is {walkin_ratio * 100:.1f}%"
            " -- consider online booking promotions to increase appointment ratio"
        )
    if walkin_ratio < 0.1:
        return (
            f"Strong appointment booking rate at {(1 - walkin_ratio) * 100:.1f}%"
            " -- indicates effective scheduling practices"
        )
    return None


def _insight_top_stylist(top_stylist_data: dict) -> str | None:
    """Return an insight highlighting the top-performing stylist."""
    if top_stylist_data and top_stylist_data.get("name"):
        return (
            f"Top stylist {top_stylist_data['name']} generated"
            f" {top_stylist_data.get('revenue_share_pct', 0):.1f}% of monthly revenue"
            f" with {top_stylist_data.get('customer_count', 0)} active clients"
        )
    return None


def _insight_noshow(noshow_rate: float) -> str | None:
    """Return a no-show rate insight when outside the acceptable band."""
    if noshow_rate > 0.10:
        return (
            f"No-show rate of {noshow_rate * 100:.1f}% is above target"
            " -- consider automated reminder messaging"
        )
    if noshow_rate < 0.05:
        return (
            f"No-show rate of {noshow_rate * 100:.1f}% is well below industry average"
            " -- effective reminder system in place"
        )
    return None


def _insight_ticket_value(avg_ticket: float) -> str:
    """Return a ticket-value benchmark insight (always produced)."""
    position = "above" if avg_ticket >= NETWORK_AVG_TICKET_JPY else "below"
    return (
        f"Average ticket value of {avg_ticket:.0f} JPY is {position}"
        f" network average for standard tier salons"
    )


def _insight_new_customers(new_customer_count: int) -> str | None:
    """Return a new-customer acquisition insight when count is notable."""
    if new_customer_count > 10:
        return (
            f"{new_customer_count} new customers acquired this month"
            " -- focus on second-visit conversion for retention"
        )
    return None


def _generate_business_insights(
    mom_revenue_change: float,
    at_risk_count: int,
    total_customers: int,
    walkin_ratio: float,
    top_stylist_data: dict,
    noshow_rate: float,
    avg_ticket: float,
    new_customer_count: int,
) -> list[str]:
    """Generate contextual business insight strings from computed metrics.

    Each insight category is handled by a dedicated helper to keep
    cyclomatic complexity within acceptable bounds.
    """
    candidates = [
        _insight_risk(at_risk_count, total_customers),
        _insight_revenue_trend(mom_revenue_change),
        _insight_walkin(walkin_ratio),
        _insight_top_stylist(top_stylist_data),
        _insight_noshow(noshow_rate),
        _insight_ticket_value(avg_ticket),
        _insight_new_customers(new_customer_count),
    ]
    return [insight for insight in candidates if insight is not None]


@router.get("/v1/salon/{salon_id}/insights")
async def salon_insights(salon_id: str):
    """Comprehensive business intelligence for the salon dashboard.

    Aggregates revenue trends, visit analytics, action items, stylist
    performance, service mix, product sales, retention trends, and
    generates contextual business insights from the computed data.
    """
    logger.info("Business insights requested for salon_id=%s", salon_id)

    rd = _get_redis_client()

    # Check Redis cache first
    cache_key = f"insights:{salon_id}"
    cached = rd.get(cache_key)
    if cached:
        logger.debug("Business insights cache hit for salon_id=%s", salon_id)
        return json.loads(cached)

    conn = _get_pg_connection()
    _verify_salon_exists(conn, salon_id)

    # ------------------------------------------------------------------
    # 1. Revenue comparison: last 30 days vs previous 30 days
    # ------------------------------------------------------------------
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT COALESCE(SUM(total_amount), 0) AS revenue,
                   COUNT(*) AS visit_count,
                   COALESCE(AVG(total_amount), 0) AS avg_ticket
            FROM visits
            WHERE salon_id = %s AND status = 'COMPLETED'
              AND visit_date >= NOW() - INTERVAL '30 days'
            """,
            (salon_id,),
        )
        rev_current = cur.fetchone()

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT COALESCE(SUM(total_amount), 0) AS revenue,
                   COUNT(*) AS visit_count
            FROM visits
            WHERE salon_id = %s AND status = 'COMPLETED'
              AND visit_date >= NOW() - INTERVAL '60 days'
              AND visit_date < NOW() - INTERVAL '30 days'
            """,
            (salon_id,),
        )
        rev_previous = cur.fetchone()

    revenue_30d = round(float(rev_current["revenue"]), 2)
    visit_count_30d = int(rev_current["visit_count"])
    avg_ticket = round(float(rev_current["avg_ticket"]), 1)
    revenue_prev_30d = round(float(rev_previous["revenue"]), 2)
    visit_count_prev_30d = int(rev_previous["visit_count"])

    mom_revenue_change = _safe_pct_change(revenue_30d, revenue_prev_30d)
    mom_visit_change = _safe_pct_change(visit_count_30d, visit_count_prev_30d)
    avg_daily_revenue = round(revenue_30d / 30, 1)

    # ------------------------------------------------------------------
    # 2. Visit type breakdown (last 90 days)
    # ------------------------------------------------------------------
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT visit_type, COUNT(*) AS cnt
            FROM visits
            WHERE salon_id = %s AND status = 'COMPLETED'
              AND visit_date >= NOW() - INTERVAL '90 days'
            GROUP BY visit_type
            """,
            (salon_id,),
        )
        visit_type_rows = cur.fetchall()

    walkin_count = 0
    appointment_count = 0
    for vt in visit_type_rows:
        vtype = (vt["visit_type"] or "").upper()
        cnt = int(vt["cnt"])
        if vtype == "WALKIN":
            walkin_count += cnt
        else:
            appointment_count += cnt

    total_typed_visits = walkin_count + appointment_count
    walkin_ratio = _safe_ratio(walkin_count, total_typed_visits)

    # ------------------------------------------------------------------
    # 3. No-show rate (last 90 days)
    # ------------------------------------------------------------------
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
              COUNT(*) FILTER (WHERE status = 'NOSHOW') AS noshow_count,
              COUNT(*) AS total_bookings
            FROM bookings
            WHERE salon_id = %s
              AND appointment_date >= NOW() - INTERVAL '90 days'
            """,
            (salon_id,),
        )
        noshow_row = cur.fetchone()

    noshow_count = int(noshow_row["noshow_count"])
    total_bookings = int(noshow_row["total_bookings"])
    noshow_rate = _safe_ratio(noshow_count, total_bookings)

    # ------------------------------------------------------------------
    # 4. Action items from predictions (distributed by recency/severity)
    # ------------------------------------------------------------------
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT p.risk_level, cf.days_since_last_visit, cf.avg_inter_visit_days
            FROM predictions p
            JOIN customer_features cf ON p.salon_id = cf.salon_id
                                      AND p.customer_id = cf.customer_id
            WHERE p.salon_id = %s
              AND p.risk_level IN ('MEDIUM', 'HIGH', 'CRITICAL')
            """,
            (salon_id,),
        )
        action_rows = cur.fetchall()

    followup_count = 0
    booking_count = 0
    offer_count = 0
    for ar in action_rows:
        days = int(ar.get("days_since_last_visit") or 0)
        avg_interval = float(ar.get("avg_inter_visit_days") or 30)
        overdue_ratio = days / avg_interval if avg_interval > 0 else 0
        if ar["risk_level"] == "MEDIUM":
            booking_count += 1
        elif ar["risk_level"] == "HIGH":
            followup_count += 1
        elif overdue_ratio < 2.0:
            followup_count += 1
        elif overdue_ratio < 3.5:
            booking_count += 1
        else:
            offer_count += 1

    action_total = followup_count + booking_count + offer_count
    action_items = {
        "send_followup": followup_count,
        "book_appointment": booking_count,
        "offer_promotion": offer_count,
        "total": action_total,
    }

    # ------------------------------------------------------------------
    # 5. Top stylist (last 30 days)
    # ------------------------------------------------------------------
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT st.stylist_name,
                   COALESCE(SUM(v.total_amount), 0) AS revenue,
                   COUNT(DISTINCT v.customer_id) AS customer_count
            FROM visits v
            JOIN stylists st ON v.stylist_id = st.stylist_id
            WHERE v.salon_id = %s AND v.status = 'COMPLETED'
              AND v.visit_date >= NOW() - INTERVAL '30 days'
            GROUP BY st.stylist_name
            ORDER BY revenue DESC
            LIMIT 1
            """,
            (salon_id,),
        )
        top_stylist_row = cur.fetchone()

    top_stylist_data = {}
    if top_stylist_row:
        stylist_revenue = round(float(top_stylist_row["revenue"]), 2)
        stylist_customer_count = int(top_stylist_row["customer_count"])
        revenue_share_pct = _safe_ratio(stylist_revenue, revenue_30d) * 100 if revenue_30d > 0 else 0.0
        top_stylist_data = {
            "name": top_stylist_row["stylist_name"],
            "revenue_30d": stylist_revenue,
            "customer_count": stylist_customer_count,
            "revenue_share_pct": round(revenue_share_pct, 1),
        }

    # ------------------------------------------------------------------
    # 6. Monthly retention trend (last 6 months)
    # ------------------------------------------------------------------
    retention_trend = _compute_retention_trend(conn, salon_id)

    # ------------------------------------------------------------------
    # 7. Product sales summary (last 30 days)
    # ------------------------------------------------------------------
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT pp.category, COUNT(*) AS count,
                   COALESCE(SUM(pp.price), 0) AS revenue
            FROM product_purchases pp
            JOIN visits v ON pp.visit_id = v.visit_id
            WHERE v.salon_id = %s AND v.status = 'COMPLETED'
              AND v.visit_date >= NOW() - INTERVAL '30 days'
            GROUP BY pp.category
            ORDER BY revenue DESC
            """,
            (salon_id,),
        )
        product_rows = cur.fetchall()

    product_categories = []
    product_total_revenue = 0.0
    product_total_count = 0
    for pr in product_rows:
        cat_revenue = round(float(pr["revenue"]), 2)
        cat_count = int(pr["count"])
        product_categories.append({
            "category": pr["category"],
            "count": cat_count,
            "revenue": cat_revenue,
        })
        product_total_revenue += cat_revenue
        product_total_count += cat_count

    # ------------------------------------------------------------------
    # 8. Service mix (last 30 days)
    # ------------------------------------------------------------------
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT s.category, COUNT(*) AS count,
                   COALESCE(SUM(vs.price), 0) AS revenue
            FROM visit_services vs
            JOIN visits v ON vs.visit_id = v.visit_id
            JOIN services s ON vs.service_id = s.service_id
            WHERE v.salon_id = %s AND v.status = 'COMPLETED'
              AND v.visit_date >= NOW() - INTERVAL '30 days'
            GROUP BY s.category
            ORDER BY revenue DESC
            """,
            (salon_id,),
        )
        service_rows = cur.fetchall()

    service_total_revenue = sum(float(sr["revenue"]) for sr in service_rows)
    service_mix = []
    for sr in service_rows:
        svc_revenue = round(float(sr["revenue"]), 2)
        svc_count = int(sr["count"])
        svc_pct = round(
            (svc_revenue / service_total_revenue * 100) if service_total_revenue > 0 else 0.0,
            1,
        )
        service_mix.append({
            "category": sr["category"],
            "count": svc_count,
            "revenue": svc_revenue,
            "percentage": svc_pct,
        })

    # ------------------------------------------------------------------
    # 9. Customer health from predictions
    # ------------------------------------------------------------------
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT risk_level, COUNT(*) AS cnt
            FROM predictions
            WHERE salon_id = %s
            GROUP BY risk_level
            """,
            (salon_id,),
        )
        health_rows = cur.fetchall()

    risk_to_health = {
        "LOW": "healthy",
        "MEDIUM": "watch",
        "HIGH": "warning",
        "CRITICAL": "critical",
    }
    customer_health = {"healthy": 0, "watch": 0, "warning": 0, "critical": 0}
    total_customers = 0
    for hr in health_rows:
        health_key = risk_to_health.get(hr["risk_level"], "healthy")
        count = int(hr["cnt"])
        customer_health[health_key] += count
        total_customers += count

    at_risk_count = customer_health["watch"] + customer_health["warning"] + customer_health["critical"]

    # Count new customers (first visit in the last 30 days)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT COUNT(*) AS new_count
            FROM (
                SELECT customer_id, MIN(visit_date) AS first_visit
                FROM visits
                WHERE salon_id = %s AND status = 'COMPLETED'
                GROUP BY customer_id
                HAVING MIN(visit_date) >= NOW() - INTERVAL '30 days'
            ) sub
            """,
            (salon_id,),
        )
        new_row = cur.fetchone()
    new_customer_count = int(new_row["new_count"]) if new_row else 0

    # ------------------------------------------------------------------
    # 10. Generate contextual business insights
    # ------------------------------------------------------------------
    business_insights = _generate_business_insights(
        mom_revenue_change=mom_revenue_change,
        at_risk_count=at_risk_count,
        total_customers=total_customers,
        walkin_ratio=walkin_ratio,
        top_stylist_data=top_stylist_data,
        noshow_rate=noshow_rate,
        avg_ticket=avg_ticket,
        new_customer_count=new_customer_count,
    )

    # ------------------------------------------------------------------
    # Assemble response
    # ------------------------------------------------------------------
    response = {
        "salon_id": salon_id,
        "revenue": {
            "last_30d": revenue_30d,
            "prev_30d": revenue_prev_30d,
            "mom_change_pct": mom_revenue_change,
            "avg_ticket": avg_ticket,
            "avg_daily_revenue": avg_daily_revenue,
        },
        "visits": {
            "last_30d": visit_count_30d,
            "prev_30d": visit_count_prev_30d,
            "mom_change_pct": mom_visit_change,
            "walkin_count": walkin_count,
            "appointment_count": appointment_count,
            "walkin_ratio": walkin_ratio,
            "noshow_rate": noshow_rate,
            "avg_per_day": round(visit_count_30d / 30, 1),
        },
        "action_items": action_items,
        "top_stylist": top_stylist_data if top_stylist_data else {
            "name": None,
            "revenue_30d": 0.0,
            "customer_count": 0,
            "revenue_share_pct": 0.0,
        },
        "service_mix": service_mix,
        "product_sales": {
            "total_revenue": round(product_total_revenue, 2),
            "total_count": product_total_count,
            "categories": product_categories,
        },
        "retention_trend": retention_trend,
        "business_insights": business_insights,
        "customer_health": customer_health,
    }

    # Cache in Redis
    rd.set(cache_key, json.dumps(response, default=str), ex=INSIGHTS_CACHE_TTL_SECONDS)

    logger.info(
        "Business insights computed for salon_id=%s: revenue_30d=%.2f, visits=%d, actions=%d, insights=%d",
        salon_id,
        revenue_30d,
        visit_count_30d,
        action_total,
        len(business_insights),
    )

    return response
