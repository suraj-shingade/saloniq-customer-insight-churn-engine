"""Feature computation engine for SalonIQ Customer Insight Engine.

Loads visit, booking, product, and service data per salon in bulk,
computes behavioural and transactional features per customer, then
persists results to PostgreSQL and Redis.
"""

import json
import logging
from collections import Counter, defaultdict
from datetime import date, datetime
from typing import Any

import numpy as np
import psycopg2.extras

logger = logging.getLogger(__name__)

_FEATURE_DEFAULTS: dict[str, Any] = {
    "visit_count_total": 0,
    "visit_count_90d": 0,
    "visit_count_180d": 0,
    "days_since_last_visit": 9999,
    "avg_inter_visit_days": 0.0,
    "std_inter_visit_days": 0.0,
    "inter_visit_trend": 0.0,
    "visit_regularity_index": 0.0,
    "overdue_ratio": 0.0,
    "dow_mode": -1,
    "hour_mode": -1,
    "avg_ticket_value": 0.0,
    "ticket_value_trend": 0.0,
    "max_ticket_value": 0.0,
    "service_variety_index": 0.0,
    "has_color_service": False,
    "has_treatment_service": False,
    "color_frequency_ratio": 0.0,
    "primary_stylist_id": None,
    "stylist_consistency_ratio": 0.0,
    "stylist_change_count": 0,
    "multi_stylist_flag": False,
    "walkin_ratio": 0.0,
    "noshow_rate": 0.0,
    "cancellation_rate": 0.0,
    "avg_booking_lead_days": 0.0,
    "product_purchase_count": 0,
    "product_purchase_frequency": 0.0,
    "product_category_count": 0,
}


def _to_date(val: Any) -> date:
    """Normalise a date or datetime value to a plain date."""
    if isinstance(val, datetime):
        return val.date()
    return val


def _safe_mode(values: list[int]) -> int:
    """Return the most common value, or -1 if the list is empty."""
    if not values:
        return -1
    return Counter(values).most_common(1)[0][0]


def _compute_inter_visit_gaps(sorted_dates: list[date]) -> list[float]:
    """Return list of inter-visit gaps in days from a sorted date list."""
    return [
        (sorted_dates[i + 1] - sorted_dates[i]).days
        for i in range(len(sorted_dates) - 1)
    ]


def _compute_visit_features(
    visits: list[dict], today: date
) -> dict[str, Any]:
    """Compute visit-pattern features from completed visits."""
    features: dict[str, Any] = {}

    completed = [v for v in visits if v["status"] == "COMPLETED"]
    total = len(completed)
    features["visit_count_total"] = total

    if total == 0:
        features["visit_count_90d"] = 0
        features["visit_count_180d"] = 0
        features["days_since_last_visit"] = 9999
        features["avg_inter_visit_days"] = 0.0
        features["std_inter_visit_days"] = 0.0
        features["inter_visit_trend"] = 0.0
        features["visit_regularity_index"] = 0.0
        features["overdue_ratio"] = 0.0
        features["dow_mode"] = -1
        features["hour_mode"] = -1
        return features

    visit_dates = sorted(_to_date(v["visit_date"]) for v in completed)
    visit_datetimes = [v["visit_date"] for v in completed]

    features["visit_count_90d"] = sum(
        1 for d in visit_dates if (today - d).days <= 90
    )
    features["visit_count_180d"] = sum(
        1 for d in visit_dates if (today - d).days <= 180
    )
    features["days_since_last_visit"] = (today - visit_dates[-1]).days

    gaps = _compute_inter_visit_gaps(visit_dates)

    if len(gaps) >= 1:
        mean_gap = float(np.mean(gaps))
        features["avg_inter_visit_days"] = round(mean_gap, 2)
    else:
        mean_gap = 0.0
        features["avg_inter_visit_days"] = 0.0

    if len(gaps) >= 2:
        std_gap = float(np.std(gaps, ddof=1))
        features["std_inter_visit_days"] = round(std_gap, 2)
    else:
        std_gap = 0.0
        features["std_inter_visit_days"] = 0.0

    if len(gaps) >= 3:
        last_gaps = gaps[-5:]
        x = np.arange(len(last_gaps), dtype=float)
        coeffs = np.polyfit(x, last_gaps, 1)
        features["inter_visit_trend"] = round(float(coeffs[0]), 4)

        if mean_gap > 0:
            regularity = 1.0 - (std_gap / mean_gap)
            features["visit_regularity_index"] = round(
                float(np.clip(regularity, 0.0, 1.0)), 4
            )
        else:
            features["visit_regularity_index"] = 0.0
    else:
        features["inter_visit_trend"] = 0.0
        features["visit_regularity_index"] = 0.0

    if mean_gap > 0 and total > 0:
        features["overdue_ratio"] = round(
            features["days_since_last_visit"] / mean_gap, 4
        )
    else:
        features["overdue_ratio"] = 0.0

    features["dow_mode"] = _safe_mode(
        [_to_date(d).weekday() for d in visit_datetimes]
    )
    features["hour_mode"] = _safe_mode(
        [
            d.hour
            for d in visit_datetimes
            if isinstance(d, datetime)
        ]
    )
    if features["hour_mode"] == -1 and visit_datetimes:
        features["hour_mode"] = 0

    return features


def _compute_service_revenue_features(
    visits: list[dict],
    visit_services: list[dict],
    service_categories: dict[str, str],
) -> dict[str, Any]:
    """Compute service and revenue features."""
    features: dict[str, Any] = {}

    completed = [v for v in visits if v["status"] == "COMPLETED"]
    total = len(completed)

    amounts = [
        float(v["total_amount"])
        for v in completed
        if v.get("total_amount") is not None
    ]

    if amounts:
        features["avg_ticket_value"] = round(float(np.mean(amounts)), 2)
        features["max_ticket_value"] = round(float(np.max(amounts)), 2)
    else:
        features["avg_ticket_value"] = 0.0
        features["max_ticket_value"] = 0.0

    if len(amounts) >= 3:
        last_amounts = amounts[-10:]
        x = np.arange(len(last_amounts), dtype=float)
        coeffs = np.polyfit(x, last_amounts, 1)
        features["ticket_value_trend"] = round(float(coeffs[0]), 4)
    else:
        features["ticket_value_trend"] = 0.0

    categories_used: set[str] = set()
    total_services = 0
    color_visit_ids: set[str] = set()
    has_color = False
    has_treatment = False

    for vs in visit_services:
        total_services += 1
        cat = service_categories.get(vs["service_id"], "")
        if cat:
            categories_used.add(cat)
        if cat == "COLOR":
            has_color = True
            color_visit_ids.add(vs["visit_id"])
        if cat == "TREATMENT":
            has_treatment = True

    features["service_variety_index"] = (
        round(len(categories_used) / total_services, 4)
        if total_services > 0
        else 0.0
    )
    features["has_color_service"] = has_color
    features["has_treatment_service"] = has_treatment

    if total > 0:
        features["color_frequency_ratio"] = round(
            len(color_visit_ids) / total, 4
        )
    else:
        features["color_frequency_ratio"] = 0.0

    return features


def _compute_stylist_features(visits: list[dict]) -> dict[str, Any]:
    """Compute stylist relationship features."""
    features: dict[str, Any] = {}

    completed = sorted(
        [v for v in visits if v["status"] == "COMPLETED"],
        key=lambda v: v["visit_date"],
    )
    total = len(completed)

    if total == 0:
        features["primary_stylist_id"] = None
        features["stylist_consistency_ratio"] = 0.0
        features["stylist_change_count"] = 0
        features["multi_stylist_flag"] = False
        return features

    stylist_counter: Counter = Counter(
        v["stylist_id"] for v in completed if v.get("stylist_id")
    )

    if stylist_counter:
        primary_stylist, primary_count = stylist_counter.most_common(1)[0]
        features["primary_stylist_id"] = str(primary_stylist)
        features["stylist_consistency_ratio"] = round(
            primary_count / total, 4
        )
    else:
        features["primary_stylist_id"] = None
        features["stylist_consistency_ratio"] = 0.0

    change_count = 0
    if total >= 2:
        for i in range(1, len(completed)):
            prev_stylist = completed[i - 1].get("stylist_id")
            curr_stylist = completed[i].get("stylist_id")
            if prev_stylist and curr_stylist and prev_stylist != curr_stylist:
                change_count += 1
    features["stylist_change_count"] = change_count

    distinct_stylists = set(
        v["stylist_id"] for v in completed if v.get("stylist_id")
    )
    features["multi_stylist_flag"] = len(distinct_stylists) >= 3

    return features


def _compute_booking_features(
    visits: list[dict], bookings: list[dict]
) -> dict[str, Any]:
    """Compute booking behaviour features."""
    features: dict[str, Any] = {}

    completed_visits = [v for v in visits if v["status"] == "COMPLETED"]
    total_visits = len(completed_visits)
    total_bookings = len(bookings)

    if total_visits > 0:
        walkin_count = sum(
            1 for v in completed_visits if v.get("visit_type") == "WALKIN"
        )
        features["walkin_ratio"] = round(walkin_count / total_visits, 4)
    else:
        features["walkin_ratio"] = 0.0

    if total_bookings > 0:
        noshow_count = sum(
            1 for b in bookings if b.get("status") == "NOSHOW"
        )
        cancel_count = sum(
            1 for b in bookings if b.get("status") == "CANCELLED"
        )
        features["noshow_rate"] = round(noshow_count / total_bookings, 4)
        features["cancellation_rate"] = round(
            cancel_count / total_bookings, 4
        )

        lead_days = []
        for b in bookings:
            if (
                b.get("status") == "COMPLETED"
                and b.get("appointment_date")
                and b.get("booked_date")
            ):
                appt = _to_date(b["appointment_date"])
                booked = _to_date(b["booked_date"])
                lead_days.append((appt - booked).days)
        features["avg_booking_lead_days"] = (
            round(float(np.mean(lead_days)), 2) if lead_days else 0.0
        )
    else:
        features["noshow_rate"] = 0.0
        features["cancellation_rate"] = 0.0
        features["avg_booking_lead_days"] = 0.0

    return features


def _compute_product_features(
    purchases: list[dict], visit_count_total: int
) -> dict[str, Any]:
    """Compute product purchase features."""
    features: dict[str, Any] = {}
    total_purchases = len(purchases)
    features["product_purchase_count"] = total_purchases

    if visit_count_total > 0:
        features["product_purchase_frequency"] = round(
            total_purchases / visit_count_total, 4
        )
    else:
        features["product_purchase_frequency"] = 0.0

    distinct_categories = set(
        p["category"] for p in purchases if p.get("category")
    )
    features["product_category_count"] = len(distinct_categories)

    return features


def _load_salon_customers(cursor, salon_id: str) -> list[str]:
    """Load all customer IDs for a given salon."""
    cursor.execute(
        "SELECT customer_id FROM customers WHERE salon_id = %s",
        (salon_id,),
    )
    return [row["customer_id"] for row in cursor.fetchall()]


def _load_salon_visits(cursor, salon_id: str) -> list[dict]:
    """Bulk-load all visits for a salon."""
    cursor.execute(
        """
        SELECT visit_id, customer_id, stylist_id, visit_date,
               visit_type, status, total_amount
        FROM visits
        WHERE salon_id = %s
        """,
        (salon_id,),
    )
    return cursor.fetchall()


def _load_salon_visit_services(
    cursor, visit_ids: list[str]
) -> list[dict]:
    """Bulk-load visit services for a set of visit IDs."""
    if not visit_ids:
        return []
    cursor.execute(
        """
        SELECT vs.visit_id, vs.service_id
        FROM visit_services vs
        WHERE vs.visit_id = ANY(%s)
        """,
        (visit_ids,),
    )
    return cursor.fetchall()


def _load_service_categories(cursor, _salon_id: str) -> dict[str, str]:
    """Load service_id -> category mapping (global catalog, not per-salon)."""
    cursor.execute("SELECT service_id, category FROM services")
    return {row["service_id"]: row["category"] for row in cursor.fetchall()}


def _load_salon_bookings(cursor, salon_id: str) -> list[dict]:
    """Bulk-load all bookings for a salon."""
    cursor.execute(
        """
        SELECT booking_id, customer_id, appointment_date,
               booked_date, status
        FROM bookings
        WHERE salon_id = %s
        """,
        (salon_id,),
    )
    return cursor.fetchall()


def _load_salon_product_purchases(cursor, salon_id: str) -> list[dict]:
    """Bulk-load all product purchases for a salon (joined through visits)."""
    cursor.execute(
        """
        SELECT pp.purchase_id, pp.customer_id, pp.category
        FROM product_purchases pp
        JOIN visits v ON pp.visit_id = v.visit_id
        WHERE v.salon_id = %s
        """,
        (salon_id,),
    )
    return cursor.fetchall()


def _group_by_customer(rows: list[dict], key: str = "customer_id") -> dict:
    """Group a list of row dicts by a given key."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return grouped


_FEATURE_COLUMNS = list(_FEATURE_DEFAULTS.keys())


def _batch_upsert_features(
    cursor, records: list[tuple]
) -> None:
    """Batch upsert feature records into customer_features (individual columns)."""
    if not records:
        return
    col_list = ", ".join(["salon_id", "customer_id"] + _FEATURE_COLUMNS + ["computed_at"])
    placeholders = ", ".join(["%s"] * (2 + len(_FEATURE_COLUMNS)) + ["NOW()"])
    set_clause = ", ".join(
        f"{col} = EXCLUDED.{col}" for col in _FEATURE_COLUMNS
    )
    set_clause += ", computed_at = EXCLUDED.computed_at"
    query = f"""
        INSERT INTO customer_features ({col_list})
        VALUES ({placeholders})
        ON CONFLICT (salon_id, customer_id) DO UPDATE SET
            {set_clause}
    """
    cursor.executemany(query, records)


def compute_all_features(pg_conn, redis_client) -> int:
    """Compute and persist features for every customer across all salons.

    Returns the total number of customer feature vectors computed.
    """
    today = date.today()
    total_computed = 0

    with pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT DISTINCT salon_id FROM customers ORDER BY salon_id"
        )
        salon_ids = [row["salon_id"] for row in cur.fetchall()]

    total_salons = len(salon_ids)
    logger.info(
        "Starting feature computation for %d salons", total_salons
    )

    for idx, salon_id in enumerate(salon_ids, start=1):
        logger.info(
            "Computing features for salon %s (%d/%d)",
            salon_id,
            idx,
            total_salons,
        )

        with pg_conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor
        ) as cur:
            customer_ids = _load_salon_customers(cur, salon_id)
            if not customer_ids:
                continue

            visits = _load_salon_visits(cur, salon_id)
            visit_ids = [v["visit_id"] for v in visits]
            visit_services = _load_salon_visit_services(cur, visit_ids)
            service_categories = _load_service_categories(cur, salon_id)
            bookings = _load_salon_bookings(cur, salon_id)
            purchases = _load_salon_product_purchases(cur, salon_id)

        visits_by_customer = _group_by_customer(visits)
        vs_by_visit = _group_by_customer(visit_services, key="visit_id")
        bookings_by_customer = _group_by_customer(bookings)
        purchases_by_customer = _group_by_customer(purchases)

        upsert_records: list[tuple[str, str, str]] = []
        pipeline = redis_client.pipeline(transaction=False)

        for customer_id in customer_ids:
            cust_visits = visits_by_customer.get(customer_id, [])
            cust_bookings = bookings_by_customer.get(customer_id, [])
            cust_purchases = purchases_by_customer.get(customer_id, [])

            cust_visit_ids = [v["visit_id"] for v in cust_visits]
            cust_vs = []
            for vid in cust_visit_ids:
                cust_vs.extend(vs_by_visit.get(vid, []))

            feat = dict(_FEATURE_DEFAULTS)
            feat.update(_compute_visit_features(cust_visits, today))
            feat.update(
                _compute_service_revenue_features(
                    cust_visits, cust_vs, service_categories
                )
            )
            feat.update(_compute_stylist_features(cust_visits))
            feat.update(
                _compute_booking_features(cust_visits, cust_bookings)
            )
            feat.update(
                _compute_product_features(
                    cust_purchases, feat["visit_count_total"]
                )
            )

            features_json = json.dumps(feat, default=str)
            row_values = tuple(
                [salon_id, customer_id]
                + [feat[col] for col in _FEATURE_COLUMNS]
            )
            upsert_records.append(row_values)

            redis_key = f"features:{salon_id}:{customer_id}"
            pipeline.set(redis_key, features_json)

        with pg_conn.cursor() as cur:
            _batch_upsert_features(cur, upsert_records)
        pg_conn.commit()

        pipeline.execute()

        salon_count = len(customer_ids)
        total_computed += salon_count
        logger.info(
            "Completed salon %s -- %d customer features persisted",
            salon_id,
            salon_count,
        )

    logger.info(
        "Feature computation complete -- %d total features computed",
        total_computed,
    )
    return total_computed
