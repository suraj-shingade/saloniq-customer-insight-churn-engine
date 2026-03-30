"""
SalonIQ Data Generator -- One-shot seeder for PostgreSQL with realistic synthetic Japanese salon data.
Produces deterministic output (numpy seed=42) spanning services, salons, stylists, customers,
visits, visit-services, bookings, product purchases, and case-study metrics.
"""

import logging
import os
import sys
import time
from datetime import date, timedelta

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("saloniq.data-generator")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------
DATA_START = date(2023, 3, 1)
DATA_END = date(2026, 3, 15)
CUSTOMER_FIRST_VISIT_START = date(2023, 1, 1)
CUSTOMER_FIRST_VISIT_END = date(2026, 1, 1)
REFERENCE_DATE = date(2026, 3, 15)


def _random_date_between(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=int(RNG.integers(0, max(delta, 1))))


def _clamp_date(d: date) -> date:
    if d < DATA_START:
        return DATA_START
    if d > DATA_END:
        return DATA_END
    return d


# ---------------------------------------------------------------------------
# Name pools
# ---------------------------------------------------------------------------
FAMILY_NAMES = [
    "Tanaka", "Sato", "Suzuki", "Takahashi", "Watanabe",
    "Ito", "Yamamoto", "Nakamura", "Kobayashi", "Kato",
    "Yoshida", "Yamada", "Sasaki", "Yamaguchi", "Matsumoto",
    "Inoue", "Kimura", "Shimizu", "Hayashi", "Saito",
    "Sakai", "Maeda", "Fujita", "Ogawa", "Goto",
    "Hasegawa", "Murakami", "Kondo", "Ishii", "Ueda",
    "Morita", "Hara", "Fujii", "Okada", "Endo",
    "Aoki", "Nishimura", "Fukuda", "Miura", "Tamura",
]

GIVEN_NAMES = [
    "Yuki", "Haruto", "Sakura", "Ren", "Hina",
    "Sota", "Mei", "Yuto", "Aoi", "Riku",
    "Mio", "Kaito", "Koharu", "Hinata", "Yui",
    "Sora", "Akari", "Takumi", "Nana", "Hayato",
    "Riko", "Shun", "Miyu", "Kenta", "Rin",
    "Daiki", "Yuna", "Ryota", "Saki", "Kazuki",
    "Mana", "Naoki", "Chihiro", "Kenji", "Ayaka",
    "Tsubasa", "Kanon", "Yuya", "Misaki", "Ryusei",
]

SALON_NAMES = [
    "Atelier KAZE", "SALON ARIA", "Hair Studio REVE", "BELLE EPOQUE", "PRISM HAIR",
    "Salon de FLEUR", "LUXE HAIR STUDIO", "Hair Labo MIST", "CACHE CACHE", "SOIE Hair Design",
    "ECLAT SALON", "Hair Room LILAS", "Salon VERDE", "CIEL BLEU Hair", "Atelier NOEL",
    "BRANCHE Hair Studio", "Salon de LUCE", "RAFFINEE HAIR", "Hair Craft SION", "AUREOLE Salon",
    "Salon NAGARÉ", "Hair Atelier HAKU", "LUMIERE Hair", "Salon POLARIS", "CREER Hair Studio",
    "Hair Room COCO", "SERENITE Salon", "Atelier SOLA", "Salon de BLOOM", "FIERTE Hair",
    "Hair Studio KAI", "RUBAN Salon", "Salon VIVACE", "PLAISIR Hair", "Hair Labo NINE",
    "TERRACE Hair Studio", "Salon NUAGE", "RESPIRE Hair", "Hair Room LIEN", "Atelier MUZE",
    "Salon LUCIOLE", "CANDEUR Hair Studio", "Hair Craft YUME", "Salon de PAUME", "NEIGE Hair",
    "Hair Studio RIN", "BONHEUR Salon", "Atelier CLAIR", "Salon AURORE", "SOUFFLE Hair Design",
]

# ---------------------------------------------------------------------------
# Service catalog (15 records)
# ---------------------------------------------------------------------------
SERVICE_CATALOG = [
    ("SVC_001", "Cut", "CUT", 5500),
    ("SVC_002", "Men's Cut", "CUT", 4500),
    ("SVC_003", "Kids Cut", "CUT", 3500),
    ("SVC_004", "Full Color", "COLOR", 8500),
    ("SVC_005", "Highlight", "COLOR", 10000),
    ("SVC_006", "Balayage", "COLOR", 13000),
    ("SVC_007", "Digital Perm", "PERM", 12000),
    ("SVC_008", "Cold Perm", "PERM", 9000),
    ("SVC_009", "Straight Perm", "PERM", 15000),
    ("SVC_010", "Salon Treatment", "TREATMENT", 4000),
    ("SVC_011", "Deep Treatment", "TREATMENT", 6500),
    ("SVC_012", "Keratin Treatment", "TREATMENT", 9000),
    ("SVC_013", "Head Spa", "SPA", 5000),
    ("SVC_014", "Scalp Treatment", "SPA", 6000),
    ("SVC_015", "Shampoo and Blow", "OTHER", 2500),
]

SERVICE_PRICE_MAP = {s[0]: s[3] for s in SERVICE_CATALOG}

CATEGORY_DURATION = {
    "CUT": 45, "COLOR": 90, "PERM": 120,
    "TREATMENT": 60, "SPA": 45, "OTHER": 30,
}

SERVICE_CATEGORY_MAP = {s[0]: s[2] for s in SERVICE_CATALOG}

CUT_SERVICES = ["SVC_001", "SVC_002", "SVC_003"]
COLOR_SERVICES = ["SVC_004", "SVC_005", "SVC_006"]
TREATMENT_SERVICES = ["SVC_010", "SVC_011", "SVC_012"]
SPA_SERVICES = ["SVC_013", "SVC_014"]

# ---------------------------------------------------------------------------
# Region / city distribution
# ---------------------------------------------------------------------------
REGION_CONFIG = {
    "Kanto": {"count": 20, "cities": ["Tokyo", "Yokohama", "Chiba", "Saitama"]},
    "Kansai": {"count": 12, "cities": ["Osaka", "Kobe", "Kyoto"]},
    "Chubu": {"count": 8, "cities": ["Nagoya", "Shizuoka"]},
    "Tohoku": {"count": 4, "cities": ["Sendai"]},
    "Kyushu": {"count": 4, "cities": ["Fukuoka", "Kumamoto"]},
    "Hokkaido": {"count": 2, "cities": ["Sapporo"]},
}

# ---------------------------------------------------------------------------
# Product catalog
# ---------------------------------------------------------------------------
PRODUCTS = [
    ("Moisture Shampoo", "HAIRCARE"),
    ("Repair Conditioner", "HAIRCARE"),
    ("Argan Treatment Oil", "HAIRCARE"),
    ("Styling Wax", "STYLING"),
    ("Hair Serum", "HAIRCARE"),
    ("Volume Mousse", "STYLING"),
    ("Color Protect Spray", "HAIRCARE"),
    ("Scalp Tonic", "SKINCARE"),
    ("Heat Protectant", "HAIRCARE"),
    ("Silk Hair Mask", "HAIRCARE"),
]

# ---------------------------------------------------------------------------
# Archetype distribution
# ---------------------------------------------------------------------------
ARCHETYPE_WEIGHTS = {
    "LOYAL_REGULAR": 0.35,
    "DECLINING": 0.20,
    "CHURNED": 0.15,
    "SEASONAL": 0.15,
    "NEW": 0.10,
    "SPORADIC": 0.05,
}

# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------

def connect_with_retry(max_retries: int = 10, delay_seconds: int = 3):
    """Establish a PostgreSQL connection with retry logic for container startup ordering."""
    dsn_params = {
        "host": os.environ.get("POSTGRES_HOST", "localhost"),
        "port": os.environ.get("POSTGRES_PORT", "5432"),
        "dbname": os.environ.get("POSTGRES_DB", "saloniq"),
        "user": os.environ.get("POSTGRES_USER", "saloniq"),
        "password": os.environ.get("POSTGRES_PASSWORD", "saloniq"),
    }
    for attempt in range(1, max_retries + 1):
        try:
            conn = psycopg2.connect(**dsn_params)
            logger.info("Database connection established on attempt %d", attempt)
            return conn
        except psycopg2.OperationalError as exc:
            if attempt == max_retries:
                logger.error("Failed to connect after %d attempts", max_retries)
                raise
            logger.warning(
                "Connection attempt %d/%d failed: %s -- retrying in %ds",
                attempt, max_retries, exc, delay_seconds,
            )
            time.sleep(delay_seconds)


# ---------------------------------------------------------------------------
# Data already seeded?
# ---------------------------------------------------------------------------

def data_exists(cur) -> bool:
    try:
        cur.execute("SELECT count(*) FROM salons")
        count = cur.fetchone()[0]
        return count > 0
    except psycopg2.errors.UndefinedTable:
        cur.connection.rollback()
        return False


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def generate_salons():
    """Return list of salon tuples distributed across regions."""
    salons = []
    idx = 0
    region_order = ["Kanto", "Kansai", "Chubu", "Tohoku", "Kyushu", "Hokkaido"]
    for region in region_order:
        cfg = REGION_CONFIG[region]
        for _ in range(cfg["count"]):
            salon_num = idx + 1
            salon_id = f"SALON_{salon_num:03d}"
            city = RNG.choice(cfg["cities"])

            if salon_id == "SALON_001":
                name = "Atelier KAZE"
                city = "Tokyo"
                stylist_count = 6
                chair_count = 8
                tier = "BASIC_PLUS"
            else:
                name = SALON_NAMES[idx]
                stylist_count = int(RNG.integers(3, 9))
                chair_count = stylist_count + int(RNG.integers(1, 4))
                tier_roll = RNG.random()
                if tier_roll < 0.20:
                    tier = "SIMPLE"
                elif tier_roll < 0.70:
                    tier = "BASIC"
                else:
                    tier = "BASIC_PLUS"

            salons.append((salon_id, name, region, city, stylist_count, chair_count, tier))
            idx += 1
    return salons


def generate_stylists(salons):
    """Generate stylists for every salon based on its stylist_count."""
    stylists = []
    specialties = ["CUT", "COLOR", "PERM", "TREATMENT"]
    specialty_weights = [0.40, 0.25, 0.20, 0.15]

    for salon_id, _, _, _, stylist_count, _, _ in salons:
        salon_num = salon_id.split("_")[1]
        for seq in range(1, stylist_count + 1):
            stylist_id = f"STY_{salon_num}_{seq:02d}"
            family = RNG.choice(FAMILY_NAMES)
            given = RNG.choice(GIVEN_NAMES)
            name = f"{family} {given}"
            specialty = RNG.choice(specialties, p=specialty_weights)
            experience = int(RNG.integers(1, 21))
            stylists.append((stylist_id, salon_id, name, specialty, experience))
    return stylists


def generate_customers(salons):
    """Generate ~5000 customers distributed across salons proportional to size."""
    total_stylists = sum(s[4] for s in salons)
    target_total = 5000

    customers = []
    cust_idx = 0
    archetype_names = list(ARCHETYPE_WEIGHTS.keys())
    archetype_probs = list(ARCHETYPE_WEIGHTS.values())

    gender_choices = ["M", "F", "OTHER"]
    gender_weights = [0.30, 0.65, 0.05]

    age_choices = ["20s", "30s", "40s", "50s", "60+"]
    age_weights = [0.25, 0.30, 0.25, 0.15, 0.05]

    for salon_id, _, _, _, stylist_count, _, _ in salons:
        proportion = stylist_count / total_stylists
        count = max(60, min(150, int(round(proportion * target_total))))
        for _ in range(count):
            cust_idx += 1
            customer_id = f"CUST_{cust_idx:05d}"
            family = RNG.choice(FAMILY_NAMES)
            given = RNG.choice(GIVEN_NAMES)
            name = f"{family} {given}"
            gender = RNG.choice(gender_choices, p=gender_weights)
            age_group = RNG.choice(age_choices, p=age_weights)
            archetype = RNG.choice(archetype_names, p=archetype_probs)

            if archetype == "NEW":
                first_visit = _random_date_between(
                    REFERENCE_DATE - timedelta(days=90), REFERENCE_DATE,
                )
            else:
                first_visit = _random_date_between(
                    CUSTOMER_FIRST_VISIT_START, CUSTOMER_FIRST_VISIT_END,
                )

            customers.append((
                customer_id, salon_id, name, gender, age_group,
                first_visit.isoformat(), archetype,
            ))

    logger.info("Generated %d customers across %d salons", len(customers), len(salons))
    return customers


# ---------------------------------------------------------------------------
# Visit generation (archetype-aware)
# ---------------------------------------------------------------------------

def _pick_cut_service(gender: str) -> str:
    if gender == "M":
        return "SVC_002"
    if gender == "F":
        return "SVC_001"
    return RNG.choice(["SVC_001", "SVC_002"])


def _generate_visit_services(gender: str, archetype: str):
    """Return list of (service_id, price) for a single visit."""
    services = []
    cut_svc = _pick_cut_service(gender)
    price = float(SERVICE_PRICE_MAP[cut_svc] * RNG.uniform(0.9, 1.1))
    services.append((cut_svc, round(price)))

    color_chance = 0.50 if archetype in ("LOYAL_REGULAR", "SEASONAL") else 0.40
    if RNG.random() < color_chance:
        svc = RNG.choice(COLOR_SERVICES)
        price = float(SERVICE_PRICE_MAP[svc] * RNG.uniform(0.9, 1.1))
        services.append((svc, round(price)))

    if RNG.random() < 0.20:
        svc = RNG.choice(TREATMENT_SERVICES)
        price = float(SERVICE_PRICE_MAP[svc] * RNG.uniform(0.9, 1.1))
        services.append((svc, round(price)))

    if RNG.random() < 0.10:
        svc = RNG.choice(SPA_SERVICES)
        price = float(SERVICE_PRICE_MAP[svc] * RNG.uniform(0.9, 1.1))
        services.append((svc, round(price)))

    return services


def _compute_duration(service_ids: list[str]) -> int:
    categories = [SERVICE_CATEGORY_MAP[sid] for sid in service_ids]
    return sum(CATEGORY_DURATION.get(c, 30) for c in categories)


def _determine_visit_type(archetype: str) -> str:
    """Determine APPOINTMENT vs WALKIN based on archetype probabilities."""
    appt_probability = 0.50 if archetype == "NEW" else 0.85
    return "APPOINTMENT" if RNG.random() < appt_probability else "WALKIN"


def _process_single_visit(visit_id, customer_id, salon_id, stylist_id,
                          v_date, gender, archetype, collector):
    """Build visit, visit_services, booking, and purchase records for one visit.
    Returns 1 if the visit was an APPOINTMENT, 0 otherwise.
    """
    svc_list = _generate_visit_services(gender, archetype)
    svc_ids = [s[0] for s in svc_list]
    total_amount = sum(s[1] for s in svc_list)
    duration = _compute_duration(svc_ids)
    visit_type = _determine_visit_type(archetype)

    collector["visits"].append((
        visit_id, customer_id, salon_id, stylist_id,
        v_date.isoformat(), visit_type, "COMPLETED",
        duration, total_amount,
    ))

    for svc_id, svc_price in svc_list:
        collector["visit_services"].append((visit_id, svc_id, svc_price))

    appt_flag = 0
    if visit_type == "APPOINTMENT":
        appt_flag = 1
        collector["booking_counter"] += 1
        booking_id = f"BKG_{collector['booking_counter']:06d}"
        booked_date = v_date - timedelta(days=int(RNG.integers(1, 22)))
        collector["bookings"].append((
            booking_id, customer_id, salon_id, stylist_id,
            booked_date.isoformat(), v_date.isoformat(), "COMPLETED",
        ))

    purchase_chance = 0.30 if archetype == "LOYAL_REGULAR" else 0.20
    if RNG.random() < purchase_chance:
        collector["purchase_counter"] += 1
        purchase_id = f"PP_{collector['purchase_counter']:06d}"
        product_name, category = PRODUCTS[int(RNG.integers(0, len(PRODUCTS)))]
        price = int(RNG.integers(1500, 5001))
        collector["purchases"].append((
            purchase_id, visit_id, customer_id,
            product_name, category, price,
        ))

    return appt_flag


def _generate_extra_bookings(archetype, customer_appt_count, customer_id,
                             salon_id, stylist_ids, first_visit, collector):
    """Generate additional CANCELLED / NOSHOW bookings for archetype flavoring."""
    is_at_risk = archetype in ("DECLINING", "CHURNED")
    rates = [
        ("CANCELLED", 0.15 if is_at_risk else 0.05),
        ("NOSHOW", 0.10 if is_at_risk else 0.03),
    ]
    for extra_status, rate in rates:
        extra_count = int(round(customer_appt_count * rate))
        for _ in range(extra_count):
            collector["booking_counter"] += 1
            booking_id = f"BKG_{collector['booking_counter']:06d}"
            rand_date = _random_date_between(first_visit, DATA_END)
            booked_date = rand_date - timedelta(days=int(RNG.integers(1, 22)))
            collector["bookings"].append((
                booking_id, customer_id, salon_id,
                RNG.choice(stylist_ids),
                booked_date.isoformat(), rand_date.isoformat(),
                extra_status,
            ))


def generate_visits_and_related(customers, salon_stylist_map):
    """
    Generate visits, visit_services, bookings, and product_purchases
    for all customers based on their archetype.
    Returns (visits, visit_services, bookings, purchases).
    """
    collector = {
        "visits": [],
        "visit_services": [],
        "bookings": [],
        "purchases": [],
        "booking_counter": 0,
        "purchase_counter": 0,
    }
    visit_counter = 0

    for (customer_id, salon_id, _name, gender, _age,
         first_visit_str, archetype) in customers:

        first_visit = date.fromisoformat(first_visit_str)
        stylist_ids = salon_stylist_map.get(salon_id, [])
        if not stylist_ids:
            continue

        primary_stylist = RNG.choice(stylist_ids)
        visit_dates_and_stylists = _compute_visit_sequence(
            archetype, first_visit, stylist_ids, primary_stylist,
        )

        customer_appt_count = 0
        for v_date, stylist_id in visit_dates_and_stylists:
            if v_date < DATA_START or v_date > DATA_END:
                continue
            visit_counter += 1
            visit_id = f"VIS_{visit_counter:06d}"
            customer_appt_count += _process_single_visit(
                visit_id, customer_id, salon_id, stylist_id,
                v_date, gender, archetype, collector,
            )

        _generate_extra_bookings(
            archetype, customer_appt_count, customer_id,
            salon_id, stylist_ids, first_visit, collector,
        )

    logger.info(
        "Generated %d visits, %d visit-services, %d bookings, %d purchases",
        len(collector["visits"]), len(collector["visit_services"]),
        len(collector["bookings"]), len(collector["purchases"]),
    )
    return (collector["visits"], collector["visit_services"],
            collector["bookings"], collector["purchases"])


def _compute_visit_sequence(archetype, first_visit, stylist_ids, primary_stylist):
    """Return list of (date, stylist_id) tuples for one customer."""
    sequence = []

    if archetype == "LOYAL_REGULAR":
        current = first_visit
        while current <= DATA_END:
            stylist = primary_stylist if RNG.random() < 0.90 else RNG.choice(stylist_ids)
            sequence.append((current, stylist))
            interval = max(14, int(RNG.normal(35, 4)))
            current += timedelta(days=interval)

    elif archetype == "DECLINING":
        current = first_visit
        intervals = []
        while current <= DATA_END:
            sequence.append((current, primary_stylist))
            interval = max(14, int(RNG.normal(35, 5)))
            intervals.append(interval)
            current += timedelta(days=interval)

        if len(sequence) > 6:
            # Rebuild the last 6 visits with increasing intervals
            base_seq = sequence[:-6]
            rebuild_start = sequence[-7][0] if len(sequence) > 7 else sequence[-6][0]
            current = rebuild_start + timedelta(days=intervals[-7] if len(intervals) > 6 else 35)
            last_interval = 35
            for i in range(6):
                last_interval += int(RNG.integers(5, 16))
                current += timedelta(days=last_interval)
                if current > DATA_END:
                    break
                stylist = RNG.choice(stylist_ids) if (i >= 3 and RNG.random() < 0.50) else primary_stylist
                base_seq.append((current, stylist))
            sequence = base_seq

        # Ensure last visit is 30-90 days ago
        if sequence:
            target_last = REFERENCE_DATE - timedelta(days=int(RNG.integers(30, 91)))
            shift = (target_last - sequence[-1][0]).days
            if shift != 0:
                sequence = [(_clamp_date(d + timedelta(days=shift)), s) for d, s in sequence]

    elif archetype == "CHURNED":
        num_visits = int(RNG.integers(5, 21))
        current = first_visit
        interval = max(14, int(RNG.normal(35, 5)))
        for i in range(num_visits):
            sequence.append((current, primary_stylist))
            if i > num_visits // 2:
                interval += int(RNG.integers(5, 16))
            current += timedelta(days=max(14, interval))
            if current > DATA_END:
                break

        # Last visit between 91 and 365 days ago
        if sequence:
            target_last = REFERENCE_DATE - timedelta(days=int(RNG.integers(91, 366)))
            shift = (target_last - sequence[-1][0]).days
            if shift != 0:
                sequence = [(_clamp_date(d + timedelta(days=shift)), s) for d, s in sequence]

    elif archetype == "SEASONAL":
        seasonal_months = {3, 4, 6, 7, 10, 11, 12}
        year = first_visit.year
        while year <= DATA_END.year:
            visits_this_year = int(RNG.integers(4, 9))
            chosen_months = RNG.choice(sorted(seasonal_months), size=min(visits_this_year, len(seasonal_months)), replace=False)
            for m in sorted(chosen_months):
                day = int(RNG.integers(1, 29))
                try:
                    vd = date(year, int(m), day)
                except ValueError:
                    vd = date(year, int(m), 28)
                if vd >= first_visit and vd <= DATA_END:
                    sequence.append((vd, primary_stylist))
            year += 1

    elif archetype == "NEW":
        num_visits = int(RNG.integers(1, 4))
        current = first_visit
        for _ in range(num_visits):
            stylist = RNG.choice(stylist_ids)
            sequence.append((current, stylist))
            current += timedelta(days=int(RNG.integers(7, 30)))
            if current > DATA_END:
                break

    elif archetype == "SPORADIC":
        num_visits = int(RNG.integers(5, 16))
        current = first_visit
        for _ in range(num_visits):
            stylist = RNG.choice(stylist_ids)
            sequence.append((current, stylist))
            interval = int(RNG.integers(20, 121))
            current += timedelta(days=interval)
            if current > DATA_END:
                break

    return sequence


# ---------------------------------------------------------------------------
# Case study metrics
# ---------------------------------------------------------------------------

CASE_STUDY_METRICS = [
    ("SALON_001", "BEFORE", "churn_rate", 18.5, "percent", "retention"),
    ("SALON_001", "BEFORE", "avg_inter_visit_days", 42.0, "days", "engagement"),
    ("SALON_001", "BEFORE", "reengagement_rate", 12.0, "percent", "retention"),
    ("SALON_001", "BEFORE", "revenue_per_customer_year", 52000.0, "jpy", "revenue"),
    ("SALON_001", "BEFORE", "admin_time_per_day", 25.0, "minutes", "operations"),
    ("SALON_001", "BEFORE", "customer_retention_12m", 68.0, "percent", "retention"),
    ("SALON_001", "BEFORE", "at_risk_identification_rate", 15.0, "percent", "analytics"),
    ("SALON_001", "BEFORE", "campaign_response_rate", 4.2, "percent", "marketing"),
    ("SALON_001", "BEFORE", "avg_ticket_value", 8500.0, "jpy", "revenue"),
    ("SALON_001", "BEFORE", "noshow_rate", 12.0, "percent", "operations"),
    ("SALON_001", "BEFORE", "active_customers", 285.0, "count", "engagement"),
    ("SALON_001", "BEFORE", "monthly_revenue", 2420000.0, "jpy", "revenue"),
    ("SALON_001", "AFTER", "churn_rate", 11.2, "percent", "retention"),
    ("SALON_001", "AFTER", "avg_inter_visit_days", 36.0, "days", "engagement"),
    ("SALON_001", "AFTER", "reengagement_rate", 38.0, "percent", "retention"),
    ("SALON_001", "AFTER", "revenue_per_customer_year", 61500.0, "jpy", "revenue"),
    ("SALON_001", "AFTER", "admin_time_per_day", 10.0, "minutes", "operations"),
    ("SALON_001", "AFTER", "customer_retention_12m", 82.0, "percent", "retention"),
    ("SALON_001", "AFTER", "at_risk_identification_rate", 94.0, "percent", "analytics"),
    ("SALON_001", "AFTER", "campaign_response_rate", 11.8, "percent", "marketing"),
    ("SALON_001", "AFTER", "avg_ticket_value", 9800.0, "jpy", "revenue"),
    ("SALON_001", "AFTER", "noshow_rate", 7.2, "percent", "operations"),
    ("SALON_001", "AFTER", "active_customers", 342.0, "count", "engagement"),
    ("SALON_001", "AFTER", "monthly_revenue", 3150000.0, "jpy", "revenue"),
]

# ---------------------------------------------------------------------------
# Insertion helpers
# ---------------------------------------------------------------------------

INSERT_SERVICES = """
    INSERT INTO services (service_id, service_name, category, base_price)
    VALUES %s
"""

INSERT_SALONS = """
    INSERT INTO salons (salon_id, salon_name, region, city, stylist_count, chair_count, subscription_tier)
    VALUES %s
"""

INSERT_STYLISTS = """
    INSERT INTO stylists (stylist_id, salon_id, stylist_name, specialty, experience_years)
    VALUES %s
"""

INSERT_CUSTOMERS = """
    INSERT INTO customers (customer_id, salon_id, customer_name, gender, age_group, first_visit_date)
    VALUES %s
"""

INSERT_VISITS = """
    INSERT INTO visits (visit_id, customer_id, salon_id, stylist_id, visit_date, visit_type, status, duration_minutes, total_amount)
    VALUES %s
"""

INSERT_VISIT_SERVICES = """
    INSERT INTO visit_services (visit_id, service_id, price)
    VALUES %s
"""

INSERT_BOOKINGS = """
    INSERT INTO bookings (booking_id, customer_id, salon_id, stylist_id, booked_date, appointment_date, status)
    VALUES %s
"""

INSERT_PURCHASES = """
    INSERT INTO product_purchases (purchase_id, visit_id, customer_id, product_name, category, price)
    VALUES %s
"""

INSERT_METRICS = """
    INSERT INTO case_study_metrics (salon_id, period, metric_name, metric_value, metric_unit, category)
    VALUES %s
"""


def _batch_insert(cur, sql, data, table_name, page_size=1000):
    logger.info("Inserting %d records into %s...", len(data), table_name)
    execute_values(cur, sql, data, page_size=page_size)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    start_time = time.time()
    logger.info("SalonIQ data generator starting")

    conn = connect_with_retry()
    conn.autocommit = False
    cur = conn.cursor()

    try:
        if data_exists(cur):
            logger.info("Data already exists, skipping generation")
            cur.close()
            conn.close()
            return

        # -- Services --
        _batch_insert(cur, INSERT_SERVICES, SERVICE_CATALOG, "services")

        # -- Salons --
        salons = generate_salons()
        _batch_insert(cur, INSERT_SALONS, salons, "salons")

        # -- Stylists --
        stylists = generate_stylists(salons)
        _batch_insert(cur, INSERT_STYLISTS, stylists, "stylists")

        # Build salon -> stylist_id list map
        salon_stylist_map: dict[str, list[str]] = {}
        for sty_id, sal_id, _, _, _ in stylists:
            salon_stylist_map.setdefault(sal_id, []).append(sty_id)

        # -- Customers --
        customers = generate_customers(salons)
        customers_db = [row[:-1] for row in customers]  # strip archetype for DB insert
        _batch_insert(cur, INSERT_CUSTOMERS, customers_db, "customers")

        # -- Visits, visit_services, bookings, purchases --
        visits, visit_services, bookings, purchases = generate_visits_and_related(
            customers, salon_stylist_map,
        )
        _batch_insert(cur, INSERT_VISITS, visits, "visits")
        _batch_insert(cur, INSERT_VISIT_SERVICES, visit_services, "visit_services")
        _batch_insert(cur, INSERT_BOOKINGS, bookings, "bookings")
        _batch_insert(cur, INSERT_PURCHASES, purchases, "product_purchases")

        # -- Case study metrics --
        _batch_insert(cur, INSERT_METRICS, CASE_STUDY_METRICS, "case_study_metrics")

        conn.commit()
        elapsed = time.time() - start_time
        logger.info(
            "Data generation complete -- %d salons, %d stylists, %d customers, "
            "%d visits, %d bookings, %d purchases in %.1f seconds",
            len(salons), len(stylists), len(customers),
            len(visits), len(bookings), len(purchases), elapsed,
        )

    except Exception:
        logger.exception("Data generation failed, rolling back transaction")
        conn.rollback()
        sys.exit(1)
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
