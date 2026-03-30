-- SalonIQ Customer Insight Engine -- Database Schema
-- Masked project: SalonIQ (branch: feature/saloniq-customer-insight-churn-engine)

-- Core Entities

CREATE TABLE salons (
    salon_id VARCHAR(20) PRIMARY KEY,
    salon_name VARCHAR(100) NOT NULL,
    region VARCHAR(50) NOT NULL,
    city VARCHAR(50) NOT NULL,
    stylist_count INTEGER DEFAULT 0,
    chair_count INTEGER DEFAULT 0,
    subscription_tier VARCHAR(20) DEFAULT 'BASIC'
        CHECK (subscription_tier IN ('SIMPLE', 'BASIC', 'BASIC_PLUS')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE stylists (
    stylist_id VARCHAR(20) PRIMARY KEY,
    salon_id VARCHAR(20) NOT NULL REFERENCES salons(salon_id),
    stylist_name VARCHAR(100) NOT NULL,
    specialty VARCHAR(50),
    experience_years INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE customers (
    customer_id VARCHAR(20) PRIMARY KEY,
    salon_id VARCHAR(20) NOT NULL REFERENCES salons(salon_id),
    customer_name VARCHAR(100) NOT NULL,
    gender VARCHAR(10) CHECK (gender IN ('M', 'F', 'OTHER')),
    age_group VARCHAR(20),
    first_visit_date DATE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE services (
    service_id VARCHAR(20) PRIMARY KEY,
    service_name VARCHAR(100) NOT NULL,
    category VARCHAR(50) NOT NULL
        CHECK (category IN ('CUT', 'COLOR', 'PERM', 'TREATMENT', 'SPA', 'OTHER')),
    base_price DECIMAL(10,2) NOT NULL
);

CREATE TABLE visits (
    visit_id VARCHAR(30) PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL REFERENCES customers(customer_id),
    salon_id VARCHAR(20) NOT NULL REFERENCES salons(salon_id),
    stylist_id VARCHAR(20) REFERENCES stylists(stylist_id),
    visit_date TIMESTAMP NOT NULL,
    visit_type VARCHAR(20) DEFAULT 'APPOINTMENT'
        CHECK (visit_type IN ('APPOINTMENT', 'WALKIN')),
    total_amount DECIMAL(10,2) DEFAULT 0,
    duration_minutes INTEGER,
    status VARCHAR(20) DEFAULT 'COMPLETED'
        CHECK (status IN ('COMPLETED', 'NOSHOW', 'CANCELLED')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE visit_services (
    visit_id VARCHAR(30) NOT NULL REFERENCES visits(visit_id),
    service_id VARCHAR(20) NOT NULL REFERENCES services(service_id),
    price DECIMAL(10,2) NOT NULL,
    PRIMARY KEY (visit_id, service_id)
);

CREATE TABLE product_purchases (
    purchase_id VARCHAR(30) PRIMARY KEY,
    visit_id VARCHAR(30) REFERENCES visits(visit_id),
    customer_id VARCHAR(20) NOT NULL REFERENCES customers(customer_id),
    product_name VARCHAR(100) NOT NULL,
    category VARCHAR(50)
        CHECK (category IN ('HAIRCARE', 'STYLING', 'SKINCARE', 'TOOLS', 'OTHER')),
    price DECIMAL(10,2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE bookings (
    booking_id VARCHAR(30) PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL REFERENCES customers(customer_id),
    salon_id VARCHAR(20) NOT NULL REFERENCES salons(salon_id),
    stylist_id VARCHAR(20) REFERENCES stylists(stylist_id),
    booked_date TIMESTAMP NOT NULL,
    appointment_date TIMESTAMP NOT NULL,
    status VARCHAR(20) DEFAULT 'CONFIRMED'
        CHECK (status IN ('CONFIRMED', 'CANCELLED', 'NOSHOW', 'COMPLETED')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ML Feature Store

CREATE TABLE customer_features (
    salon_id VARCHAR(20) NOT NULL,
    customer_id VARCHAR(20) NOT NULL,
    visit_count_total INTEGER DEFAULT 0,
    visit_count_90d INTEGER DEFAULT 0,
    visit_count_180d INTEGER DEFAULT 0,
    days_since_last_visit INTEGER DEFAULT 0,
    avg_inter_visit_days DOUBLE PRECISION,
    std_inter_visit_days DOUBLE PRECISION,
    inter_visit_trend DOUBLE PRECISION,
    visit_regularity_index DOUBLE PRECISION,
    overdue_ratio DOUBLE PRECISION,
    dow_mode INTEGER,
    hour_mode INTEGER,
    avg_ticket_value DOUBLE PRECISION,
    ticket_value_trend DOUBLE PRECISION,
    max_ticket_value DOUBLE PRECISION,
    service_variety_index DOUBLE PRECISION,
    has_color_service BOOLEAN DEFAULT FALSE,
    has_treatment_service BOOLEAN DEFAULT FALSE,
    color_frequency_ratio DOUBLE PRECISION DEFAULT 0,
    primary_stylist_id VARCHAR(20),
    stylist_consistency_ratio DOUBLE PRECISION,
    stylist_change_count INTEGER DEFAULT 0,
    multi_stylist_flag BOOLEAN DEFAULT FALSE,
    walkin_ratio DOUBLE PRECISION DEFAULT 0,
    noshow_rate DOUBLE PRECISION DEFAULT 0,
    cancellation_rate DOUBLE PRECISION DEFAULT 0,
    avg_booking_lead_days DOUBLE PRECISION,
    product_purchase_count INTEGER DEFAULT 0,
    product_purchase_frequency DOUBLE PRECISION DEFAULT 0,
    product_category_count INTEGER DEFAULT 0,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (salon_id, customer_id)
);

-- Prediction Results

CREATE TABLE predictions (
    salon_id VARCHAR(20) NOT NULL,
    customer_id VARCHAR(20) NOT NULL,
    churn_probability DOUBLE PRECISION,
    risk_level VARCHAR(20)
        CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    predicted_next_visit_days INTEGER,
    estimated_ltv_12m DOUBLE PRECISION,
    upsell_propensity DOUBLE PRECISION,
    risk_factors JSONB,
    model_version VARCHAR(50),
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (salon_id, customer_id)
);

-- Case Study Metrics

CREATE TABLE case_study_metrics (
    metric_id SERIAL PRIMARY KEY,
    salon_id VARCHAR(20),
    period VARCHAR(20) NOT NULL CHECK (period IN ('BEFORE', 'AFTER')),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    metric_unit VARCHAR(50),
    category VARCHAR(50),
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance Indexes

CREATE INDEX idx_visits_customer_date ON visits(customer_id, visit_date);
CREATE INDEX idx_visits_salon_date ON visits(salon_id, visit_date);
CREATE INDEX idx_visits_status ON visits(status);
CREATE INDEX idx_bookings_customer ON bookings(customer_id);
CREATE INDEX idx_bookings_salon ON bookings(salon_id);
CREATE INDEX idx_bookings_status ON bookings(status);
CREATE INDEX idx_customers_salon ON customers(salon_id);
CREATE INDEX idx_stylists_salon ON stylists(salon_id);
CREATE INDEX idx_product_purchases_customer ON product_purchases(customer_id);
CREATE INDEX idx_predictions_risk ON predictions(salon_id, risk_level);
CREATE INDEX idx_customer_features_computed ON customer_features(computed_at);
