from pydantic import BaseModel


class RiskFactor(BaseModel):
    """Individual risk factor contributing to churn prediction."""

    feature: str
    description: str
    impact: float


class CustomerPrediction(BaseModel):
    """Full prediction payload for a single customer."""

    customer_id: str
    salon_id: str
    customer_name: str
    churn_probability_90d: float
    risk_level: str
    risk_factors: list[RiskFactor]
    predicted_next_visit_days: int
    estimated_ltv_12m: float
    upsell_propensity: float
    model_version: str
    predicted_at: str


class AtRiskCustomer(BaseModel):
    """Summarized at-risk customer record."""

    customer_id: str
    customer_name: str
    churn_probability: float
    risk_level: str
    days_since_last_visit: int
    avg_inter_visit_days: float
    primary_risk_factor: str
    suggested_action: str
    estimated_ltv_12m: float


class SalonInfo(BaseModel):
    """Basic salon metadata with customer count."""

    salon_id: str
    salon_name: str
    region: str
    city: str
    customer_count: int


class DashboardKPIs(BaseModel):
    """Key performance indicators for the salon dashboard."""

    total_active_customers: int
    at_risk_count: int
    at_risk_percentage: float
    churn_rate_trailing_90d: float
    avg_inter_visit_days: float
    retention_rate_trailing_90d: float
    avg_ticket_value: float
    total_revenue_30d: float


class DashboardResponse(BaseModel):
    """Complete dashboard response for a single salon."""

    salon_id: str
    salon_name: str
    period: str
    kpis: DashboardKPIs
    risk_distribution: dict
    top_churn_drivers: list[dict]


class CaseStudyResponse(BaseModel):
    """Before/after case study comparison for a salon."""

    salon_id: str
    salon_name: str
    before: dict
    after: dict
    improvements: list[dict]


class VisitRecord(BaseModel):
    """Single visit history entry."""

    visit_date: str
    services: str
    amount: float
    stylist: str
