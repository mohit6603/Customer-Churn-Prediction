"""
Author : Mohit Patle

Description:
Request/response contracts for the churn API. Pydantic validates every
payload before it reaches the model, so a typo like "Month to month"
is rejected with a clear 422 instead of being silently one-hot encoded
as an unseen category.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

YesNo = Literal["Yes", "No"]

# API field name -> column name in the raw IBM export schema.
RAW_COLUMN_NAMES: dict[str, str] = {
    "customer_id": "CustomerID",
    "gender": "Gender",
    "senior_citizen": "Senior Citizen",
    "partner": "Partner",
    "dependents": "Dependents",
    "tenure_months": "Tenure Months",
    "phone_service": "Phone Service",
    "multiple_lines": "Multiple Lines",
    "internet_service": "Internet Service",
    "online_security": "Online Security",
    "online_backup": "Online Backup",
    "device_protection": "Device Protection",
    "tech_support": "Tech Support",
    "streaming_tv": "Streaming TV",
    "streaming_movies": "Streaming Movies",
    "contract": "Contract",
    "paperless_billing": "Paperless Billing",
    "payment_method": "Payment Method",
    "monthly_charges": "Monthly Charges",
    "total_charges": "Total Charges",
}


class CustomerIn(BaseModel):
    """One customer, described with the fields the model was trained on."""

    customer_id: str | None = Field(default=None, examples=["7590-VHVEG"])
    gender: Literal["Male", "Female"]
    senior_citizen: YesNo
    partner: YesNo
    dependents: YesNo
    tenure_months: int = Field(ge=0, le=120, examples=[2])
    phone_service: YesNo
    multiple_lines: Literal["Yes", "No", "No phone service"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    online_security: Literal["Yes", "No", "No internet service"]
    online_backup: Literal["Yes", "No", "No internet service"]
    device_protection: Literal["Yes", "No", "No internet service"]
    tech_support: Literal["Yes", "No", "No internet service"]
    streaming_tv: Literal["Yes", "No", "No internet service"]
    streaming_movies: Literal["Yes", "No", "No internet service"]
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: YesNo
    payment_method: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ]
    monthly_charges: float = Field(ge=0, examples=[70.7])
    total_charges: float | None = Field(
        default=None, ge=0,
        description="Defaults to tenure_months × monthly_charges when omitted.",
    )

    @model_validator(mode="after")
    def _default_total_charges(self) -> CustomerIn:
        if self.total_charges is None:
            self.total_charges = round(self.tenure_months * self.monthly_charges, 2)
        return self

    def to_raw_record(self) -> dict[str, object]:
        """Convert to the raw export schema expected by src.inference."""
        return {RAW_COLUMN_NAMES[k]: v for k, v in self.model_dump().items()}

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "customer_id": "7590-VHVEG",
                "gender": "Female", "senior_citizen": "No", "partner": "Yes",
                "dependents": "No", "tenure_months": 2, "phone_service": "Yes",
                "multiple_lines": "No", "internet_service": "Fiber optic",
                "online_security": "No", "online_backup": "No",
                "device_protection": "No", "tech_support": "No",
                "streaming_tv": "Yes", "streaming_movies": "Yes",
                "contract": "Month-to-month", "paperless_billing": "Yes",
                "payment_method": "Electronic check", "monthly_charges": 89.1,
            }]
        }
    }


class PredictionOut(BaseModel):
    """Score and recommended action for one customer."""

    customer_id: str | None
    churn_probability: float
    churn_prediction: int
    risk_band: Literal["Low", "Medium", "High"]
    recommended_action: str
    threshold: float
    model_version: str


class BatchPredictionOut(BaseModel):
    """Ranked scores for an uploaded customer file."""

    model_version: str
    n_customers: int
    n_flagged: int
    predictions: list[PredictionOut]


class HealthOut(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    model_version: str


class ModelInfoOut(BaseModel):
    model_name: str
    model_version: str
    trained_at: str
    threshold: float
    cv_roc_auc: float | None
    test_metrics: dict[str, float]
    input_columns: list[str]


class DriftOut(BaseModel):
    n_samples: int
    drift_detected: bool
    retraining_recommended: bool
    significant_features: list[str]
    moderate_features: list[str]
    features: list[dict[str, object]]
