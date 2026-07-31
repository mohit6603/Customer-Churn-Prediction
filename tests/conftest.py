"""
Author : Mohit Patle

Description:
Shared pytest fixtures. Tests run against a small synthetic dataframe
that mimics the raw 33-column IBM export, so the suite never depends
on the real (gitignored) Excel file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config  # noqa: E402


@pytest.fixture(scope="session")
def config() -> dict:
    """Project configuration loaded from configs/config.yaml."""
    return load_config()


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    """Eight synthetic customers in the raw 33-column export schema."""
    n = 8
    return pd.DataFrame(
        {
            "CustomerID": [f"C{i:04d}" for i in range(n)],
            "Count": 1,
            "Country": "United States",
            "State": "California",
            "City": "Los Angeles",
            "Zip Code": 90001,
            "Lat Long": "33.96, -118.24",
            "Latitude": 33.96,
            "Longitude": -118.24,
            "Gender": ["Male", "Female"] * 4,
            "Senior Citizen": "No",
            "Partner": "No",
            "Dependents": "No",
            "Tenure Months": [0, 1, 5, 12, 24, 40, 60, 72],
            "Phone Service": ["No", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes"],
            "Multiple Lines": ["No phone service", "No", "Yes", "No",
                               "Yes", "No", "Yes", "No"],
            "Internet Service": ["DSL", "Fiber optic", "No", "DSL",
                                 "Fiber optic", "DSL", "No", "DSL"],
            "Online Security": ["Yes", "No", "No internet service", "Yes",
                                "No", "Yes", "No internet service", "No"],
            "Online Backup": ["Yes", "No", "No internet service", "No",
                              "No", "No", "No internet service", "No"],
            "Device Protection": "No",
            "Tech Support": "No",
            "Streaming TV": "No",
            "Streaming Movies": "No",
            "Contract": ["Month-to-month", "One year", "Two year",
                         "Month-to-month"] * 2,
            "Paperless Billing": "Yes",
            "Payment Method": "Electronic check",
            "Monthly Charges": [50.0, 70.5, 20.0, 55.0, 80.0, 65.0, 25.0, 90.0],
            "Total Charges": [" ", "70.5", "100.0", "660.0",
                              "1920.0", "2600.0", "1500.0", "6480.0"],
            "Churn Label": ["Yes", "No"] * 4,
            "Churn Value": [1, 0] * 4,
            "Churn Score": 50,
            "CLTV": 4000,
            "Churn Reason": [None] * n,
        }
    )
