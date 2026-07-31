"""
Author : Mohit Patle

Description:
Streamlit UI for the churn model. Two modes:
1. Single customer — a form that scores one customer and explains the
   risk band.
2. Batch scoring — upload a CSV in the raw export schema and download
   the scored, ranked list for the retention team.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make src/ importable when the app is launched from anywhere.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference import load_artifact, predict_batch, predict_single  # noqa: E402

RISK_COLORS = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}


@st.cache_resource
def get_artifact() -> dict:
    """Load the serialized model artifact once per session."""
    return load_artifact()


def single_customer_form(artifact: dict) -> None:
    """Render the single-customer scoring form."""
    st.subheader("Score a single customer")

    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Has partner", ["No", "Yes"])
        dependents = st.selectbox("Has dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
    with col2:
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multiple = st.selectbox("Multiple lines", ["No", "Yes"])
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        security = st.selectbox("Online security", ["No", "Yes"])
        backup = st.selectbox("Online backup", ["No", "Yes"])
        protection = st.selectbox("Device protection", ["No", "Yes"])
    with col3:
        support = st.selectbox("Tech support", ["No", "Yes"])
        tv = st.selectbox("Streaming TV", ["No", "Yes"])
        movies = st.selectbox("Streaming movies", ["No", "Yes"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly = st.number_input("Monthly charges ($)", 0.0, 200.0, 70.0, step=5.0)

    total = st.number_input(
        "Total charges ($) — defaults to tenure × monthly",
        min_value=0.0,
        value=float(round(tenure * monthly, 2)),
        step=10.0,
    )

    if st.button("Predict churn risk", type="primary"):
        record = {
            "Gender": gender, "Senior Citizen": senior, "Partner": partner,
            "Dependents": dependents, "Tenure Months": tenure,
            "Phone Service": phone, "Multiple Lines": multiple,
            "Internet Service": internet, "Online Security": security,
            "Online Backup": backup, "Device Protection": protection,
            "Tech Support": support, "Streaming TV": tv,
            "Streaming Movies": movies, "Contract": contract,
            "Paperless Billing": paperless, "Payment Method": payment,
            "Monthly Charges": monthly, "Total Charges": total,
        }
        result = predict_single(record, artifact)
        proba = result["churn_probability"]
        band = result["risk_band"]

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Churn probability", f"{proba:.1%}")
        m2.metric("Risk band", f"{RISK_COLORS[band]} {band}")
        m3.metric(
            "Decision",
            "At risk" if result["churn_prediction"] else "Likely to stay",
            help=f"Flagged when probability ≥ tuned threshold {result['threshold']:.2f}",
        )
        st.progress(min(proba, 1.0))

        if band == "High":
            st.warning("Recommend immediate retention action: contract upgrade "
                       "offer or targeted discount.")
        elif band == "Medium":
            st.info("Recommend proactive check-in or a service add-on offer.")
        else:
            st.success("No action needed — customer profile looks stable.")


def batch_scoring(artifact: dict) -> None:
    """Render the batch CSV scoring panel."""
    st.subheader("Score a customer file")
    st.caption(
        "Upload a CSV using the raw export schema (same column names as "
        "the IBM Telco file, e.g. 'Tenure Months', 'Monthly Charges')."
    )

    uploaded = st.file_uploader("Customer CSV", type="csv")
    if uploaded is None:
        return

    try:
        df = pd.read_csv(uploaded)
        scored = predict_batch(df, artifact)
    except (ValueError, KeyError) as exc:
        st.error(f"Could not score this file: {exc}")
        return

    flagged = int(scored["churn_prediction"].sum())
    st.metric("Customers flagged as churn risk", f"{flagged} / {len(scored)}")
    st.dataframe(
        scored[["churn_probability", "risk_band", "churn_prediction"]]
        .join(scored.drop(columns=["churn_probability", "risk_band", "churn_prediction"]))
        .head(200),
        use_container_width=True,
    )
    st.download_button(
        "Download scored file",
        scored.to_csv(index=False).encode("utf-8"),
        file_name="scored_customers.csv",
        mime="text/csv",
    )


def main() -> None:
    """App entry point."""
    st.set_page_config(page_title="Churn Prediction", page_icon="📉", layout="wide")
    st.title("📉 Customer Churn Prediction")

    try:
        artifact = get_artifact()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    metrics = artifact["metrics"]["test_at_tuned_threshold"]
    st.caption(
        f"Model: **{artifact['model_name']}** · trained {artifact['trained_at']} · "
        f"test ROC-AUC **{metrics['roc_auc']:.3f}** · "
        f"recall **{metrics['recall']:.1%}** at threshold {artifact['threshold']:.2f}"
    )

    tab1, tab2 = st.tabs(["Single customer", "Batch scoring"])
    with tab1:
        single_customer_form(artifact)
    with tab2:
        batch_scoring(artifact)


if __name__ == "__main__":
    main()
