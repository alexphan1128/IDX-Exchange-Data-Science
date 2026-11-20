"""
Week 9 - Simple Prediction App (Streamlit)

Enter LivingArea, Beds, Baths and LotSize to get a predicted
closing price for a California single-family home.

Run from the repo root:
    python train_model.py      # creates model.joblib (once)
    streamlit run app.py
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "model.joblib"

st.set_page_config(page_title="CA Home Price Predictor", page_icon="🏠", layout="centered")


@st.cache_resource
def load_model():
    """Load the trained model once and reuse it across reruns."""
    return joblib.load(MODEL_PATH)


st.title("🏠 California Home Price Predictor")
st.write(
    "Estimate the closing price of a California single-family home "
    "from its size. Model: Gradient Boosting trained on CRMLS sold listings."
)

if not MODEL_PATH.exists():
    st.error("`model.joblib` not found. Run `python train_model.py` first, then reload this page.")
    st.stop()

bundle = load_model()
model = bundle["model"]
features = bundle["features"]
metrics = bundle["metrics"]

# ---- Inputs ---------------------------------------------------------------
with st.form("inputs"):
    col1, col2 = st.columns(2)
    with col1:
        living_area = st.number_input(
            "Living area (sq ft)", min_value=200, max_value=20000, value=1800, step=50
        )
        beds = st.number_input("Bedrooms", min_value=0, max_value=15, value=3, step=1)
    with col2:
        lot_size = st.number_input(
            "Lot size (sq ft)", min_value=0, max_value=500000, value=6000, step=100
        )
        baths = st.number_input("Bathrooms", min_value=0, max_value=15, value=2, step=1)

    submitted = st.form_submit_button("Predict price", type="primary", use_container_width=True)

# ---- Prediction -----------------------------------------------------------
if submitted:
    # Column names must match the ones the model was trained on.
    row = pd.DataFrame(
        [[living_area, beds, baths, lot_size]],
        columns=features,  # LivingArea, BedroomsTotal, BathroomsTotalInteger, LotSizeSquareFeet
    )
    price = float(model.predict(row)[0])

    st.metric("Predicted closing price", f"${price:,.0f}")
    st.caption(f"≈ ${price / living_area:,.0f} per sq ft")

    # Typical error range, based on the median % error on the test set.
    mdape = metrics["MdAPE"] / 100
    st.write(
        f"A typical range for this estimate is **${price * (1 - mdape):,.0f} – "
        f"${price * (1 + mdape):,.0f}**."
    )

# ---- Model info -----------------------------------------------------------
with st.expander("About this model"):
    st.write(
        f"""
- **Inputs:** living area, bedrooms, bathrooms, lot size
- **Test-set performance:** R² = {metrics['R2']:.2f}, MAE = ${metrics['MAE']:,.0f},
  median error = {metrics['MdAPE']:.0f}%
- **Limitation:** location isn't an input, so the same house gets the same price
  in Fresno and in Palo Alto. Treat the result as a statewide ballpark.
"""
    )