"""
Week 9 - Train the model used by the Streamlit app.

The app only asks for four inputs (LivingArea, Beds, Baths, LotSize),
so this script trains a Gradient Boosting model on just those four
features and saves it with joblib.

Run from the repo root:
    python train_model.py
Output:
    model.joblib
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

DATA_PATH = Path("notebooks/cleaned_california_single_family_sales.csv")
MODEL_PATH = Path("model.joblib")

FEATURES = ["LivingArea", "BedroomsTotal", "BathroomsTotalInteger", "LotSizeSquareFeet"]
TARGET = "ClosePrice"

# Load data and make sure every column is numeric.
df = pd.read_csv(DATA_PATH, low_memory=False)
for column in FEATURES + [TARGET]:
    df[column] = pd.to_numeric(df[column], errors="coerce")
df = df[df[TARGET] > 0].copy()

# Use the same train/test split as the cleaned dataset.
train = df[df["split"] == "train"].copy()
test = df[df["split"] == "test"].copy()

# Drop extreme outliers from training only (e.g. a $142M sale, 153 baths).
# They are rare and pull the model away from typical homes.
low, high = train[TARGET].quantile([0.01, 0.99])
train = train[train[TARGET].between(low, high)]
for column in ["LivingArea", "LotSizeSquareFeet"]:
    train = train[train[column].isna() | (train[column] <= train[column].quantile(0.99))]

# Missing inputs are median-imputed. The target is modeled on a log scale
# because home prices are heavily right-skewed.
model = TransformedTargetRegressor(
    regressor=Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
        )),
    ]),
    func=np.log1p,
    inverse_func=np.expm1,
)

model.fit(train[FEATURES], train[TARGET])

# Evaluate on the held-out test set.
predictions = model.predict(test[FEATURES])
errors = np.abs(predictions - test[TARGET]) / test[TARGET]
metrics = {
    "R2": r2_score(test[TARGET], predictions),
    "MAE": mean_absolute_error(test[TARGET], predictions),
    "MdAPE": float(np.median(errors) * 100),
}
print(f"Training rows: {len(train):,}   Test rows: {len(test):,}")
print(f"R²: {metrics['R2']:.3f}   MAE: ${metrics['MAE']:,.0f}   MdAPE: {metrics['MdAPE']:.1f}%")

# Save the model plus a little metadata for the app.
joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "metrics": metrics,
        "train_medians": train[FEATURES].median().to_dict(),
    },
    MODEL_PATH,
)
print(f"Saved {MODEL_PATH}")