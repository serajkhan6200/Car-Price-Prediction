"""Train and save the Car Price Predictor model and preprocessing artifacts."""

from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "Dataset" / "1.04. Real-life example.csv"


# ============================================================
# 2. LOAD LOCAL DATASET
# ============================================================

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at:\n{DATASET_PATH}\n\n"
        "Please make sure the CSV is inside the Dataset folder."
    )

print(f"Loading dataset from:\n{DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)

print(f"\nDataset loaded successfully.")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


# ============================================================
# 3. BASIC DATA CHECK
# ============================================================

required_columns = [
    "Brand",
    "Price",
    "Body",
    "Mileage",
    "EngineV",
    "Engine Type",
    "Registration",
    "Year",
    "Model",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


print("\nOriginal columns:")
print(df.columns.tolist())


# ============================================================
# 4. DATA CLEANING
# ============================================================

# Remove rows where target or engine volume is missing.
df = df.dropna(
    subset=["Price", "EngineV"]
).copy()

# Price must be positive because log(Price) is used.
df = df[df["Price"] > 0].copy()

# Remove unrealistic engine-volume values.
df = df[df["EngineV"] <= 10].copy()


print(f"\nRows after cleaning: {len(df)}")


# ============================================================
# 5. LOG TRANSFORMATION OF TARGET
# ============================================================

df["Log_price"] = np.log(df["Price"])


# ============================================================
# 6. ENCODE CATEGORICAL FEATURES
# ============================================================

cat_cols = df.select_dtypes(
    include="object"
).columns.tolist()

encoders = {}

print("\nCategorical columns:")
print(cat_cols)

for col in cat_cols:
    encoder = LabelEncoder()

    df[col] = encoder.fit_transform(
        df[col].astype(str)
    )

    encoders[col] = encoder


# ============================================================
# 7. FEATURES AND TARGET
# ============================================================

# IMPORTANT:
# Price and Log_price must NOT be included in X.
# Otherwise target leakage occurs.

X = df.drop(
    columns=["Price", "Log_price"]
)

y = df["Log_price"]

feature_columns = X.columns.tolist()


print("\nFeatures used by model:")
print(feature_columns)

print("\nTarget:")
print("Log_price")


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTrain samples:", len(X_train))
print("Test samples :", len(X_test))


# ============================================================
# 9. DEFINE MODELS
# ============================================================

models = {
    "Linear Regression": LinearRegression(),

    "Ridge Regression": Ridge(
        alpha=1.0
    ),

    "Lasso Regression": Lasso(
        alpha=1.0,
        random_state=42
    ),
}


# ============================================================
# 10. TRAIN AND EVALUATE MODELS
# ============================================================

results = []
fitted = {}

for name, estimator in models.items():

    print(f"\nTraining {name}...")

    estimator.fit(
        X_train,
        y_train
    )

    preds = estimator.predict(
        X_test
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_test,
                preds
            )
        )
    )

    r2 = float(
        r2_score(
            y_test,
            preds
        )
    )

    results.append({
        "Model": name,
        "RMSE": rmse,
        "R2": r2,
    })

    fitted[name] = estimator

    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")


# ============================================================
# 11. MODEL COMPARISON
# ============================================================

results_df = (
    pd.DataFrame(results)
    .sort_values("RMSE")
    .reset_index(drop=True)
)

print("\n========================================")
print("MODEL COMPARISON")
print("========================================")
print(
    results_df.to_string(index=False)
)

best_name = results_df.iloc[0]["Model"]

print("\nSelected model:")
print(best_name)


# ============================================================
# 12. SAVE TRAINED MODEL
# ============================================================

joblib.dump(
    fitted[best_name],
    BASE_DIR / "car_price_model.pkl"
)


# ============================================================
# 13. SAVE ENCODERS
# ============================================================

joblib.dump(
    encoders,
    BASE_DIR / "encoders.pkl"
)


# ============================================================
# 14. SAVE CATEGORICAL COLUMN LIST
# ============================================================

joblib.dump(
    cat_cols,
    BASE_DIR / "cat_cols.pkl"
)


# ============================================================
# 15. SAVE FEATURE COLUMN ORDER
# ============================================================

joblib.dump(
    feature_columns,
    BASE_DIR / "feature_columns.pkl"
)


# ============================================================
# 16. SAVE NUMERIC RANGES
# ============================================================

numeric_cols = [
    col
    for col in feature_columns
    if col not in cat_cols
]

ranges = {
    col: (
        float(df[col].min()),
        float(df[col].max())
    )
    for col in numeric_cols
}

joblib.dump(
    ranges,
    BASE_DIR / "numeric_ranges.pkl"
)


# ============================================================
# 17. SAVE MODEL RESULTS
# ============================================================

results_df.to_csv(
    BASE_DIR / "model_results.csv",
    index=False
)


# ============================================================
# 18. FINISHED
# ============================================================

print("\n========================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("========================================")

print("\nSaved files:")

print("✓ car_price_model.pkl")
print("✓ encoders.pkl")
print("✓ cat_cols.pkl")
print("✓ feature_columns.pkl")
print("✓ numeric_ranges.pkl")
print("✓ model_results.csv")

print("\nBest model:", best_name)