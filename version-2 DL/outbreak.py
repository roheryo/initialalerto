# ============================================================
# DENGUE OUTBREAK EARLY-WARNING SYSTEM
# Bi-LSTM + Attention (PERCENTILE-BASED OUTBREAKS)
# Province: Davao de Oro
# ============================================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    r2_score,
    f1_score
)
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Bidirectional, Dropout, Layer
)
import tensorflow.keras.backend as K

import joblib

# ============================================================
# STEP 1: LOAD DATA
# ============================================================

FILE_PATH = "Davao_de_Oro_Cases_2025-Generated-2.csv"
df = pd.read_csv(FILE_PATH)
df["date"] = pd.to_datetime(df["create_date"])
df["municipality"] = df["current_address_city"]
df["barangay"] = df["current_address_barangay"]

# ============================================================
# STEP 2: WEEKLY AGGREGATION
# ============================================================

df["year_week"] = df["date"].dt.to_period("W").astype(str)

df["positive"] = df["lab_result"].str.lower().isin(
    ["positive", "confirmed"]
).astype(int)

weekly = (
    df.groupby(["municipality", "barangay", "year_week"])
      .agg(confirmed_cases=("positive", "sum"))
      .reset_index()
)

# ============================================================
# STEP 3: LOCATION ENCODING
# ============================================================

weekly["location_id"] = (
    weekly["municipality"] + "_" + weekly["barangay"]
).astype("category").cat.codes

# ============================================================
# STEP 4: TIME FEATURES
# ============================================================

weekly["year"] = weekly["year_week"].str[:4].astype(int)
weekly["week"] = weekly["year_week"].str[-2:].astype(int)

# ============================================================
# STEP 5: PERCENTILE-BASED OUTBREAK LABEL (CRITICAL FIX)
# ============================================================

weekly["outbreak"] = 0

for loc in weekly["location_id"].unique():
    loc_df = weekly[weekly["location_id"] == loc]
    threshold = loc_df["confirmed_cases"].quantile(0.90)

    weekly.loc[
        (weekly["location_id"] == loc) &
        (weekly["confirmed_cases"] >= threshold),
        "outbreak"
    ] = 1

print("Outbreak distribution:")
print(weekly["outbreak"].value_counts())

# ============================================================
# STEP 6: LAG & ROLLING FEATURES
# ============================================================

weekly = weekly.sort_values(
    ["location_id", "year", "week"]
)

for lag in [1, 2, 3]:
    weekly[f"lag_{lag}"] = (
        weekly.groupby("location_id")["confirmed_cases"]
              .shift(lag)
    )

weekly["rolling_mean_4"] = (
    weekly.groupby("location_id")["confirmed_cases"]
          .rolling(4).mean()
          .reset_index(level=0, drop=True)
)

weekly["rolling_std_4"] = (
    weekly.groupby("location_id")["confirmed_cases"]
          .rolling(4).std()
          .reset_index(level=0, drop=True)
)

weekly = weekly.dropna().reset_index(drop=True)

# ============================================================
# STEP 7: FEATURE SCALING
# ============================================================

feature_cols = [
    "confirmed_cases",
    "lag_1", "lag_2", "lag_3",
    "rolling_mean_4", "rolling_std_4",
    "week"
]

scaler = MinMaxScaler()
weekly[feature_cols] = scaler.fit_transform(weekly[feature_cols])

# ============================================================
# STEP 8: CREATE LSTM SEQUENCES
# ============================================================

TIME_STEPS = 4

X, y_cases, y_outbreak = [], [], []

for loc in weekly["location_id"].unique():
    loc_df = weekly[weekly["location_id"] == loc]

    for i in range(TIME_STEPS, len(loc_df)):
        X.append(loc_df.iloc[i - TIME_STEPS:i][feature_cols].values)
        y_cases.append(loc_df.iloc[i]["confirmed_cases"])
        y_outbreak.append(loc_df.iloc[i]["outbreak"])

X = np.array(X)
y_cases = np.array(y_cases)
y_outbreak = np.array(y_outbreak)

print("Input shape:", X.shape)
print("Total outbreak samples:", np.sum(y_outbreak))

# ============================================================
# STEP 9: TIME-AWARE SPLIT (SAFE)
# ============================================================

outbreak_indices = np.where(y_outbreak == 1)[0]

test_start = outbreak_indices[int(0.7 * len(outbreak_indices))]

X_train, X_test = X[:test_start], X[test_start:]
y_cases_train, y_cases_test = y_cases[:test_start], y_cases[test_start:]
y_outbreak_train, y_outbreak_test = y_outbreak[:test_start], y_outbreak[test_start:]

print("Train outbreaks:", np.sum(y_outbreak_train))
print("Test outbreaks:", np.sum(y_outbreak_test))

# ============================================================
# STEP 10: CLASS WEIGHTS
# ============================================================

classes = np.unique(y_outbreak_train)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_outbreak_train
)
class_weight = dict(zip(classes, weights))
print("Class weights:", class_weight)

# ============================================================
# STEP 11: ATTENTION LAYER
# ============================================================

class Attention(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="normal"
        )
        self.b = self.add_weight(
            shape=(input_shape[1], 1),
            initializer="zeros"
        )

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        return K.sum(x * a, axis=1)

# ============================================================
# STEP 12: Bi-LSTM + ATTENTION MODEL
# ============================================================

inputs = Input(shape=(TIME_STEPS, X.shape[2]))

x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
x = Dropout(0.3)(x)

x = Bidirectional(LSTM(32, return_sequences=True))(x)
x = Dropout(0.3)(x)

x = Attention()(x)

cases_output = Dense(1, name="cases")(x)
outbreak_output = Dense(1, activation="sigmoid", name="outbreak")(x)

model = Model(inputs, [cases_output, outbreak_output])

model.compile(
    optimizer="adam",
    loss={
        "cases": "mse",
        "outbreak": "binary_crossentropy"
    },
    metrics={
        "cases": "mae",
        "outbreak": "accuracy"
    }
)

model.summary()

# ============================================================
# CREATE SAMPLE WEIGHTS (FOR OUTBREAK ONLY)
# ============================================================

sample_weight_outbreak = np.ones(len(y_outbreak_train))

# Increase weight for outbreak samples
sample_weight_outbreak[y_outbreak_train == 1] = class_weight[1]
sample_weight_outbreak[y_outbreak_train == 0] = class_weight[0]

# Regression task has uniform weights
sample_weight_cases = np.ones(len(y_cases_train))


# ============================================================
# STEP 13: TRAIN MODEL
# ============================================================

model.fit(
    X_train,
    [y_cases_train, y_outbreak_train],
    sample_weight=[sample_weight_cases, sample_weight_outbreak],
    validation_data=(
        X_test,
        [y_cases_test, y_outbreak_test]
    ),
    epochs=50,
    batch_size=32,
    verbose=1
)



# ============================================================
# STEP 14: EVALUATION (SAFE & FINAL)
# ============================================================

# Predict on test set (MUST come first)
pred_cases_test, pred_outbreak_test = model.predict(X_test)

threshold = 0.3
pred_outbreak_class = (pred_outbreak_test > threshold).astype(int)

print("\n--- CASE METRICS ---")
print("RMSE:", np.sqrt(mean_squared_error(y_cases_test, pred_cases_test)))
print("MAE:", mean_absolute_error(y_cases_test, pred_cases_test))

print("\n--- OUTBREAK METRICS ---")
print("Accuracy:", accuracy_score(y_outbreak_test, pred_outbreak_class))
print("Precision:", precision_score(y_outbreak_test, pred_outbreak_class, zero_division=0))
print("Recall:", recall_score(y_outbreak_test, pred_outbreak_class, zero_division=0))
print("F1:", f1_score(y_outbreak_test, pred_outbreak_class, zero_division=0))


# ============================================================
# STEP 15: SAVE MODEL
# ============================================================

model.save("davao_bilstm_attention_percentile_outbreak.h5")
joblib.dump(scaler, "case_scaler.pkl")

print("\nModel and scaler saved successfully.")

# ============================================================
# STEP 16: WHERE & HOW MANY CASES (NEXT-WEEK FORECAST)
# ============================================================

# 1. Collect latest sequences per location
latest_sequences = []
location_ids = []

for loc in weekly["location_id"].unique():
    loc_df = weekly[weekly["location_id"] == loc]
    if len(loc_df) >= TIME_STEPS:
        latest_sequences.append(
            loc_df.iloc[-TIME_STEPS:][feature_cols].values
        )
        location_ids.append(loc)

latest_sequences = np.array(latest_sequences)

# 2. Predict (scaled values)
pred_cases_forecast, pred_outbreak_forecast = model.predict(latest_sequences)


# 3. Build forecast DataFrame (THIS WAS MISSING)
forecast_df = pd.DataFrame({
    "location_id": location_ids,
    "predicted_cases_scaled": pred_cases_forecast.flatten(),
    "outbreak_probability": pred_outbreak_forecast.flatten()
})


# 4. Outbreak decision
THRESHOLD = 0.3
forecast_df["outbreak_predicted"] = (
    forecast_df["outbreak_probability"] >= THRESHOLD
).astype(int)

# 5. Attach location names
location_lookup = weekly[
    ["location_id", "municipality", "barangay"]
].drop_duplicates()

forecast_df = forecast_df.merge(
    location_lookup,
    on="location_id",
    how="left"
)

# ============================================================
# STEP 17: INVERSE SCALE CASE PREDICTIONS (REAL COUNTS)
# ============================================================

# Prepare dummy array for inverse scaling
dummy = np.zeros((len(forecast_df), len(feature_cols)))

# Column 0 = confirmed_cases (our target)
dummy[:, 0] = forecast_df["predicted_cases_scaled"].values

# Inverse transform
real_cases = scaler.inverse_transform(dummy)[:, 0]

forecast_df["predicted_cases"] = np.round(real_cases).astype(int)

# 6. Final, readable output
forecast_df = forecast_df[
    [
        "municipality",
        "barangay",
        "predicted_cases",
        "outbreak_probability",
        "outbreak_predicted"
    ]
].sort_values(
    ["outbreak_predicted", "outbreak_probability"],
    ascending=[False, False]
)

print("\n--- NEXT WEEK OUTBREAK FORECAST (REAL CASE COUNTS) ---")
print(forecast_df.head(20))

# ============================================================
# STEP 18: OVERALL MODEL PERFORMANCE (SAFE & FINAL)
# ============================================================

from sklearn.metrics import roc_auc_score, average_precision_score, r2_score

# ---- Outbreak (classification) ----
roc_auc = roc_auc_score(y_outbreak_test, pred_outbreak_test)
pr_auc = average_precision_score(y_outbreak_test, pred_outbreak_test)

# ---- Cases (regression) ----
r2 = r2_score(y_cases_test, pred_cases_test)

print("\n--- OVERALL MODEL PERFORMANCE ---")
print(f"ROC-AUC (Outbreak): {roc_auc:.3f}")
print(f"PR-AUC  (Outbreak): {pr_auc:.3f}")
print(f"R²      (Cases):    {r2:.3f}")
