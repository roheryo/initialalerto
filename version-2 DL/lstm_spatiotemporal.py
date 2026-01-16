"""
Spatio-Temporal LSTM for Disease Outbreak Prediction
Predicts WHERE (Municipality/Barangay) and HOW MANY (Case Count)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input, 
                                      Bidirectional, BatchNormalization, Concatenate)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import joblib
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

print("="*70)
print("SPATIO-TEMPORAL LSTM FOR OUTBREAK PREDICTION")
print("Predicting WHERE and HOW MANY")
print("="*70)

# ============================================
# 1. LOAD AND PREPARE DATA
# ============================================
print("\n1. LOADING DATA...")

df = pd.read_csv(r'C:\Users\admin\Downloads\version-2 DL\Davao_de_Oro_Cases_2025-Generated-2.csv')
df['create_date'] = pd.to_datetime(df['create_date'])
df['week'] = df['create_date'].dt.isocalendar().week.astype(int)
df['year'] = df['create_date'].dt.year

print(f"Total records: {len(df)}")
print(f"Date range: {df['create_date'].min()} to {df['create_date'].max()}")
print(f"Municipalities: {df['current_address_city'].nunique()}")
print(f"Barangays: {df['current_address_barangay'].nunique()}")

# ============================================
# 2. PIVOT DATA - TIME SERIES BY LOCATION
# ============================================
print("\n" + "="*70)
print("2. PIVOTING DATA - TIME SERIES BY LOCATION")
print("="*70)

# Aggregate by week and municipality
weekly_muni = df.groupby(['year', 'week', 'current_address_city']).agg({
    'type': 'count',
    'AGE': 'mean',
    'current_address_barangay': lambda x: x.value_counts().index[0]  # Top barangay
}).reset_index()
weekly_muni.columns = ['year', 'week', 'municipality', 'cases', 'avg_age', 'top_barangay']

# Create week_id for proper ordering
weekly_muni['week_id'] = weekly_muni['year'] * 100 + weekly_muni['week']
weekly_muni = weekly_muni.sort_values(['week_id', 'municipality'])

# Pivot: Rows=Weeks, Columns=Municipalities, Values=Cases
pivot_cases = weekly_muni.pivot_table(
    index='week_id', 
    columns='municipality', 
    values='cases', 
    aggfunc='sum',
    fill_value=0
)

# Fill missing weeks with 0
all_weeks = range(pivot_cases.index.min(), pivot_cases.index.max() + 1)
pivot_cases = pivot_cases.reindex(all_weeks, fill_value=0)

print(f"\nPivot shape: {pivot_cases.shape}")
print(f"Weeks: {len(pivot_cases)}")
print(f"Municipalities: {pivot_cases.columns.tolist()}")

# Also aggregate by barangay for detailed predictions
weekly_brgy = df.groupby(['year', 'week', 'current_address_city', 'current_address_barangay']).size().reset_index(name='cases')
weekly_brgy['week_id'] = weekly_brgy['year'] * 100 + weekly_brgy['week']

print("\nSample pivot data (first 5 weeks):")
print(pivot_cases.head())

# ============================================
# 3. NORMALIZE DATA
# ============================================
print("\n" + "="*70)
print("3. NORMALIZING DATA (MinMaxScaler)")
print("="*70)

municipalities = pivot_cases.columns.tolist()
scalers = {}

# Scale each municipality separately
pivot_scaled = pd.DataFrame(index=pivot_cases.index, columns=pivot_cases.columns)

for muni in municipalities:
    scaler = MinMaxScaler(feature_range=(0, 1))
    values = pivot_cases[muni].values.reshape(-1, 1)
    scaled = scaler.fit_transform(values)
    pivot_scaled[muni] = scaled.flatten()
    scalers[muni] = scaler

print(f"Scaled {len(municipalities)} municipality time series")
print(f"Scale range: (0, 1)")

# ============================================
# 4. CREATE SLIDING WINDOW SEQUENCES
# ============================================
print("\n" + "="*70)
print("4. CREATING SLIDING WINDOW SEQUENCES")
print("="*70)

LOOK_BACK = 4  # Use past 4 weeks to predict next week
FORECAST_HORIZON = 1  # Predict 1 week ahead

def create_sequences(data, look_back, forecast_horizon=1):
    """Create sequences for LSTM with sliding window"""
    X, y = [], []
    for i in range(len(data) - look_back - forecast_horizon + 1):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back:i + look_back + forecast_horizon])
    return np.array(X), np.array(y)

# Create sequences for all municipalities combined (Multi-Series Approach)
all_data = pivot_scaled.values.astype(float)
X_all, y_all = create_sequences(all_data, LOOK_BACK)

print(f"Look-back window: {LOOK_BACK} weeks")
print(f"Forecast horizon: {FORECAST_HORIZON} week")
print(f"X shape: {X_all.shape} (samples, timesteps, municipalities)")
print(f"y shape: {y_all.shape}")

# Reshape y to 2D
y_all = y_all.reshape(y_all.shape[0], -1)

# Train/Test split (time-based)
train_size = int(len(X_all) * 0.8)
X_train, X_test = X_all[:train_size], X_all[train_size:]
y_train, y_test = y_all[:train_size], y_all[train_size:]

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# ============================================
# 5. BUILD MULTI-OUTPUT LSTM MODEL
# ============================================
print("\n" + "="*70)
print("5. BUILDING MULTI-OUTPUT LSTM MODEL")
print("="*70)

n_features = len(municipalities)  # Number of municipalities

model = Sequential([
    # First LSTM layer
    Bidirectional(LSTM(128, return_sequences=True), 
                  input_shape=(LOOK_BACK, n_features)),
    Dropout(0.3),
    BatchNormalization(),
    
    # Second LSTM layer
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    
    # Third LSTM layer
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    
    # Dense layers
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(32, activation='relu'),
    
    # Output layer - one output per municipality
    Dense(n_features)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
model.summary()

# ============================================
# 6. TRAIN MODEL
# ============================================
print("\n" + "="*70)
print("6. TRAINING MODEL")
print("="*70)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
]

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=16,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.savefig(r'C:\Users\admin\Downloads\version-2 DL\training_history.png', dpi=150)
plt.close()

# ============================================
# 7. EVALUATE MODEL
# ============================================
print("\n" + "="*70)
print("7. MODEL EVALUATION")
print("="*70)

# Predictions
y_pred_scaled = model.predict(X_test)

# Inverse transform predictions for each municipality
y_pred = np.zeros_like(y_pred_scaled)
y_actual = np.zeros_like(y_test)

for i, muni in enumerate(municipalities):
    y_pred[:, i] = scalers[muni].inverse_transform(y_pred_scaled[:, i].reshape(-1, 1)).flatten()
    y_actual[:, i] = scalers[muni].inverse_transform(y_test[:, i].reshape(-1, 1)).flatten()

# Overall metrics
print("\nOverall Model Performance:")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_actual, y_pred)):.2f}")
print(f"  MAE: {mean_absolute_error(y_actual, y_pred):.2f}")
print(f"  R²: {r2_score(y_actual.flatten(), y_pred.flatten()):.4f}")

# Per-municipality metrics
print("\nPer-Municipality Performance:")
print("-" * 60)
muni_metrics = []
for i, muni in enumerate(municipalities):
    rmse = np.sqrt(mean_squared_error(y_actual[:, i], y_pred[:, i]))
    mae = mean_absolute_error(y_actual[:, i], y_pred[:, i])
    r2 = r2_score(y_actual[:, i], y_pred[:, i])
    muni_metrics.append({'municipality': muni, 'rmse': rmse, 'mae': mae, 'r2': r2})
    print(f"  {muni:<20} RMSE: {rmse:6.2f}  MAE: {mae:6.2f}  R²: {r2:.4f}")

# ============================================
# 8. PREDICT UPCOMING WEEK
# ============================================
print("\n" + "="*70)
print("8. PREDICTING UPCOMING WEEK")
print("="*70)

# Use last LOOK_BACK weeks to predict next week
last_sequence = pivot_scaled.values[-LOOK_BACK:].astype(float)
last_sequence = last_sequence.reshape(1, LOOK_BACK, n_features)

next_week_scaled = model.predict(last_sequence, verbose=0)

# Inverse transform
next_week_pred = {}
for i, muni in enumerate(municipalities):
    pred = scalers[muni].inverse_transform(next_week_scaled[0, i].reshape(-1, 1))[0, 0]
    next_week_pred[muni] = max(0, round(pred))  # Ensure non-negative

print("\nPredicted Cases for Next Week by Municipality:")
print("-" * 60)

# Sort by predicted cases
sorted_pred = sorted(next_week_pred.items(), key=lambda x: x[1], reverse=True)
total_pred = sum(next_week_pred.values())

for muni, cases in sorted_pred:
    pct = (cases / total_pred * 100) if total_pred > 0 else 0
    print(f"  {muni:<20} {cases:4} cases ({pct:5.1f}%)")

print(f"\n  {'TOTAL':<20} {total_pred:4} cases")

# ============================================
# 9. OUTBREAK CLASSIFICATION
# ============================================
print("\n" + "="*70)
print("9. OUTBREAK RISK CLASSIFICATION")
print("="*70)

# Calculate thresholds based on historical data
historical_stats = {}
for muni in municipalities:
    muni_data = pivot_cases[muni].values
    historical_stats[muni] = {
        'mean': muni_data.mean(),
        'std': muni_data.std(),
        'max': muni_data.max(),
        'p75': np.percentile(muni_data, 75),
        'p90': np.percentile(muni_data, 90)
    }

def classify_outbreak_risk(predicted_cases, stats):
    """
    Classify outbreak risk based on historical statistics
    """
    mean = stats['mean']
    std = stats['std']
    p75 = stats['p75']
    p90 = stats['p90']
    
    if predicted_cases < mean:
        return 'LOW', 'green'
    elif predicted_cases < p75:
        return 'MODERATE', 'yellow'
    elif predicted_cases < p90:
        return 'HIGH', 'orange'
    else:
        return 'CRITICAL (OUTBREAK)', 'red'

print("\nOutbreak Risk Assessment:")
print("-" * 80)
print(f"{'Municipality':<20} {'Predicted':<10} {'Mean':<8} {'P90':<8} {'Risk Level':<20}")
print("=" * 80)

outbreak_results = []
for muni in municipalities:
    pred = next_week_pred[muni]
    stats = historical_stats[muni]
    risk, color = classify_outbreak_risk(pred, stats)
    outbreak_results.append({
        'municipality': muni,
        'predicted_cases': pred,
        'historical_mean': round(stats['mean'], 1),
        'threshold_p90': round(stats['p90'], 1),
        'risk_level': risk
    })
    print(f"{muni:<20} {pred:<10} {stats['mean']:<8.1f} {stats['p90']:<8.1f} {risk:<20}")

# ============================================
# 10. BARANGAY-LEVEL PREDICTION
# ============================================
print("\n" + "="*70)
print("10. BARANGAY-LEVEL HOTSPOT PREDICTION")
print("="*70)

# For each municipality, identify top barangays
print("\nPredicted Hotspot Barangays (Next Week):")
print("-" * 80)

barangay_predictions = []
for muni, pred_cases in sorted_pred[:5]:  # Top 5 municipalities
    # Get historical barangay distribution for this municipality
    muni_brgy = df[df['current_address_city'] == muni]['current_address_barangay'].value_counts()
    muni_brgy_pct = muni_brgy / muni_brgy.sum()
    
    print(f"\n{muni} ({pred_cases} predicted cases):")
    for brgy, pct in muni_brgy_pct.head(3).items():
        brgy_pred = round(pred_cases * pct)
        risk, _ = classify_outbreak_risk(brgy_pred, {'mean': pred_cases/5, 'std': 2, 'p75': pred_cases/3, 'p90': pred_cases/2})
        print(f"    {brgy:<30} ~{brgy_pred:3} cases ({pct*100:.1f}%) - {risk}")
        barangay_predictions.append({
            'municipality': muni,
            'barangay': brgy,
            'predicted_cases': brgy_pred,
            'percentage': round(pct * 100, 1),
            'risk_level': risk
        })

# ============================================
# 11. VISUALIZATION
# ============================================
print("\n" + "="*70)
print("11. CREATING VISUALIZATIONS")
print("="*70)

# Prediction vs Actual plot
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for i, muni in enumerate(municipalities):
    if i < 12:
        axes[i].plot(y_actual[:, i], label='Actual', color='blue', alpha=0.7)
        axes[i].plot(y_pred[:, i], label='Predicted', color='red', alpha=0.7)
        axes[i].set_title(muni, fontsize=10)
        axes[i].legend(fontsize=8)
        axes[i].set_xlabel('Week')
        axes[i].set_ylabel('Cases')

plt.suptitle('Actual vs Predicted Cases by Municipality', fontsize=14)
plt.tight_layout()
plt.savefig(r'C:\Users\admin\Downloads\version-2 DL\predictions_by_municipality.png', dpi=150)
plt.close()

# Outbreak risk heatmap
risk_df = pd.DataFrame(outbreak_results)
risk_df['risk_numeric'] = risk_df['risk_level'].map({
    'LOW': 1, 'MODERATE': 2, 'HIGH': 3, 'CRITICAL (OUTBREAK)': 4
})

plt.figure(figsize=(12, 6))
colors = {'LOW': 'green', 'MODERATE': 'yellow', 'HIGH': 'orange', 'CRITICAL (OUTBREAK)': 'red'}
risk_colors = [colors.get(r, 'gray') for r in risk_df['risk_level']]

bars = plt.barh(risk_df['municipality'], risk_df['predicted_cases'], color=risk_colors)
plt.xlabel('Predicted Cases')
plt.title('Outbreak Risk by Municipality (Next Week)')
plt.axvline(x=risk_df['threshold_p90'].mean(), color='red', linestyle='--', label='Outbreak Threshold (Avg P90)')
plt.legend()
plt.tight_layout()
plt.savefig(r'C:\Users\admin\Downloads\version-2 DL\outbreak_risk_map.png', dpi=150)
plt.close()

# ============================================
# 12. SAVE RESULTS
# ============================================
print("\n" + "="*70)
print("12. SAVING RESULTS")
print("="*70)

# Save model
model.save(r'C:\Users\admin\Downloads\version-2 DL\spatiotemporal_lstm.keras')

# Save scalers
joblib.dump(scalers, r'C:\Users\admin\Downloads\version-2 DL\location_scalers.pkl')
joblib.dump(historical_stats, r'C:\Users\admin\Downloads\version-2 DL\historical_stats.pkl')

# Save predictions
pd.DataFrame(outbreak_results).to_csv(
    r'C:\Users\admin\Downloads\version-2 DL\municipality_predictions.csv', index=False)
pd.DataFrame(barangay_predictions).to_csv(
    r'C:\Users\admin\Downloads\version-2 DL\barangay_predictions.csv', index=False)
pd.DataFrame(muni_metrics).to_csv(
    r'C:\Users\admin\Downloads\version-2 DL\municipality_metrics.csv', index=False)

# Save pivot data for future use
pivot_cases.to_csv(r'C:\Users\admin\Downloads\version-2 DL\pivot_weekly_cases.csv')

print("Saved files:")
print("  - spatiotemporal_lstm.keras")
print("  - location_scalers.pkl")
print("  - historical_stats.pkl")
print("  - municipality_predictions.csv")
print("  - barangay_predictions.csv")
print("  - municipality_metrics.csv")
print("  - pivot_weekly_cases.csv")
print("  - training_history.png")
print("  - predictions_by_municipality.png")
print("  - outbreak_risk_map.png")

# ============================================
# 13. SUMMARY
# ============================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

high_risk = [r for r in outbreak_results if 'HIGH' in r['risk_level'] or 'CRITICAL' in r['risk_level']]

print(f"""
Model Architecture:
  - Type: Bidirectional LSTM (Multi-Output)
  - Input: {LOOK_BACK} weeks x {n_features} municipalities
  - Output: Case count for each municipality
  
Performance:
  - Overall R²: {r2_score(y_actual.flatten(), y_pred.flatten()):.4f}
  - Overall RMSE: {np.sqrt(mean_squared_error(y_actual, y_pred)):.2f}
  
Next Week Predictions:
  - Total Predicted Cases: {total_pred}
  - High/Critical Risk Areas: {len(high_risk)}

Municipalities Requiring Attention:
""")

for r in sorted(outbreak_results, key=lambda x: x['predicted_cases'], reverse=True)[:5]:
    print(f"  * {r['municipality']}: {r['predicted_cases']} cases ({r['risk_level']})")

print("\nSpatio-temporal prediction complete!")
