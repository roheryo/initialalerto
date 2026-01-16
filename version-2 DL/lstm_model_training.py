"""
LSTM Model Training, Hyperparameter Tuning, and Outbreak Prediction
For Davao de Oro ILI Cases - Predicting outbreaks by municipality/barangay
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Bidirectional, 
                                      Input, BatchNormalization, Attention)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             classification_report, confusion_matrix)
import keras_tuner as kt
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("="*60)
print("LSTM MODEL TRAINING FOR ILI OUTBREAK PREDICTION")
print("="*60)

# ============================================
# 1. LOAD PREPROCESSED DATA
# ============================================
print("\n1. LOADING PREPROCESSED DATA...")

# Load the raw data for location-based analysis
df = pd.read_csv(r'C:\Users\admin\Downloads\version-2 DL\Davao_de_Oro_Cases_2025-Generated-2.csv')
df['create_date'] = pd.to_datetime(df['create_date'])

# Load preprocessed LSTM data
X_train = np.load(r'C:\Users\admin\Downloads\version-2 DL\X_train.npy')
X_test = np.load(r'C:\Users\admin\Downloads\version-2 DL\X_test.npy')
y_train = np.load(r'C:\Users\admin\Downloads\version-2 DL\y_train.npy')
y_test = np.load(r'C:\Users\admin\Downloads\version-2 DL\y_test.npy')
scaler = joblib.load(r'C:\Users\admin\Downloads\version-2 DL\scaler.pkl')

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

n_timesteps = X_train.shape[1]
n_features = X_train.shape[2]

# ============================================
# 2. CREATE LOCATION-BASED AGGREGATION
# ============================================
print("\n2. CREATING LOCATION-BASED AGGREGATION...")

# Aggregate by municipality and week
df['week'] = df['create_date'].dt.isocalendar().week.astype(int)
df['year'] = df['create_date'].dt.year

municipality_weekly = df.groupby(['current_address_city', 'year', 'week']).agg({
    'type': 'count',
    'AGE': 'mean',
    'current_address_barangay': lambda x: x.value_counts().index[0]  # Most affected barangay
}).reset_index()
municipality_weekly.columns = ['municipality', 'year', 'week', 'case_count', 'avg_age', 'top_barangay']

# Calculate historical statistics per municipality
municipality_stats = municipality_weekly.groupby('municipality').agg({
    'case_count': ['mean', 'std', 'max']
}).reset_index()
municipality_stats.columns = ['municipality', 'mean_cases', 'std_cases', 'max_cases']

print(f"Municipalities: {municipality_stats['municipality'].tolist()}")

# ============================================
# 3. DEFINE LSTM MODEL BUILDER FOR HYPERPARAMETER TUNING
# ============================================
print("\n3. SETTING UP HYPERPARAMETER TUNING...")

def build_model(hp):
    """Build LSTM model with hyperparameter tuning"""
    model = Sequential()
    
    # First Bidirectional LSTM layer
    model.add(Bidirectional(
        LSTM(
            units=hp.Int('lstm_units_1', min_value=64, max_value=256, step=64),
            return_sequences=True,
            input_shape=(n_timesteps, n_features)
        )
    ))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.4, step=0.1)))
    model.add(BatchNormalization())
    
    # Second LSTM layer
    model.add(LSTM(
        units=hp.Int('lstm_units_2', min_value=32, max_value=128, step=32),
        return_sequences=hp.Boolean('use_third_layer')
    ))
    model.add(Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.3, step=0.1)))
    
    # Optional third LSTM layer
    if hp.Boolean('use_third_layer'):
        model.add(LSTM(
            units=hp.Int('lstm_units_3', min_value=16, max_value=64, step=16),
            return_sequences=False
        ))
        model.add(Dropout(hp.Float('dropout_3', min_value=0.1, max_value=0.2, step=0.1)))
    
    # Dense layers
    model.add(Dense(hp.Int('dense_units', min_value=32, max_value=128, step=32), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output: predicted case count
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# ============================================
# 4. HYPERPARAMETER TUNING
# ============================================
print("\n4. RUNNING HYPERPARAMETER TUNING...")

tuner = kt.BayesianOptimization(
    build_model,
    objective='val_loss',
    max_trials=15,
    directory='lstm_tuning',
    project_name='ili_outbreak'
)

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Run hyperparameter search
tuner.search(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Get best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\nBest Hyperparameters:")
print(f"  LSTM Units Layer 1: {best_hps.get('lstm_units_1')}")
print(f"  LSTM Units Layer 2: {best_hps.get('lstm_units_2')}")
print(f"  Use Third Layer: {best_hps.get('use_third_layer')}")
if best_hps.get('use_third_layer'):
    print(f"  LSTM Units Layer 3: {best_hps.get('lstm_units_3')}")
print(f"  Dense Units: {best_hps.get('dense_units')}")
print(f"  Learning Rate: {best_hps.get('learning_rate'):.6f}")

# ============================================
# 5. BUILD AND TRAIN FINAL MODEL
# ============================================
print("\n5. TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS...")

best_model = tuner.hypermodel.build(best_hps)

checkpoint = ModelCheckpoint(
    r'C:\Users\admin\Downloads\version-2 DL\best_lstm_model.keras',
    monitor='val_loss',
    save_best_only=True
)

history = best_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

# ============================================
# 6. MODEL EVALUATION
# ============================================
print("\n" + "="*60)
print("6. MODEL EVALUATION")
print("="*60)

# Predictions
y_pred_train = best_model.predict(X_train).flatten()
y_pred_test = best_model.predict(X_test).flatten()

# Inverse transform to get actual case counts
# Note: We need to reconstruct the full feature array for inverse transform
def inverse_transform_target(y_scaled, scaler, target_idx=0):
    """Inverse transform just the target variable"""
    # Create a dummy array with the same shape as original features
    dummy = np.zeros((len(y_scaled), scaler.n_features_in_))
    dummy[:, target_idx] = y_scaled
    inv = scaler.inverse_transform(dummy)
    return inv[:, target_idx]

y_train_actual = inverse_transform_target(y_train, scaler)
y_test_actual = inverse_transform_target(y_test, scaler)
y_pred_train_actual = inverse_transform_target(y_pred_train, scaler)
y_pred_test_actual = inverse_transform_target(y_pred_test, scaler)

# Metrics
print("\nTraining Set Metrics:")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_train_actual, y_pred_train_actual)):.2f}")
print(f"  MAE: {mean_absolute_error(y_train_actual, y_pred_train_actual):.2f}")
print(f"  R²: {r2_score(y_train_actual, y_pred_train_actual):.4f}")

print("\nTest Set Metrics:")
rmse_test = np.sqrt(mean_squared_error(y_test_actual, y_pred_test_actual))
mae_test = mean_absolute_error(y_test_actual, y_pred_test_actual)
r2_test = r2_score(y_test_actual, y_pred_test_actual)
print(f"  RMSE: {rmse_test:.2f}")
print(f"  MAE: {mae_test:.2f}")
print(f"  R²: {r2_test:.4f}")

# ============================================
# 7. RISK LEVEL CLASSIFICATION
# ============================================
print("\n" + "="*60)
print("7. RISK LEVEL CLASSIFICATION")
print("="*60)

def classify_risk_level(cases, mean_cases, std_cases):
    """
    Classify risk level based on case count thresholds
    Low: below mean
    Medium: between mean and mean + 1 std
    High: between mean + 1 std and mean + 2 std
    Critical: above mean + 2 std (outbreak)
    """
    if cases < mean_cases:
        return 'Low'
    elif cases < mean_cases + std_cases:
        return 'Medium'
    elif cases < mean_cases + 2 * std_cases:
        return 'High'
    else:
        return 'Critical (Outbreak)'

# Calculate overall statistics for risk classification
overall_mean = y_train_actual.mean()
overall_std = y_train_actual.std()

print(f"Historical Mean Cases/Day: {overall_mean:.2f}")
print(f"Historical Std Cases/Day: {overall_std:.2f}")
print(f"\nRisk Level Thresholds:")
print(f"  Low: < {overall_mean:.0f} cases")
print(f"  Medium: {overall_mean:.0f} - {overall_mean + overall_std:.0f} cases")
print(f"  High: {overall_mean + overall_std:.0f} - {overall_mean + 2*overall_std:.0f} cases")
print(f"  Critical: > {overall_mean + 2*overall_std:.0f} cases")

# ============================================
# 8. PREDICT UPCOMING WEEK BY MUNICIPALITY
# ============================================
print("\n" + "="*60)
print("8. UPCOMING WEEK OUTBREAK PREDICTION")
print("="*60)

# Use last available sequence to predict next 7 days
daily_scaled = pd.read_csv(r'C:\Users\admin\Downloads\version-2 DL\lstm_daily_scaled.csv')
daily_unscaled = pd.read_csv(r'C:\Users\admin\Downloads\version-2 DL\lstm_daily_aggregated.csv')

# Get feature columns
feature_cols = [col for col in daily_scaled.columns if col != 'date']

# Prepare prediction sequence
last_sequence = daily_scaled[feature_cols].values[-n_timesteps:]
predictions_week = []

# Predict 7 days ahead
current_sequence = last_sequence.copy()
for day in range(7):
    pred_input = current_sequence.reshape(1, n_timesteps, n_features)
    pred_scaled = best_model.predict(pred_input, verbose=0)[0][0]
    
    # Update sequence for next prediction
    new_row = current_sequence[-1].copy()
    new_row[0] = pred_scaled  # Update case_count
    current_sequence = np.vstack([current_sequence[1:], new_row])
    
    # Inverse transform prediction
    pred_actual = inverse_transform_target(np.array([pred_scaled]), scaler)[0]
    predictions_week.append(max(0, pred_actual))  # Ensure non-negative

print("\nPredicted Cases for Next 7 Days:")
total_predicted = 0
for i, pred in enumerate(predictions_week, 1):
    risk = classify_risk_level(pred, overall_mean, overall_std)
    print(f"  Day {i}: {pred:.0f} cases - Risk Level: {risk}")
    total_predicted += pred

print(f"\nTotal Predicted Cases (Week): {total_predicted:.0f}")

# ============================================
# 9. MUNICIPALITY-LEVEL OUTBREAK PREDICTION
# ============================================
print("\n" + "="*60)
print("9. MUNICIPALITY & BARANGAY OUTBREAK PREDICTION")
print("="*60)

# Calculate historical distribution by municipality
municipality_dist = df.groupby('current_address_city').size() / len(df)

# Calculate recent trend (last 4 weeks)
recent_data = df[df['create_date'] >= df['create_date'].max() - pd.Timedelta(weeks=4)]
recent_dist = recent_data.groupby('current_address_city').size() / len(recent_data)

# Combine historical and recent for weighted prediction
combined_dist = 0.3 * municipality_dist + 0.7 * recent_dist.reindex(municipality_dist.index, fill_value=0)
combined_dist = combined_dist / combined_dist.sum()

# Predict cases per municipality for next week
print("\nPredicted Outbreak Risk by Municipality (Next Week):")
print("-" * 80)

municipality_predictions = []
for muni in combined_dist.index:
    predicted_cases = total_predicted * combined_dist[muni]
    muni_stats = municipality_stats[municipality_stats['municipality'] == muni]
    
    if len(muni_stats) > 0:
        muni_mean = muni_stats['mean_cases'].values[0]
        muni_std = muni_stats['std_cases'].values[0]
        risk = classify_risk_level(predicted_cases, muni_mean * 7, muni_std * 7)  # Weekly
    else:
        risk = 'Unknown'
    
    # Get most affected barangay from recent data
    muni_recent = recent_data[recent_data['current_address_city'] == muni]
    if len(muni_recent) > 0:
        top_barangay = muni_recent['current_address_barangay'].value_counts().index[0]
        barangay_pct = muni_recent['current_address_barangay'].value_counts().iloc[0] / len(muni_recent) * 100
    else:
        top_barangay = 'N/A'
        barangay_pct = 0
    
    municipality_predictions.append({
        'municipality': muni,
        'predicted_cases': predicted_cases,
        'risk_level': risk,
        'hotspot_barangay': top_barangay,
        'barangay_concentration': barangay_pct
    })

# Sort by predicted cases
municipality_predictions = sorted(municipality_predictions, key=lambda x: x['predicted_cases'], reverse=True)

print(f"{'Municipality':<20} {'Cases':<10} {'Risk Level':<20} {'Hotspot Barangay':<25} {'%':<5}")
print("=" * 80)
for pred in municipality_predictions:
    print(f"{pred['municipality']:<20} {pred['predicted_cases']:<10.0f} {pred['risk_level']:<20} {pred['hotspot_barangay']:<25} {pred['barangay_concentration']:.1f}%")

# ============================================
# 10. SAVE PREDICTIONS AND MODEL
# ============================================
print("\n" + "="*60)
print("10. SAVING RESULTS")
print("="*60)

# Save predictions
predictions_df = pd.DataFrame(municipality_predictions)
predictions_df.to_csv(r'C:\Users\admin\Downloads\version-2 DL\outbreak_predictions.csv', index=False)

# Save model summary
model_summary = {
    'best_hyperparameters': {
        'lstm_units_1': best_hps.get('lstm_units_1'),
        'lstm_units_2': best_hps.get('lstm_units_2'),
        'use_third_layer': best_hps.get('use_third_layer'),
        'dense_units': best_hps.get('dense_units'),
        'learning_rate': best_hps.get('learning_rate')
    },
    'metrics': {
        'test_rmse': rmse_test,
        'test_mae': mae_test,
        'test_r2': r2_test
    },
    'risk_thresholds': {
        'mean': overall_mean,
        'std': overall_std,
        'low': f'< {overall_mean:.0f}',
        'medium': f'{overall_mean:.0f} - {overall_mean + overall_std:.0f}',
        'high': f'{overall_mean + overall_std:.0f} - {overall_mean + 2*overall_std:.0f}',
        'critical': f'> {overall_mean + 2*overall_std:.0f}'
    }
}

joblib.dump(model_summary, r'C:\Users\admin\Downloads\version-2 DL\model_summary.pkl')

print("Saved files:")
print("  - best_lstm_model.keras (trained model)")
print("  - outbreak_predictions.csv (municipality predictions)")
print("  - model_summary.pkl (hyperparameters and metrics)")

# ============================================
# 11. FINAL SUMMARY
# ============================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

high_risk = [p for p in municipality_predictions if 'High' in p['risk_level'] or 'Critical' in p['risk_level']]

print(f"""
Model Performance:
  - Test RMSE: {rmse_test:.2f}
  - Test MAE: {mae_test:.2f}
  - Test R²: {r2_test:.4f}

Next Week Prediction:
  - Total Predicted Cases: {total_predicted:.0f}
  - High Risk Municipalities: {len(high_risk)}

Municipalities Requiring Immediate Attention:
""")

for pred in high_risk[:5]:
    print(f"  * {pred['municipality']}: ~{pred['predicted_cases']:.0f} cases ({pred['risk_level']})")
    print(f"    Hotspot: {pred['hotspot_barangay']}")

print("\nModel training and prediction complete!")
