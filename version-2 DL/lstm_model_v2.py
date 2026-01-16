"""
LSTM Model V2 - Multi-Output for Outbreak Prediction
Predicts: Case Count (regression) + Outbreak Flag (classification)

Uses preprocessed data from lstm_preprocessing_v2.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout, Bidirectional,
                                      BatchNormalization, Embedding, Concatenate, Flatten)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import joblib
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

print("="*70)
print("LSTM MODEL V2 - MULTI-OUTPUT OUTBREAK PREDICTION")
print("="*70)

# ============================================
# 1. LOAD PREPROCESSED DATA
# ============================================
print("\n1. LOADING PREPROCESSED DATA...")

data_dir = r'C:\Users\admin\Downloads\version-2 DL'

# Load training and test data
X_train = np.load(f'{data_dir}/X_train_v2.npy')
X_test = np.load(f'{data_dir}/X_test_v2.npy')
y_cases_train = np.load(f'{data_dir}/y_cases_train.npy')
y_cases_test = np.load(f'{data_dir}/y_cases_test.npy')
y_outbreak_train = np.load(f'{data_dir}/y_outbreak_train.npy')
y_outbreak_test = np.load(f'{data_dir}/y_outbreak_test.npy')
loc_train = np.load(f'{data_dir}/location_ids_train.npy')
loc_test = np.load(f'{data_dir}/location_ids_test.npy')

# Load scalers and config
target_scaler = joblib.load(f'{data_dir}/target_scaler_v2.pkl')
location_mapping = joblib.load(f'{data_dir}/location_mapping.pkl')
config = joblib.load(f'{data_dir}/preprocessing_config.pkl')

TIME_STEPS = config['time_steps']
N_FEATURES = config['n_features']
N_LOCATIONS = config['n_locations']

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Time steps: {TIME_STEPS}")
print(f"Features: {N_FEATURES}")
print(f"Locations: {N_LOCATIONS}")
print(f"Outbreak threshold: > {config['outbreak_threshold']} cases")

# ============================================
# 2. BUILD MULTI-OUTPUT LSTM MODEL
# ============================================
print("\n" + "="*70)
print("2. BUILDING MULTI-OUTPUT LSTM MODEL")
print("="*70)

# Input layer for time series features
sequence_input = Input(shape=(TIME_STEPS, N_FEATURES), name='sequence_input')

# Location input for embedding
location_input = Input(shape=(1,), name='location_input')

# Location embedding layer (learns spatial patterns)
EMBEDDING_DIM = 8
location_embedding = Embedding(
    input_dim=N_LOCATIONS,
    output_dim=EMBEDDING_DIM,
    name='location_embedding'
)(location_input)
location_flat = Flatten()(location_embedding)

# LSTM layers for temporal patterns
x = Bidirectional(LSTM(64, return_sequences=True), name='bilstm_1')(sequence_input)
x = Dropout(0.3)(x)
x = BatchNormalization()(x)

x = Bidirectional(LSTM(32, return_sequences=False), name='bilstm_2')(x)
x = Dropout(0.3)(x)

# Combine LSTM output with location embedding
combined = Concatenate()([x, location_flat])
combined = Dense(64, activation='relu')(combined)
combined = BatchNormalization()(combined)
combined = Dropout(0.2)(combined)
combined = Dense(32, activation='relu')(combined)

# Output 1: Case count prediction (regression)
cases_output = Dense(1, activation='linear', name='cases')(combined)

# Output 2: Outbreak prediction (binary classification)
outbreak_output = Dense(1, activation='sigmoid', name='outbreak')(combined)

# Build model
model = Model(
    inputs=[sequence_input, location_input],
    outputs=[cases_output, outbreak_output]
)

# Compile with custom losses and metrics
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={
        'cases': 'mse',
        'outbreak': 'binary_crossentropy'
    },
    loss_weights={
        'cases': 1.0,
        'outbreak': 1.0
    },
    metrics={
        'cases': ['mae'],
        'outbreak': ['accuracy']
    }
)

model.summary()

# ============================================
# 3. TRAIN MODEL
# ============================================
print("\n" + "="*70)
print("3. TRAINING MODEL")
print("="*70)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6, verbose=1),
    ModelCheckpoint(f'{data_dir}/best_lstm_v2.keras', monitor='val_loss', save_best_only=True, verbose=0)
]

history = model.fit(
    x=[X_train, loc_train],
    y={'cases': y_cases_train, 'outbreak': y_outbreak_train},
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Plot training history
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Total loss
axes[0, 0].plot(history.history['loss'], label='Train Loss')
axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
axes[0, 0].set_title('Total Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].legend()

# Cases loss
axes[0, 1].plot(history.history['cases_loss'], label='Train')
axes[0, 1].plot(history.history['val_cases_loss'], label='Val')
axes[0, 1].set_title('Cases MSE Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].legend()

# Outbreak loss
axes[1, 0].plot(history.history['outbreak_loss'], label='Train')
axes[1, 0].plot(history.history['val_outbreak_loss'], label='Val')
axes[1, 0].set_title('Outbreak BCE Loss')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].legend()

# Outbreak accuracy
axes[1, 1].plot(history.history['outbreak_accuracy'], label='Train')
axes[1, 1].plot(history.history['val_outbreak_accuracy'], label='Val')
axes[1, 1].set_title('Outbreak Accuracy')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(f'{data_dir}/training_history_v2.png', dpi=150)
plt.close()

print("\nTraining history saved to training_history_v2.png")

# ============================================
# 4. EVALUATE MODEL
# ============================================
print("\n" + "="*70)
print("4. MODEL EVALUATION")
print("="*70)

# Predictions
predictions = model.predict([X_test, loc_test])
pred_cases = predictions[0].flatten()
pred_outbreak = predictions[1].flatten()

# Inverse transform case predictions
pred_cases_actual = target_scaler.inverse_transform(pred_cases.reshape(-1, 1)).flatten()
actual_cases = target_scaler.inverse_transform(y_cases_test.reshape(-1, 1)).flatten()

# Classification threshold for outbreak
pred_outbreak_binary = (pred_outbreak > 0.5).astype(int)

# REGRESSION METRICS (Case Count)
print("\n--- CASE COUNT PREDICTION (Regression) ---")
rmse = np.sqrt(mean_squared_error(actual_cases, pred_cases_actual))
mae = mean_absolute_error(actual_cases, pred_cases_actual)
r2 = r2_score(actual_cases, pred_cases_actual)

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# CLASSIFICATION METRICS (Outbreak)
print("\n--- OUTBREAK DETECTION (Classification) ---")
accuracy = accuracy_score(y_outbreak_test, pred_outbreak_binary)
precision = precision_score(y_outbreak_test, pred_outbreak_binary, zero_division=0)
recall = recall_score(y_outbreak_test, pred_outbreak_binary, zero_division=0)
f1 = f1_score(y_outbreak_test, pred_outbreak_binary, zero_division=0)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Try to compute AUC if there are positive samples
try:
    auc = roc_auc_score(y_outbreak_test, pred_outbreak)
    print(f"AUC-ROC: {auc:.4f}")
except:
    auc = None
    print("AUC-ROC: N/A (insufficient positive samples)")

print("\nClassification Report:")
print(classification_report(y_outbreak_test, pred_outbreak_binary, 
                           target_names=['No Outbreak', 'Outbreak'],
                           zero_division=0))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_outbreak_test, pred_outbreak_binary)
print(cm)

# ============================================
# 5. PREDICTION VISUALIZATION
# ============================================
print("\n" + "="*70)
print("5. CREATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Actual vs Predicted Cases
axes[0, 0].scatter(actual_cases, pred_cases_actual, alpha=0.5, s=20)
axes[0, 0].plot([0, actual_cases.max()], [0, actual_cases.max()], 'r--', label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Cases')
axes[0, 0].set_ylabel('Predicted Cases')
axes[0, 0].set_title(f'Case Count: Actual vs Predicted (R²={r2:.4f})')
axes[0, 0].legend()

# Case prediction over time
axes[0, 1].plot(actual_cases[:100], label='Actual', alpha=0.7)
axes[0, 1].plot(pred_cases_actual[:100], label='Predicted', alpha=0.7)
axes[0, 1].set_xlabel('Sample')
axes[0, 1].set_ylabel('Cases')
axes[0, 1].set_title('Case Count: Time Series (First 100 Samples)')
axes[0, 1].legend()

# Outbreak probability distribution
axes[1, 0].hist(pred_outbreak[y_outbreak_test == 0], bins=20, alpha=0.7, label='No Outbreak', color='green')
axes[1, 0].hist(pred_outbreak[y_outbreak_test == 1], bins=20, alpha=0.7, label='Outbreak', color='red')
axes[1, 0].axvline(x=0.5, color='black', linestyle='--', label='Threshold')
axes[1, 0].set_xlabel('Outbreak Probability')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('Outbreak Probability Distribution')
axes[1, 0].legend()

# Confusion matrix heatmap
import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
            xticklabels=['Predicted No', 'Predicted Yes'],
            yticklabels=['Actual No', 'Actual Yes'])
axes[1, 1].set_title('Confusion Matrix')

plt.tight_layout()
plt.savefig(f'{data_dir}/evaluation_plots_v2.png', dpi=150)
plt.close()

print("Evaluation plots saved to evaluation_plots_v2.png")

# ============================================
# 6. PREDICT NEXT WEEK
# ============================================
print("\n" + "="*70)
print("6. PREDICTING NEXT WEEK")
print("="*70)

# Load weekly features to get last sequences
weekly_features = pd.read_csv(f'{data_dir}/weekly_features.csv')
lstm_features = joblib.load(f'{data_dir}/lstm_features.pkl')
feature_scaler = joblib.load(f'{data_dir}/feature_scaler_v2.pkl')

# Get predictions for each location
reverse_location_mapping = {v: k for k, v in location_mapping.items()}

next_week_predictions = []

for loc_id in range(N_LOCATIONS):
    loc_name = reverse_location_mapping[loc_id]
    
    # Get last TIME_STEPS data for this location
    loc_data = weekly_features[weekly_features['location_id'] == loc_id].sort_values('week_id')
    
    if len(loc_data) >= TIME_STEPS:
        # Get last sequence
        last_data = loc_data.tail(TIME_STEPS)
        
        # Scale features
        scaled_features = feature_scaler.transform(last_data[config['features_to_scale']])
        unscaled_features = last_data[config['features_no_scale']].values
        
        # Combine
        sequence = np.hstack([scaled_features, unscaled_features])
        sequence = sequence.reshape(1, TIME_STEPS, N_FEATURES)
        
        # Predict
        loc_array = np.array([[loc_id]])
        pred = model.predict([sequence, loc_array], verbose=0)
        
        pred_cases_val = target_scaler.inverse_transform([[pred[0][0][0]]])[0][0]
        pred_outbreak_prob = pred[1][0][0]
        
        next_week_predictions.append({
            'location_id': loc_id,
            'municipality': loc_name,
            'predicted_cases': max(0, round(pred_cases_val)),
            'outbreak_probability': round(pred_outbreak_prob, 4),
            'outbreak_prediction': 'YES' if pred_outbreak_prob > 0.5 else 'NO',
            'historical_mean': round(loc_data['confirmed_cases'].mean(), 1),
            'historical_max': loc_data['confirmed_cases'].max()
        })

# Create DataFrame
predictions_df = pd.DataFrame(next_week_predictions)
predictions_df = predictions_df.sort_values('predicted_cases', ascending=False)

print("\nNext Week Predictions by Municipality:")
print("-" * 80)
print(f"{'Municipality':<20} {'Pred Cases':<12} {'Outbreak Prob':<14} {'Prediction':<12} {'Hist Mean':<10}")
print("=" * 80)

for _, row in predictions_df.iterrows():
    print(f"{row['municipality']:<20} {row['predicted_cases']:<12} {row['outbreak_probability']:<14.4f} {row['outbreak_prediction']:<12} {row['historical_mean']:<10.1f}")

total_pred = predictions_df['predicted_cases'].sum()
outbreak_count = (predictions_df['outbreak_prediction'] == 'YES').sum()
print(f"\nTotal Predicted Cases: {total_pred}")
print(f"Municipalities with Outbreak Warning: {outbreak_count}")

# ============================================
# 7. SAVE RESULTS
# ============================================
print("\n" + "="*70)
print("7. SAVING RESULTS")
print("="*70)

# Save model
model.save(f'{data_dir}/lstm_model_v2.keras')

# Save predictions
predictions_df.to_csv(f'{data_dir}/next_week_predictions_v2.csv', index=False)

# Save evaluation metrics
metrics = {
    'regression': {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    },
    'classification': {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    },
    'confusion_matrix': cm.tolist()
}
joblib.dump(metrics, f'{data_dir}/evaluation_metrics_v2.pkl')

print("Saved files:")
print("  - lstm_model_v2.keras")
print("  - best_lstm_v2.keras (best weights)")
print("  - next_week_predictions_v2.csv")
print("  - evaluation_metrics_v2.pkl")
print("  - training_history_v2.png")
print("  - evaluation_plots_v2.png")

# ============================================
# 8. SUMMARY
# ============================================
print("\n" + "="*70)
print("MODEL SUMMARY")
print("="*70)

print(f"""
Model Architecture:
  - Type: Multi-Output LSTM with Location Embedding
  - Inputs: 
    * Sequence: ({TIME_STEPS} weeks, {N_FEATURES} features)
    * Location: Embedding ({N_LOCATIONS} locations → {EMBEDDING_DIM} dims)
  - Outputs:
    * Cases: Linear (regression)
    * Outbreak: Sigmoid (classification)

Performance:
  CASE COUNT PREDICTION:
    - RMSE: {rmse:.2f}
    - MAE: {mae:.2f}
    - R²: {r2:.4f}
  
  OUTBREAK DETECTION:
    - Accuracy: {accuracy:.4f}
    - Precision: {precision:.4f}
    - Recall: {recall:.4f}
    - F1 Score: {f1:.4f}

Next Week Forecast:
  - Total Predicted Cases: {total_pred}
  - Outbreak Warnings: {outbreak_count} municipalities
""")

# Top risk municipalities
high_risk = predictions_df[predictions_df['outbreak_prediction'] == 'YES']
if len(high_risk) > 0:
    print("High Risk Municipalities (Outbreak Warning):")
    for _, row in high_risk.iterrows():
        print(f"  * {row['municipality']}: {row['predicted_cases']} cases ({row['outbreak_probability']*100:.1f}% probability)")
else:
    print("No municipalities with outbreak warning (probability > 50%)")

print("\n" + "="*70)
print("MODEL TRAINING COMPLETE!")
print("="*70)
