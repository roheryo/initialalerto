"""
Model Improvement Analysis and Enhanced Prediction
- Correlation Analysis
- Advanced Feature Engineering
- Multiple Model Comparison (LSTM, XGBoost, Random Forest, GRU)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import joblib
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

print("="*70)
print("MODEL IMPROVEMENT ANALYSIS")
print("="*70)

# ============================================
# 1. LOAD AND PREPARE DATA
# ============================================
print("\n1. LOADING DATA...")

df = pd.read_csv(r'C:\Users\admin\Downloads\version-2 DL\Davao_de_Oro_Cases_2025-Generated-2.csv')
df['create_date'] = pd.to_datetime(df['create_date'])

print(f"Total records: {len(df)}")

# ============================================
# 2. ADVANCED FEATURE ENGINEERING
# ============================================
print("\n" + "="*70)
print("2. ADVANCED FEATURE ENGINEERING")
print("="*70)

# Basic temporal features
df['year'] = df['create_date'].dt.year
df['month'] = df['create_date'].dt.month
df['day'] = df['create_date'].dt.day
df['day_of_week'] = df['create_date'].dt.dayofweek
df['week_of_year'] = df['create_date'].dt.isocalendar().week.astype(int)
df['quarter'] = df['create_date'].dt.quarter
df['day_of_year'] = df['create_date'].dt.dayofyear
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['is_month_start'] = df['create_date'].dt.is_month_start.astype(int)
df['is_month_end'] = df['create_date'].dt.is_month_end.astype(int)

# Encode categorical
le_sex = LabelEncoder()
le_city = LabelEncoder()
le_lab = LabelEncoder()
le_outcome = LabelEncoder()
le_class = LabelEncoder()

df['sex_encoded'] = le_sex.fit_transform(df['sex'])
df['city_encoded'] = le_city.fit_transform(df['current_address_city'])
df['lab_encoded'] = le_lab.fit_transform(df['lab_result'])
df['outcome_encoded'] = le_outcome.fit_transform(df['outcome'])
df['class_encoded'] = le_class.fit_transform(df['classification'])

# Daily aggregation with ENHANCED features
daily = df.groupby('create_date').agg({
    'type': 'count',
    'sex_encoded': ['mean', 'sum'],
    'AGE': ['mean', 'std', 'min', 'max', 'median'],
    'lab_encoded': ['mean', 'sum'],
    'outcome_encoded': ['mean', 'sum'],
    'class_encoded': ['mean', 'sum'],
    'city_encoded': ['nunique', lambda x: x.value_counts().index[0]],  # unique cities, most common
}).reset_index()

daily.columns = ['date', 'case_count', 'female_ratio', 'female_count',
                 'age_mean', 'age_std', 'age_min', 'age_max', 'age_median',
                 'positive_ratio', 'positive_count',
                 'outcome_mean', 'severe_count',
                 'confirmed_ratio', 'confirmed_count',
                 'affected_cities', 'hotspot_city']

daily['age_std'] = daily['age_std'].fillna(0)
daily = daily.sort_values('date').reset_index(drop=True)

# ENHANCED TEMPORAL FEATURES
daily['month'] = daily['date'].dt.month
daily['day_of_week'] = daily['date'].dt.dayofweek
daily['week_of_year'] = daily['date'].dt.isocalendar().week.astype(int)
daily['day_of_year'] = daily['date'].dt.dayofyear
daily['quarter'] = daily['date'].dt.quarter
daily['is_weekend'] = daily['day_of_week'].isin([5, 6]).astype(int)

# Cyclical encoding
daily['month_sin'] = np.sin(2 * np.pi * daily['month'] / 12)
daily['month_cos'] = np.cos(2 * np.pi * daily['month'] / 12)
daily['dow_sin'] = np.sin(2 * np.pi * daily['day_of_week'] / 7)
daily['dow_cos'] = np.cos(2 * np.pi * daily['day_of_week'] / 7)
daily['doy_sin'] = np.sin(2 * np.pi * daily['day_of_year'] / 365)
daily['doy_cos'] = np.cos(2 * np.pi * daily['day_of_year'] / 365)

# EXTENDED LAG FEATURES (more lags for better pattern capture)
for lag in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]:
    daily[f'lag_{lag}'] = daily['case_count'].shift(lag)

# ROLLING STATISTICS (multiple windows)
for window in [3, 5, 7, 14, 21, 28]:
    daily[f'roll_mean_{window}'] = daily['case_count'].rolling(window).mean()
    daily[f'roll_std_{window}'] = daily['case_count'].rolling(window).std()
    daily[f'roll_min_{window}'] = daily['case_count'].rolling(window).min()
    daily[f'roll_max_{window}'] = daily['case_count'].rolling(window).max()

# TREND FEATURES
daily['diff_1'] = daily['case_count'].diff(1)
daily['diff_7'] = daily['case_count'].diff(7)
daily['diff_14'] = daily['case_count'].diff(14)

# Percentage change
daily['pct_change_1'] = daily['case_count'].pct_change(1)
daily['pct_change_7'] = daily['case_count'].pct_change(7)

# EXPONENTIAL MOVING AVERAGES
for span in [3, 7, 14, 21]:
    daily[f'ema_{span}'] = daily['case_count'].ewm(span=span).mean()

# MOMENTUM FEATURES
daily['momentum_7'] = daily['case_count'] - daily['lag_7']
daily['momentum_14'] = daily['case_count'] - daily['lag_14']

# VOLATILITY
daily['volatility_7'] = daily['case_count'].rolling(7).std() / daily['case_count'].rolling(7).mean()
daily['volatility_14'] = daily['case_count'].rolling(14).std() / daily['case_count'].rolling(14).mean()

# SEASONALITY INDICATORS (month-based historical average)
monthly_avg = daily.groupby('month')['case_count'].transform('mean')
daily['seasonal_factor'] = daily['case_count'] / monthly_avg.replace(0, 1)

# OUTBREAK INDICATORS
daily['above_mean'] = (daily['case_count'] > daily['case_count'].mean()).astype(int)
daily['above_median'] = (daily['case_count'] > daily['case_count'].median()).astype(int)

# Replace inf values
daily = daily.replace([np.inf, -np.inf], np.nan)

# Drop NaN rows
daily_clean = daily.dropna().reset_index(drop=True)

print(f"Features created: {daily_clean.shape[1] - 2}")  # minus date and target
print(f"Clean samples: {len(daily_clean)}")

# ============================================
# 3. CORRELATION ANALYSIS
# ============================================
print("\n" + "="*70)
print("3. CORRELATION ANALYSIS")
print("="*70)

# Select numeric columns for correlation
numeric_cols = daily_clean.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != 'case_count']

# Calculate correlations with target
correlations = daily_clean[numeric_cols + ['case_count']].corr()['case_count'].drop('case_count')
correlations = correlations.abs().sort_values(ascending=False)

print("\nTop 20 Features by Correlation with Case Count:")
print("-" * 50)
for i, (feat, corr) in enumerate(correlations.head(20).items(), 1):
    print(f"{i:2}. {feat:<30} {corr:.4f}")

# Save correlation plot
plt.figure(figsize=(12, 10))
top_features = correlations.head(25).index.tolist() + ['case_count']
sns.heatmap(daily_clean[top_features].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Matrix - Top Features vs Case Count')
plt.tight_layout()
plt.savefig(r'C:\Users\admin\Downloads\version-2 DL\correlation_matrix.png', dpi=150)
plt.close()

print("\nCorrelation matrix saved to correlation_matrix.png")

# Filter features with correlation > 0.1
good_features = correlations[correlations > 0.1].index.tolist()
print(f"\nFeatures with correlation > 0.1: {len(good_features)}")

# ============================================
# 4. PREPARE DATA FOR MODELING
# ============================================
print("\n" + "="*70)
print("4. PREPARING DATA FOR MODELING")
print("="*70)

# Use top correlated features
feature_cols = good_features if len(good_features) >= 10 else correlations.head(30).index.tolist()
target_col = 'case_count'

X = daily_clean[feature_cols].values
y = daily_clean[target_col].values

# Scale features
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Time-based split
train_size = int(len(X) * 0.8)
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]
y_train_raw, y_test_raw = y[:train_size], y[train_size:]

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features used: {len(feature_cols)}")

# ============================================
# 5. MODEL COMPARISON
# ============================================
print("\n" + "="*70)
print("5. MODEL COMPARISON")
print("="*70)

results = {}

# --- 5.1 Random Forest ---
print("\n--- Random Forest ---")
rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, 
                           random_state=42, n_jobs=-1)
rf.fit(X_train, y_train_raw)
rf_pred = rf.predict(X_test)
results['Random Forest'] = {
    'RMSE': np.sqrt(mean_squared_error(y_test_raw, rf_pred)),
    'MAE': mean_absolute_error(y_test_raw, rf_pred),
    'R2': r2_score(y_test_raw, rf_pred)
}
print(f"R²: {results['Random Forest']['R2']:.4f}, RMSE: {results['Random Forest']['RMSE']:.2f}")

# --- 5.2 XGBoost ---
print("\n--- XGBoost ---")
xgb = XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, 
                   subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb.fit(X_train, y_train_raw)
xgb_pred = xgb.predict(X_test)
results['XGBoost'] = {
    'RMSE': np.sqrt(mean_squared_error(y_test_raw, xgb_pred)),
    'MAE': mean_absolute_error(y_test_raw, xgb_pred),
    'R2': r2_score(y_test_raw, xgb_pred)
}
print(f"R²: {results['XGBoost']['R2']:.4f}, RMSE: {results['XGBoost']['RMSE']:.2f}")

# --- 5.3 Gradient Boosting ---
print("\n--- Gradient Boosting ---")
gb = GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, 
                                subsample=0.8, random_state=42)
gb.fit(X_train, y_train_raw)
gb_pred = gb.predict(X_test)
results['Gradient Boosting'] = {
    'RMSE': np.sqrt(mean_squared_error(y_test_raw, gb_pred)),
    'MAE': mean_absolute_error(y_test_raw, gb_pred),
    'R2': r2_score(y_test_raw, gb_pred)
}
print(f"R²: {results['Gradient Boosting']['R2']:.4f}, RMSE: {results['Gradient Boosting']['RMSE']:.2f}")

# --- 5.4 Improved LSTM ---
print("\n--- Improved LSTM ---")

# Create sequences
SEQ_LENGTH = 14
def create_sequences(X, y, seq_length):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:i+seq_length])
        ys.append(y[i+seq_length])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LENGTH)
train_seq_size = int(len(X_seq) * 0.8)
X_train_seq, X_test_seq = X_seq[:train_seq_size], X_seq[train_seq_size:]
y_train_seq, y_test_seq = y_seq[:train_seq_size], y_seq[train_seq_size:]

# Build improved LSTM
lstm_model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(SEQ_LENGTH, len(feature_cols))),
    Dropout(0.3),
    BatchNormalization(),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dense(1)
])

lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
]

lstm_model.fit(X_train_seq, y_train_seq, epochs=150, batch_size=32,
               validation_split=0.2, callbacks=callbacks, verbose=0)

lstm_pred_scaled = lstm_model.predict(X_test_seq, verbose=0).flatten()
lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).flatten()
y_test_lstm = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

results['Improved LSTM'] = {
    'RMSE': np.sqrt(mean_squared_error(y_test_lstm, lstm_pred)),
    'MAE': mean_absolute_error(y_test_lstm, lstm_pred),
    'R2': r2_score(y_test_lstm, lstm_pred)
}
print(f"R²: {results['Improved LSTM']['R2']:.4f}, RMSE: {results['Improved LSTM']['RMSE']:.2f}")

# --- 5.5 GRU Model ---
print("\n--- GRU Model ---")

gru_model = Sequential([
    Bidirectional(GRU(128, return_sequences=True), input_shape=(SEQ_LENGTH, len(feature_cols))),
    Dropout(0.3),
    BatchNormalization(),
    GRU(64, return_sequences=False),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

gru_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
gru_model.fit(X_train_seq, y_train_seq, epochs=150, batch_size=32,
              validation_split=0.2, callbacks=callbacks, verbose=0)

gru_pred_scaled = gru_model.predict(X_test_seq, verbose=0).flatten()
gru_pred = scaler_y.inverse_transform(gru_pred_scaled.reshape(-1, 1)).flatten()

results['GRU'] = {
    'RMSE': np.sqrt(mean_squared_error(y_test_lstm, gru_pred)),
    'MAE': mean_absolute_error(y_test_lstm, gru_pred),
    'R2': r2_score(y_test_lstm, gru_pred)
}
print(f"R²: {results['GRU']['R2']:.4f}, RMSE: {results['GRU']['RMSE']:.2f}")

# ============================================
# 6. RESULTS COMPARISON
# ============================================
print("\n" + "="*70)
print("6. MODEL COMPARISON RESULTS")
print("="*70)

results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('R2', ascending=False)

print("\n" + results_df.to_string())

# Find best model
best_model_name = results_df['R2'].idxmax()
print(f"\n*** Best Model: {best_model_name} (R² = {results_df.loc[best_model_name, 'R2']:.4f}) ***")

# Save comparison plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# R² comparison
axes[0].barh(results_df.index, results_df['R2'], color='steelblue')
axes[0].set_xlabel('R² Score')
axes[0].set_title('Model Comparison - R² Score')
axes[0].axvline(x=0.7, color='green', linestyle='--', label='Good (0.7)')

# RMSE comparison
axes[1].barh(results_df.index, results_df['RMSE'], color='coral')
axes[1].set_xlabel('RMSE')
axes[1].set_title('Model Comparison - RMSE')

# MAE comparison
axes[2].barh(results_df.index, results_df['MAE'], color='mediumseagreen')
axes[2].set_xlabel('MAE')
axes[2].set_title('Model Comparison - MAE')

plt.tight_layout()
plt.savefig(r'C:\Users\admin\Downloads\version-2 DL\model_comparison.png', dpi=150)
plt.close()

# ============================================
# 7. FEATURE IMPORTANCE (from best tree model)
# ============================================
print("\n" + "="*70)
print("7. FEATURE IMPORTANCE")
print("="*70)

# Use XGBoost feature importance
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features:")
print("-" * 50)
for i, row in importance.head(15).iterrows():
    print(f"{row['feature']:<30} {row['importance']:.4f}")

# Save feature importance plot
plt.figure(figsize=(10, 8))
plt.barh(importance.head(20)['feature'], importance.head(20)['importance'], color='teal')
plt.xlabel('Importance')
plt.title('Top 20 Feature Importances (XGBoost)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(r'C:\Users\admin\Downloads\version-2 DL\feature_importance.png', dpi=150)
plt.close()

# ============================================
# 8. SAVE BEST MODEL AND PREDICTIONS
# ============================================
print("\n" + "="*70)
print("8. SAVING RESULTS")
print("="*70)

# Save models
joblib.dump(xgb, r'C:\Users\admin\Downloads\version-2 DL\best_xgboost_model.pkl')
joblib.dump(rf, r'C:\Users\admin\Downloads\version-2 DL\random_forest_model.pkl')
lstm_model.save(r'C:\Users\admin\Downloads\version-2 DL\improved_lstm_model.keras')
gru_model.save(r'C:\Users\admin\Downloads\version-2 DL\gru_model.keras')

# Save scalers
joblib.dump(scaler_X, r'C:\Users\admin\Downloads\version-2 DL\scaler_X.pkl')
joblib.dump(scaler_y, r'C:\Users\admin\Downloads\version-2 DL\scaler_y.pkl')

# Save feature list
joblib.dump(feature_cols, r'C:\Users\admin\Downloads\version-2 DL\feature_cols.pkl')

# Save results
results_df.to_csv(r'C:\Users\admin\Downloads\version-2 DL\model_comparison_results.csv')
importance.to_csv(r'C:\Users\admin\Downloads\version-2 DL\feature_importance.csv', index=False)

print("Saved files:")
print("  - correlation_matrix.png")
print("  - model_comparison.png")
print("  - feature_importance.png")
print("  - best_xgboost_model.pkl")
print("  - random_forest_model.pkl")
print("  - improved_lstm_model.keras")
print("  - gru_model.keras")
print("  - model_comparison_results.csv")
print("  - feature_importance.csv")

# ============================================
# 9. SUMMARY
# ============================================
print("\n" + "="*70)
print("IMPROVEMENT SUMMARY")
print("="*70)

print(f"""
Correlation Analysis:
  - Features with strong correlation (>0.3): {len(correlations[correlations > 0.3])}
  - Features with moderate correlation (0.1-0.3): {len(correlations[(correlations > 0.1) & (correlations <= 0.3)])}
  - Top correlated feature: {correlations.index[0]} ({correlations.iloc[0]:.4f})

Feature Engineering:
  - Total engineered features: {len(feature_cols)}
  - Lag features: 1-28 days
  - Rolling statistics: 3, 5, 7, 14, 21, 28 day windows
  - Trend features: diff, pct_change, momentum
  - Exponential moving averages: 3, 7, 14, 21 spans

Model Performance (Best: {best_model_name}):
  - R² Score: {results_df.loc[best_model_name, 'R2']:.4f}
  - RMSE: {results_df.loc[best_model_name, 'RMSE']:.2f}
  - MAE: {results_df.loc[best_model_name, 'MAE']:.2f}

Recommendations:
  1. Use {best_model_name} for predictions
  2. Focus on top features: {', '.join(importance.head(5)['feature'].tolist())}
  3. Consider ensemble of top 2-3 models for robust predictions
""")

print("\nModel improvement complete!")
