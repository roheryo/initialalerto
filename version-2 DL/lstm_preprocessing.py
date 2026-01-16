"""
LSTM Preprocessing and Feature Engineering Script
For Davao de Oro ILI (Influenza-Like Illness) Cases Data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("Loading data...")
df = pd.read_csv(r'C:\Users\admin\Downloads\version-2 DL\Davao_de_Oro_Cases_2025-Generated-2.csv')
print(f"Original data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# ============================================
# 1. DATA CLEANING
# ============================================
print("\n" + "="*50)
print("1. DATA CLEANING")
print("="*50)

# Convert date columns to datetime
df['create_date'] = pd.to_datetime(df['create_date'])
df['created_at'] = pd.to_datetime(df['created_at'])
df['birthdate'] = pd.to_datetime(df['birthdate'], errors='coerce')

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Fill or drop missing values as appropriate
df = df.dropna(subset=['create_date', 'sex', 'AGE', 'current_address_city'])

print(f"\nData shape after cleaning: {df.shape}")

# ============================================
# 2. TEMPORAL FEATURE ENGINEERING
# ============================================
print("\n" + "="*50)
print("2. TEMPORAL FEATURE ENGINEERING")
print("="*50)

# Extract temporal features from create_date
df['year'] = df['create_date'].dt.year
df['month'] = df['create_date'].dt.month
df['day'] = df['create_date'].dt.day
df['day_of_week'] = df['create_date'].dt.dayofweek  # 0=Monday, 6=Sunday
df['week_of_year'] = df['create_date'].dt.isocalendar().week.astype(int)
df['quarter'] = df['create_date'].dt.quarter
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Cyclical encoding for temporal features (important for LSTM)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['week_of_year_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
df['week_of_year_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

print("Temporal features created: year, month, day, day_of_week, week_of_year, quarter, is_weekend")
print("Cyclical encodings created for: month, day_of_week, week_of_year")

# ============================================
# 3. CATEGORICAL ENCODING
# ============================================
print("\n" + "="*50)
print("3. CATEGORICAL ENCODING")
print("="*50)

# Label encode categorical variables
label_encoders = {}
categorical_cols = ['sex', 'current_address_city', 'lab_result', 'outcome', 'classification']

for col in categorical_cols:
    le = LabelEncoder()
    df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"{col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ============================================
# 4. AGGREGATE DATA BY DATE (Time Series for LSTM)
# ============================================
print("\n" + "="*50)
print("4. CREATING TIME SERIES AGGREGATIONS")
print("="*50)

# Daily aggregation
daily_agg = df.groupby('create_date').agg({
    'type': 'count',  # Total cases per day
    'sex_encoded': 'mean',  # Proportion (since binary encoded)
    'AGE': ['mean', 'std', 'min', 'max'],
    'lab_result_encoded': 'mean',  # Proportion of positive results
    'outcome_encoded': 'mean',
    'classification_encoded': 'mean',
    'MORBIDITY WEEK': 'first'
}).reset_index()

# Flatten column names
daily_agg.columns = ['date', 'case_count', 'sex_ratio', 'age_mean', 'age_std', 
                     'age_min', 'age_max', 'lab_result_ratio', 'outcome_ratio', 
                     'classification_ratio', 'morbidity_week']

# Fill NaN in std with 0 (when only 1 case per day)
daily_agg['age_std'] = daily_agg['age_std'].fillna(0)

# Sort by date
daily_agg = daily_agg.sort_values('date').reset_index(drop=True)

print(f"Daily aggregated data shape: {daily_agg.shape}")
print(daily_agg.head(10))

# ============================================
# 5. LAG FEATURES (Critical for LSTM)
# ============================================
print("\n" + "="*50)
print("5. CREATING LAG FEATURES")
print("="*50)

# Create lag features for case_count
for lag in [1, 2, 3, 7, 14]:  # Previous day, 2 days, 3 days, 1 week, 2 weeks
    daily_agg[f'cases_lag_{lag}'] = daily_agg['case_count'].shift(lag)

# Rolling statistics (moving averages)
for window in [3, 7, 14]:
    daily_agg[f'cases_rolling_mean_{window}'] = daily_agg['case_count'].rolling(window=window).mean()
    daily_agg[f'cases_rolling_std_{window}'] = daily_agg['case_count'].rolling(window=window).std()

# Trend features
daily_agg['cases_diff_1'] = daily_agg['case_count'].diff(1)  # Day-over-day change
daily_agg['cases_diff_7'] = daily_agg['case_count'].diff(7)  # Week-over-week change

print("Lag features created: 1, 2, 3, 7, 14 days")
print("Rolling statistics created: 3, 7, 14 day windows")

# ============================================
# 6. ADD TEMPORAL FEATURES TO AGGREGATED DATA
# ============================================
print("\n" + "="*50)
print("6. ADDING TEMPORAL FEATURES TO AGGREGATED DATA")
print("="*50)

daily_agg['month'] = pd.to_datetime(daily_agg['date']).dt.month
daily_agg['day_of_week'] = pd.to_datetime(daily_agg['date']).dt.dayofweek
daily_agg['week_of_year'] = pd.to_datetime(daily_agg['date']).dt.isocalendar().week.astype(int)
daily_agg['is_weekend'] = daily_agg['day_of_week'].isin([5, 6]).astype(int)

# Cyclical encoding
daily_agg['month_sin'] = np.sin(2 * np.pi * daily_agg['month'] / 12)
daily_agg['month_cos'] = np.cos(2 * np.pi * daily_agg['month'] / 12)
daily_agg['day_of_week_sin'] = np.sin(2 * np.pi * daily_agg['day_of_week'] / 7)
daily_agg['day_of_week_cos'] = np.cos(2 * np.pi * daily_agg['day_of_week'] / 7)

# ============================================
# 7. HANDLE MISSING VALUES FROM LAG/ROLLING
# ============================================
print("\n" + "="*50)
print("7. HANDLING MISSING VALUES")
print("="*50)

print(f"Missing values before handling:\n{daily_agg.isnull().sum()}")

# Drop rows with NaN (from lag features) or use forward fill
daily_agg_clean = daily_agg.dropna().reset_index(drop=True)
print(f"\nData shape after dropping NaN: {daily_agg_clean.shape}")

# ============================================
# 8. FEATURE SCALING (Critical for LSTM)
# ============================================
print("\n" + "="*50)
print("8. FEATURE SCALING")
print("="*50)

# Select features for scaling (exclude date column)
feature_cols = [col for col in daily_agg_clean.columns if col != 'date']

# Initialize scalers
scaler = MinMaxScaler(feature_range=(0, 1))

# Scale features
daily_agg_scaled = daily_agg_clean.copy()
daily_agg_scaled[feature_cols] = scaler.fit_transform(daily_agg_clean[feature_cols])

print(f"Features scaled to range (0, 1)")
print(f"Feature columns: {feature_cols}")

# ============================================
# 9. CREATE SEQUENCES FOR LSTM
# ============================================
print("\n" + "="*50)
print("9. CREATING SEQUENCES FOR LSTM")
print("="*50)

def create_sequences(data, seq_length, target_col='case_count'):
    """
    Create sequences for LSTM input
    
    Parameters:
    - data: DataFrame with features
    - seq_length: Number of time steps to look back
    - target_col: Column to predict
    
    Returns:
    - X: Input sequences (samples, seq_length, features)
    - y: Target values
    """
    feature_cols = [col for col in data.columns if col != 'date']
    target_idx = feature_cols.index(target_col)
    
    values = data[feature_cols].values
    X, y = [], []
    
    for i in range(len(values) - seq_length):
        X.append(values[i:i+seq_length])
        y.append(values[i+seq_length, target_idx])
    
    return np.array(X), np.array(y)

# Create sequences with different lookback periods
SEQ_LENGTH = 14  # Look back 14 days

X, y = create_sequences(daily_agg_scaled, SEQ_LENGTH)

print(f"Sequence length (lookback): {SEQ_LENGTH} days")
print(f"X shape: {X.shape} (samples, timesteps, features)")
print(f"y shape: {y.shape}")

# ============================================
# 10. TRAIN/TEST SPLIT (Time-based)
# ============================================
print("\n" + "="*50)
print("10. TRAIN/TEST SPLIT")
print("="*50)

# Time-based split (80% train, 20% test)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Training set: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"Test set: X_test {X_test.shape}, y_test {y_test.shape}")

# ============================================
# 11. SAVE PREPROCESSED DATA
# ============================================
print("\n" + "="*50)
print("11. SAVING PREPROCESSED DATA")
print("="*50)

# Save daily aggregated data (unscaled) for reference
daily_agg_clean.to_csv(r'C:\Users\admin\Downloads\version-2 DL\lstm_daily_aggregated.csv', index=False)

# Save scaled data
daily_agg_scaled.to_csv(r'C:\Users\admin\Downloads\version-2 DL\lstm_daily_scaled.csv', index=False)

# Save numpy arrays for direct LSTM use
np.save(r'C:\Users\admin\Downloads\version-2 DL\X_train.npy', X_train)
np.save(r'C:\Users\admin\Downloads\version-2 DL\X_test.npy', X_test)
np.save(r'C:\Users\admin\Downloads\version-2 DL\y_train.npy', y_train)
np.save(r'C:\Users\admin\Downloads\version-2 DL\y_test.npy', y_test)

# Save scaler for later inverse transform
import joblib
joblib.dump(scaler, r'C:\Users\admin\Downloads\version-2 DL\scaler.pkl')
joblib.dump(label_encoders, r'C:\Users\admin\Downloads\version-2 DL\label_encoders.pkl')

print("Saved files:")
print("  - lstm_daily_aggregated.csv (unscaled daily data)")
print("  - lstm_daily_scaled.csv (scaled daily data)")
print("  - X_train.npy, X_test.npy (LSTM input sequences)")
print("  - y_train.npy, y_test.npy (target values)")
print("  - scaler.pkl (MinMaxScaler for inverse transform)")
print("  - label_encoders.pkl (LabelEncoders for categorical vars)")

# ============================================
# 12. SUMMARY
# ============================================
print("\n" + "="*50)
print("PREPROCESSING SUMMARY")
print("="*50)

print(f"""
Data Processing Complete!

Original Records: {len(df)}
Daily Aggregated Records: {len(daily_agg_clean)}
Sequence Length: {SEQ_LENGTH} days
Total Features: {X.shape[2]}

Training Samples: {len(X_train)}
Test Samples: {len(X_test)}

Features included:
- Case count and demographic aggregations
- Lag features (1, 2, 3, 7, 14 days)
- Rolling statistics (3, 7, 14 day windows)
- Trend features (day-over-day, week-over-week changes)
- Cyclical temporal encodings (month, day_of_week)

Ready for LSTM model training!
""")

# Feature list for reference
print("\nFeature columns for LSTM:")
for i, col in enumerate(feature_cols):
    print(f"  {i}: {col}")
