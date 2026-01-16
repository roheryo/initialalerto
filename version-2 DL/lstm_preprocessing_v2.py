"""
LSTM-Ready Preprocessing Pipeline V2
For Davao de Oro Disease Surveillance Dataset
Predicts: WHEN, WHERE, and HOW MANY cases

Key Features:
- Weekly aggregation by location
- Lag features and rolling statistics
- Location embedding preparation
- Proper sliding window sequences
- Multi-output support (cases + outbreak flag)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*70)
print("LSTM-READY PREPROCESSING PIPELINE V2")
print("Preparing Data for Outbreak Prediction")
print("="*70)

# ============================================
# 1. LOAD RAW DATA
# ============================================
print("\n" + "="*70)
print("1. LOADING RAW DATA")
print("="*70)

df = pd.read_csv(r'C:\Users\admin\Downloads\version-2 DL\Davao_de_Oro_Cases_2025-Generated-2.csv')
df['create_date'] = pd.to_datetime(df['create_date'])

print(f"Total records: {len(df)}")
print(f"Date range: {df['create_date'].min()} to {df['create_date'].max()}")
print(f"Columns: {df.columns.tolist()}")

# ============================================
# 2. DATE & TIME AGGREGATION (WEEKLY)
# ============================================
print("\n" + "="*70)
print("2. WEEKLY AGGREGATION BY LOCATION")
print("="*70)

# Extract time components
df['year'] = df['create_date'].dt.year
df['week'] = df['create_date'].dt.isocalendar().week.astype(int)
df['month'] = df['create_date'].dt.month
df['week_of_year'] = df['week']

# Create unique week identifier for proper ordering
df['year_week'] = df['year'].astype(str) + '-W' + df['week'].astype(str).str.zfill(2)
df['week_id'] = df['year'] * 100 + df['week']  # Numeric for sorting

# Aggregate by Municipality + Week
weekly_muni = df.groupby(['year', 'week', 'week_id', 'current_address_city']).agg({
    'type': 'count',  # Total cases
    'lab_result': lambda x: (x == 'Positive').sum(),  # Confirmed cases
    'AGE': ['mean', 'std'],
    'sex': lambda x: (x == 'Female').mean(),  # Female ratio
    'outcome': lambda x: (x == 'DIED').sum(),  # Deaths
    'current_address_barangay': lambda x: x.value_counts().index[0]  # Top barangay
}).reset_index()

# Flatten column names
weekly_muni.columns = ['year', 'week', 'week_id', 'municipality', 
                       'total_cases', 'confirmed_cases', 
                       'age_mean', 'age_std', 'female_ratio', 
                       'deaths', 'top_barangay']

# Fill NaN in std
weekly_muni['age_std'] = weekly_muni['age_std'].fillna(0)

# Sort by time and location
weekly_muni = weekly_muni.sort_values(['week_id', 'municipality']).reset_index(drop=True)

print(f"Weekly aggregated records: {len(weekly_muni)}")
print(f"Municipalities: {weekly_muni['municipality'].nunique()}")
print(f"Weeks: {weekly_muni['week_id'].nunique()}")

# ============================================
# 3. CREATE OUTBREAK LABEL
# ============================================
print("\n" + "="*70)
print("3. CREATING OUTBREAK LABELS")
print("="*70)

# Define outbreak threshold
OUTBREAK_THRESHOLD = 30  # More than 30 confirmed cases = outbreak

weekly_muni['outbreak'] = (weekly_muni['confirmed_cases'] > OUTBREAK_THRESHOLD).astype(int)

outbreak_count = weekly_muni['outbreak'].sum()
print(f"Outbreak threshold: > {OUTBREAK_THRESHOLD} confirmed cases/week")
print(f"Total outbreak weeks: {outbreak_count} ({outbreak_count/len(weekly_muni)*100:.1f}%)")

# ============================================
# 4. LOCATION ENCODING (FOR EMBEDDING)
# ============================================
print("\n" + "="*70)
print("4. LOCATION ENCODING")
print("="*70)

# Create location IDs for embedding layer
location_encoder = LabelEncoder()
weekly_muni['location_id'] = location_encoder.fit_transform(weekly_muni['municipality'])

# Save mapping
location_mapping = dict(zip(location_encoder.classes_, 
                           location_encoder.transform(location_encoder.classes_)))
print("Location ID Mapping:")
for loc, idx in location_mapping.items():
    print(f"  {idx}: {loc}")

n_locations = len(location_mapping)
print(f"\nTotal locations: {n_locations}")

# ============================================
# 5. FEATURE ENGINEERING (PER LOCATION)
# ============================================
print("\n" + "="*70)
print("5. FEATURE ENGINEERING")
print("="*70)

# Process each municipality separately to avoid data leakage
all_features = []

for municipality in weekly_muni['municipality'].unique():
    muni_data = weekly_muni[weekly_muni['municipality'] == municipality].copy()
    muni_data = muni_data.sort_values('week_id')
    
    # LAG FEATURES (Past weeks)
    for lag in [1, 2, 3, 4]:
        muni_data[f'lag_{lag}_cases'] = muni_data['confirmed_cases'].shift(lag)
    
    # ROLLING STATISTICS (Moving windows)
    muni_data['rolling_mean_4'] = muni_data['confirmed_cases'].rolling(window=4, min_periods=1).mean()
    muni_data['rolling_std_4'] = muni_data['confirmed_cases'].rolling(window=4, min_periods=1).std()
    muni_data['rolling_max_4'] = muni_data['confirmed_cases'].rolling(window=4, min_periods=1).max()
    muni_data['rolling_min_4'] = muni_data['confirmed_cases'].rolling(window=4, min_periods=1).min()
    
    # 8-week rolling (2 months trend)
    muni_data['rolling_mean_8'] = muni_data['confirmed_cases'].rolling(window=8, min_periods=1).mean()
    muni_data['rolling_std_8'] = muni_data['confirmed_cases'].rolling(window=8, min_periods=1).std()
    
    # TREND FEATURES
    muni_data['diff_1'] = muni_data['confirmed_cases'].diff(1)  # Week-over-week change
    muni_data['diff_4'] = muni_data['confirmed_cases'].diff(4)  # Month-over-month change
    
    # PERCENTAGE CHANGE
    muni_data['pct_change_1'] = muni_data['confirmed_cases'].pct_change(1)
    
    # MOMENTUM
    muni_data['momentum_4'] = muni_data['confirmed_cases'] - muni_data['lag_4_cases']
    
    # EXPONENTIAL MOVING AVERAGE
    muni_data['ema_4'] = muni_data['confirmed_cases'].ewm(span=4, adjust=False).mean()
    
    # VOLATILITY (coefficient of variation)
    muni_data['volatility_4'] = muni_data['rolling_std_4'] / (muni_data['rolling_mean_4'] + 1)
    
    # OUTBREAK LAG (was there outbreak last week?)
    muni_data['outbreak_lag_1'] = muni_data['outbreak'].shift(1)
    muni_data['outbreak_lag_2'] = muni_data['outbreak'].shift(2)
    
    all_features.append(muni_data)

# Combine all municipalities
weekly_features = pd.concat(all_features, ignore_index=True)

# Fill NaN from lag/rolling operations
weekly_features['rolling_std_4'] = weekly_features['rolling_std_4'].fillna(0)
weekly_features['rolling_std_8'] = weekly_features['rolling_std_8'].fillna(0)
weekly_features = weekly_features.fillna(0)

# Replace inf values
weekly_features = weekly_features.replace([np.inf, -np.inf], 0)

print("Features created:")
print("  - Lag features: lag_1 to lag_4")
print("  - Rolling statistics: mean, std, max, min (4-week and 8-week)")
print("  - Trend features: diff_1, diff_4, pct_change")
print("  - Momentum and EMA")
print("  - Volatility coefficient")
print("  - Outbreak lags")

print(f"\nDataset shape: {weekly_features.shape}")

# ============================================
# 6. TIME FEATURES (NUMERIC)
# ============================================
print("\n" + "="*70)
print("6. TIME FEATURES (NUMERIC)")
print("="*70)

# Week of year (1-52) - numeric
weekly_features['week_of_year'] = weekly_features['week']

# Month (1-12)
weekly_features['month'] = ((weekly_features['week'] - 1) // 4) + 1
weekly_features['month'] = weekly_features['month'].clip(1, 12)

# Cyclical encoding for seasonality
weekly_features['week_sin'] = np.sin(2 * np.pi * weekly_features['week'] / 52)
weekly_features['week_cos'] = np.cos(2 * np.pi * weekly_features['week'] / 52)
weekly_features['month_sin'] = np.sin(2 * np.pi * weekly_features['month'] / 12)
weekly_features['month_cos'] = np.cos(2 * np.pi * weekly_features['month'] / 12)

print("Time features added:")
print("  - week_of_year (1-52)")
print("  - month (1-12)")
print("  - Cyclical encodings (sin/cos)")

# ============================================
# 7. DEFINE FEATURE SETS
# ============================================
print("\n" + "="*70)
print("7. DEFINING FEATURE SETS")
print("="*70)

# Features to scale (continuous)
features_to_scale = [
    'confirmed_cases', 'total_cases', 'age_mean', 'age_std', 'female_ratio', 'deaths',
    'lag_1_cases', 'lag_2_cases', 'lag_3_cases', 'lag_4_cases',
    'rolling_mean_4', 'rolling_std_4', 'rolling_max_4', 'rolling_min_4',
    'rolling_mean_8', 'rolling_std_8',
    'diff_1', 'diff_4', 'pct_change_1', 'momentum_4', 'ema_4', 'volatility_4',
    'week_sin', 'week_cos', 'month_sin', 'month_cos'
]

# Features NOT to scale (binary/categorical)
features_no_scale = ['outbreak_lag_1', 'outbreak_lag_2', 'location_id', 'week_of_year']

# Target variables
target_regression = 'confirmed_cases'  # For case count prediction
target_classification = 'outbreak'     # For outbreak detection

print(f"Features to scale: {len(features_to_scale)}")
print(f"Features not scaled: {len(features_no_scale)}")

# ============================================
# 8. SCALING (MinMaxScaler)
# ============================================
print("\n" + "="*70)
print("8. SCALING FEATURES")
print("="*70)

scaler = MinMaxScaler(feature_range=(0, 1))

# Scale only continuous features
weekly_features_scaled = weekly_features.copy()
weekly_features_scaled[features_to_scale] = scaler.fit_transform(weekly_features[features_to_scale])

# Keep a separate scaler for target (for inverse transform later)
target_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler.fit(weekly_features[['confirmed_cases']])

print(f"Scaled {len(features_to_scale)} features to range (0, 1)")
print("Target scaler saved for inverse transform")

# ============================================
# 9. SEQUENCE CONSTRUCTION (SLIDING WINDOW)
# ============================================
print("\n" + "="*70)
print("9. CREATING SLIDING WINDOW SEQUENCES")
print("="*70)

TIME_STEPS = 4  # Use 4 weeks of history to predict next week

def create_sequences_per_location(df, features, target_col, time_steps):
    """
    Create LSTM sequences per location
    Returns X, y, and location_ids
    """
    X_list, y_cases_list, y_outbreak_list, loc_list = [], [], [], []
    
    for loc_id in df['location_id'].unique():
        loc_data = df[df['location_id'] == loc_id].sort_values('week_id')
        
        if len(loc_data) <= time_steps:
            continue
        
        feature_values = loc_data[features].values
        target_cases = loc_data['confirmed_cases'].values
        target_outbreak = loc_data['outbreak'].values
        
        for i in range(len(loc_data) - time_steps):
            # Input: past TIME_STEPS weeks
            X_list.append(feature_values[i:i+time_steps])
            
            # Target: next week's values
            y_cases_list.append(target_cases[i+time_steps])
            y_outbreak_list.append(target_outbreak[i+time_steps])
            loc_list.append(loc_id)
    
    return (np.array(X_list), 
            np.array(y_cases_list), 
            np.array(y_outbreak_list), 
            np.array(loc_list))

# All features for LSTM input
lstm_features = features_to_scale + features_no_scale

X, y_cases, y_outbreak, location_ids = create_sequences_per_location(
    weekly_features_scaled, 
    lstm_features, 
    'confirmed_cases', 
    TIME_STEPS
)

print(f"Time steps (lookback): {TIME_STEPS} weeks")
print(f"\nSequence shapes:")
print(f"  X: {X.shape} (samples, time_steps, features)")
print(f"  y_cases: {y_cases.shape} (regression target)")
print(f"  y_outbreak: {y_outbreak.shape} (classification target)")
print(f"  location_ids: {location_ids.shape}")

# ============================================
# 10. TRAIN-TEST SPLIT (TIME-AWARE)
# ============================================
print("\n" + "="*70)
print("10. TIME-AWARE TRAIN-TEST SPLIT")
print("="*70)

# Use first 80% for training, last 20% for testing
# This maintains temporal order - NO RANDOM SPLIT!
train_size = int(len(X) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]
y_cases_train = y_cases[:train_size]
y_cases_test = y_cases[train_size:]
y_outbreak_train = y_outbreak[:train_size]
y_outbreak_test = y_outbreak[train_size:]
loc_train = location_ids[:train_size]
loc_test = location_ids[train_size:]

print(f"Training samples: {len(X_train)} (first 80%)")
print(f"Test samples: {len(X_test)} (last 20%)")
print(f"\nTraining outbreak ratio: {y_outbreak_train.mean()*100:.1f}%")
print(f"Test outbreak ratio: {y_outbreak_test.mean()*100:.1f}%")

# ============================================
# 11. MULTI-OUTPUT TARGET (COMBINED)
# ============================================
print("\n" + "="*70)
print("11. MULTI-OUTPUT TARGET PREPARATION")
print("="*70)

# Combine targets for multi-output model
y_train_multi = np.column_stack([y_cases_train, y_outbreak_train])
y_test_multi = np.column_stack([y_cases_test, y_outbreak_test])

print(f"Multi-output target shape:")
print(f"  y_train: {y_train_multi.shape} [cases, outbreak_flag]")
print(f"  y_test: {y_test_multi.shape}")

# ============================================
# 12. SAVE PREPROCESSED DATA
# ============================================
print("\n" + "="*70)
print("12. SAVING PREPROCESSED DATA")
print("="*70)

output_dir = r'C:\Users\admin\Downloads\version-2 DL'

# Save numpy arrays
np.save(f'{output_dir}/X_train_v2.npy', X_train)
np.save(f'{output_dir}/X_test_v2.npy', X_test)
np.save(f'{output_dir}/y_cases_train.npy', y_cases_train)
np.save(f'{output_dir}/y_cases_test.npy', y_cases_test)
np.save(f'{output_dir}/y_outbreak_train.npy', y_outbreak_train)
np.save(f'{output_dir}/y_outbreak_test.npy', y_outbreak_test)
np.save(f'{output_dir}/y_train_multi.npy', y_train_multi)
np.save(f'{output_dir}/y_test_multi.npy', y_test_multi)
np.save(f'{output_dir}/location_ids_train.npy', loc_train)
np.save(f'{output_dir}/location_ids_test.npy', loc_test)

# Save scalers and encoders
joblib.dump(scaler, f'{output_dir}/feature_scaler_v2.pkl')
joblib.dump(target_scaler, f'{output_dir}/target_scaler_v2.pkl')
joblib.dump(location_encoder, f'{output_dir}/location_encoder.pkl')
joblib.dump(location_mapping, f'{output_dir}/location_mapping.pkl')

# Save feature list
joblib.dump(lstm_features, f'{output_dir}/lstm_features.pkl')

# Save weekly features for reference
weekly_features.to_csv(f'{output_dir}/weekly_features.csv', index=False)

# Save configuration
config = {
    'time_steps': TIME_STEPS,
    'n_features': len(lstm_features),
    'n_locations': n_locations,
    'outbreak_threshold': OUTBREAK_THRESHOLD,
    'features_to_scale': features_to_scale,
    'features_no_scale': features_no_scale,
    'train_size': len(X_train),
    'test_size': len(X_test)
}
joblib.dump(config, f'{output_dir}/preprocessing_config.pkl')

print("Saved files:")
print("  - X_train_v2.npy, X_test_v2.npy (input sequences)")
print("  - y_cases_train/test.npy (regression targets)")
print("  - y_outbreak_train/test.npy (classification targets)")
print("  - y_train/test_multi.npy (multi-output targets)")
print("  - location_ids_train/test.npy")
print("  - feature_scaler_v2.pkl, target_scaler_v2.pkl")
print("  - location_encoder.pkl, location_mapping.pkl")
print("  - lstm_features.pkl")
print("  - weekly_features.csv")
print("  - preprocessing_config.pkl")

# ============================================
# 13. SUMMARY
# ============================================
print("\n" + "="*70)
print("PREPROCESSING SUMMARY")
print("="*70)

print(f"""
Dataset Characteristics:
  - Original records: {len(df)}
  - Weekly aggregated records: {len(weekly_features)}
  - Municipalities: {n_locations}
  - Time period: {df['create_date'].min().date()} to {df['create_date'].max().date()}

LSTM-Ready Data:
  - Input shape: {X.shape} (samples, {TIME_STEPS} weeks, {len(lstm_features)} features)
  - Training samples: {len(X_train)}
  - Test samples: {len(X_test)}

Target Variables:
  1. Case count (regression): confirmed_cases
  2. Outbreak flag (classification): outbreak (threshold > {OUTBREAK_THRESHOLD})

Feature Categories:
  - Lag features: lag_1 to lag_4 (past 4 weeks)
  - Rolling statistics: 4-week and 8-week windows
  - Trend features: diff, pct_change, momentum
  - Time features: week_of_year, month (cyclical encoded)
  - Location: location_id (for embedding)

Ready for LSTM training!
""")

print("\n" + "="*70)
print("NEXT STEP: Run lstm_model_v2.py for model training")
print("="*70)
