# file: generate_car_cleaning.py
"""Data cleaning and preprocessing script for car_details.csv.

The script performs the following production‑ready steps:
1. Load the raw CSV.
2. Add a `car_age` feature (current year – manufacturing year).
3. Separate the target (`selling_price`).
4. Identify numeric and categorical feature columns.
5. Impute missing values:
   * Numeric → median
   * Categorical → most frequent (mode)
6. Encode categoricals with OneHotEncoder (drop='first' to avoid perfect collinearity).
7. Assemble a clean DataFrame that includes the imputed & encoded features **and** the target column.
8. Save the cleaned dataset to `car_details_cleaned.csv`.

The preprocessing pipeline (sklearn ColumnTransformer) is built explicitly so that the same steps can be saved and re‑used on future data (e.g., validation or production inference).
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ------------------------------------------------------------------
# 1. Load raw data
# ------------------------------------------------------------------
RAW_PATH = os.path.join('data', 'car_details.csv')
df_raw = pd.read_csv(RAW_PATH)
print('✅ Loaded raw dataset – shape:', df_raw.shape)

# ------------------------------------------------------------------
# 2. Feature engineering – car age
# ------------------------------------------------------------------
CURRENT_YEAR = datetime.now().year
if 'year' in df_raw.columns:
    df_raw['car_age'] = CURRENT_YEAR - df_raw['year']
    print('✅ Added `car_age` feature')

# ------------------------------------------------------------------
# 3. Separate target and feature matrix
# ------------------------------------------------------------------
TARGET_COL = 'selling_price'
if TARGET_COL not in df_raw.columns:
    raise KeyError(f"Target column '{TARGET_COL}' not found in the dataset")

y = df_raw[TARGET_COL].reset_index(drop=True)
X = df_raw.drop(columns=[TARGET_COL])

# ------------------------------------------------------------------
# 4. Identify column types
# ------------------------------------------------------------------
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
print('\nNumeric feature columns:', numeric_features)
print('Categorical feature columns:', categorical_features)

# ------------------------------------------------------------------
# 5. Build preprocessing pipelines
# ------------------------------------------------------------------
numeric_transformer = SimpleImputer(strategy='median')
# OneHotEncoder: dense output for easy DataFrame creation
# Scikit‑learn >=1.2 uses `sparse_output=False`; older versions use `sparse=False`.
try:
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
    ])
except TypeError:
    # fallback for older sklearn versions
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse=False))
    ])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'  # any column not listed is discarded (none in this case)
)

# ------------------------------------------------------------------
# 6. Fit & transform the feature matrix
# ------------------------------------------------------------------
X_processed = preprocessor.fit_transform(X)
print('\n✅ Preprocessing fitted. Processed feature matrix shape:', X_processed.shape)

# ------------------------------------------------------------------
# 7. Construct DataFrame with proper column names
# ------------------------------------------------------------------
# Retrieve one‑hot column names
onehot_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(onehot_feature_names)

# Convert to DataFrame (ensure dense array)
X_clean = pd.DataFrame(X_processed, columns=all_feature_names)
# Cast one‑hot columns to int for clarity (0/1)
for col in onehot_feature_names:
    X_clean[col] = X_clean[col].astype(int)

# ------------------------------------------------------------------
# 8. Combine features with target column
# ------------------------------------------------------------------
df_cleaned = pd.concat([X_clean, y], axis=1)
print('✅ Final cleaned DataFrame shape (features + target):', df_cleaned.shape)

# ------------------------------------------------------------------
# 9. Save cleaned dataset
# ------------------------------------------------------------------
CLEAN_PATH = 'car_details_cleaned.csv'
df_cleaned.to_csv(CLEAN_PATH, index=False)
print(f'🗂️ Cleaned dataset saved to {CLEAN_PATH}')