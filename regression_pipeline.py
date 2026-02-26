# file: regression_pipeline.py
"""Regression modeling pipeline for predicting `selling_price`.

Steps:
1. Load cleaned dataset (`car_details_cleaned.csv`).
2. Train‑test split (80‑20).
3. Train three models:
   - Linear Regression (with StandardScaler)
   - Random Forest Regressor
   - Gradient Boosting Regressor
4. Evaluate each model using R² and RMSE on the test set.
5. Perform 5‑fold cross‑validation on the training data for a robust estimate.
6. Save the fitted models to disk (`.joblib`).
7. Output feature‑importance analysis:
   * Linear model – absolute coefficients.
   * Tree models – `feature_importances_`.

All results are printed to the console for quick inspection.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# ---------------------------------------------------------------
# 1. Load cleaned data
# ---------------------------------------------------------------
DATA_PATH = 'car_details_cleaned.csv'
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Cleaned dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print('✅ Loaded cleaned data – shape:', df.shape)

# Assume the target column is named exactly 'selling_price'
TARGET_COL = 'selling_price'
if TARGET_COL not in df.columns:
    raise KeyError(f"Target column '{TARGET_COL}' not present in data")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# ---------------------------------------------------------------
# 2. Train‑test split (80‑20)
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"🚀 Training set: {X_train.shape[0]} rows, Test set: {X_test.shape[0]} rows")

# ---------------------------------------------------------------
# 3. Model definitions & training
# ---------------------------------------------------------------
models = {}

# 3.a Linear Regression (with scaling)
lin_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LinearRegression())
])
lin_pipe.fit(X_train, y_train)
models['LinearRegression'] = lin_pipe
print('✅ Trained Linear Regression')

# 3.b Random Forest Regressor
rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    max_depth=None
)
rf.fit(X_train, y_train)
models['RandomForest'] = rf
print('✅ Trained Random Forest')

# 3.c Gradient Boosting Regressor
gbr = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    random_state=42
)
gbr.fit(X_train, y_train)
models['GradientBoosting'] = gbr
print('✅ Trained Gradient Boosting')

# ---------------------------------------------------------------
# 4. Evaluation on the hold‑out test set
# ---------------------------------------------------------------
print('\n=== Test‑set performance ===')
performance = {}
for name, model in models.items():
    # Predict
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    performance[name] = {'R2': r2, 'RMSE': rmse}
    print(f"{name}: R2 = {r2:.4f}, RMSE = {rmse:,.2f}")

# ---------------------------------------------------------------
# 5. Cross‑validation (5‑fold) on training data
# ---------------------------------------------------------------
print('\n=== 5‑fold CV on training data ===')
cv_results = {}
for name, model in models.items():
    # Use neg_mean_squared_error for RMSE calculation
    cv_scores_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    cv_scores_mse = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores_mse)
    cv_results[name] = {
        'CV_R2_Mean': cv_scores_r2.mean(),
        'CV_R2_Std': cv_scores_r2.std(),
        'CV_RMSE_Mean': cv_rmse.mean(),
        'CV_RMSE_Std': cv_rmse.std()
    }
    print(f"{name}: CV R2 = {cv_scores_r2.mean():.4f} ± {cv_scores_r2.std():.4f}, "
          f"CV RMSE = {cv_rmse.mean():,.2f} ± {cv_rmse.std():.2f}")

# ---------------------------------------------------------------
# 6. Save trained models for future use
# ---------------------------------------------------------------
MODEL_DIR = 'trained_models'
os.makedirs(MODEL_DIR, exist_ok=True)
for name, model in models.items():
    model_path = os.path.join(MODEL_DIR, f"{name}.joblib")
    joblib.dump(model, model_path)
    print(f"💾 Saved {name} to {model_path}")

# ---------------------------------------------------------------
# 7. Feature importance analysis
# ---------------------------------------------------------------
print('\n=== Feature Importance ===')
feature_names = X.columns.tolist()

# Linear Regression – absolute coefficient values (scaled version)
lin_coef = models['LinearRegression'].named_steps['lr'].coef_
lin_importance = pd.Series(np.abs(lin_coef), index=feature_names).sort_values(ascending=False)
print('\nLinear Regression (abs coefficients):')
print(lin_importance.head(10))

# Random Forest feature importances
rf_importance = pd.Series(models['RandomForest'].feature_importances_, index=feature_names).sort_values(ascending=False)
print('\nRandom Forest feature importance:')
print(rf_importance.head(10))

# Gradient Boosting feature importances
gbr_importance = pd.Series(models['GradientBoosting'].feature_importances_, index=feature_names).sort_values(ascending=False)
print('\nGradient Boosting feature importance:')
print(gbr_importance.head(10))

# ---------------------------------------------------------------
# 8. Summary table
# ---------------------------------------------------------------
summary = pd.DataFrame(performance).T
summary['CV_R2_Mean'] = [cv_results[m]['CV_R2_Mean'] for m in summary.index]
summary['CV_RMSE_Mean'] = [cv_results[m]['CV_RMSE_Mean'] for m in summary.index]
print('\n=== Summary of Model Performance ===')
print(summary)