# ==================================================
# Hotel Demand Prediction Model
# ==================================================

import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# ======================================
# Load Preprocessed Data
# ======================================

print("=" * 50)
print("Loading Processed Data...")
print("=" * 50)

X_train = pd.read_csv("processed_data/X_train.csv")
X_test = pd.read_csv("processed_data/X_test.csv")

y_train = pd.read_csv("processed_data/y_train.csv").values.ravel()
y_test = pd.read_csv("processed_data/y_test.csv").values.ravel()

print(f"Training Shape : {X_train.shape}")
print(f"Testing Shape  : {X_test.shape}")

# ======================================
# Random Forest Model
# ======================================

print("\nTraining Random Forest Model...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_prediction = rf_model.predict(X_test)
rf_probability = rf_model.predict_proba(X_test)[:, 1]

print("\nRandom Forest Performance")
print("-" * 40)

print("Accuracy :", accuracy_score(y_test, rf_prediction))
print("F1 Score :", f1_score(y_test, rf_prediction))
print("ROC-AUC  :", roc_auc_score(y_test, rf_probability))

# ======================================
# LightGBM Model
# ======================================

print("\nTraining LightGBM Model...")

lgbm_model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=8,
    random_state=42
)

lgbm_model.fit(X_train, y_train)

lgbm_prediction = lgbm_model.predict(X_test)
lgbm_probability = lgbm_model.predict_proba(X_test)[:, 1]

print("\nLightGBM Performance")
print("-" * 40)

print("Accuracy :", accuracy_score(y_test, lgbm_prediction))
print("F1 Score :", f1_score(y_test, lgbm_prediction))
print("ROC-AUC  :", roc_auc_score(y_test, lgbm_probability))

print("\nClassification Report")
print(classification_report(y_test, lgbm_prediction))

# ======================================
# Model Comparison
# ======================================

comparison = pd.DataFrame({
    "Model": ["Random Forest", "LightGBM"],
    "Accuracy": [
        accuracy_score(y_test, rf_prediction),
        accuracy_score(y_test, lgbm_prediction)
    ],
    "F1 Score": [
        f1_score(y_test, rf_prediction),
        f1_score(y_test, lgbm_prediction)
    ],
    "ROC-AUC": [
        roc_auc_score(y_test, rf_probability),
        roc_auc_score(y_test, lgbm_probability)
    ]
})

print("\nModel Comparison")
print(comparison)

# ======================================
# Save Models
# ======================================

joblib.dump(rf_model, "models/random_forest_model.pkl")
joblib.dump(lgbm_model, "models/hotel_demand_model.pkl")
joblib.dump(list(X_train.columns), "models/feature_names.pkl")

print("\nModels Saved Successfully!")

# ======================================
# Best Model
# ======================================

rf_f1 = f1_score(y_test, rf_prediction)
lgbm_f1 = f1_score(y_test, lgbm_prediction)

if lgbm_f1 > rf_f1:
    print("\nBest Model : LightGBM")
else:
    print("\nBest Model : Random Forest")

print("\n" + "=" * 50)
print("MODEL TRAINING COMPLETED")
print("=" * 50)