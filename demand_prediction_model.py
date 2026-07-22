# ==================================================
# Hotel Demand Prediction Model
# ==================================================

import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score
)


# ======================================
# 1. Load Preprocessed Data
# ======================================

print("Loading Processed Data...")

X_train = pd.read_csv(
    "processed_data/X_train.csv"
)

X_test = pd.read_csv(
    "processed_data/X_test.csv"
)

y_train = pd.read_csv(
    "processed_data/y_train.csv"
).values.ravel()

y_test = pd.read_csv(
    "processed_data/y_test.csv"
).values.ravel()


print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)



# ======================================
# 2. Random Forest Model
# ======================================

print("\nTraining Random Forest...")


rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


rf_model.fit(
    X_train,
    y_train
)


rf_prediction = rf_model.predict(
    X_test
)


print("\nRandom Forest Results")

print(
    "Accuracy:",
    accuracy_score(
        y_test,
        rf_prediction
    )
)

print(
    "F1 Score:",
    f1_score(
        y_test,
        rf_prediction
    )
)



# ======================================
# 3. LightGBM Demand Prediction Model
# ======================================

print("\nTraining LightGBM...")


lgbm_model = LGBMClassifier(

    n_estimators=200,
    learning_rate=0.05,
    max_depth=8,
    random_state=42
)


lgbm_model.fit(
    X_train,
    y_train
)



# Prediction

lgbm_prediction = lgbm_model.predict(
    X_test
)


lgbm_probability = lgbm_model.predict_proba(
    X_test
)[:,1]



# ======================================
# 4. Model Evaluation
# ======================================


print("\nLightGBM Results")


print(
    "Accuracy:",
    accuracy_score(
        y_test,
        lgbm_prediction
    )
)


print(
    "F1 Score:",
    f1_score(
        y_test,
        lgbm_prediction
    )
)


print(
    "ROC-AUC:",
    roc_auc_score(
        y_test,
        lgbm_probability
    )
)


print("\nClassification Report")

print(
    classification_report(
        y_test,
        lgbm_prediction
    )
)



# ======================================
# 5. Save Demand Prediction Model
# ======================================

joblib.dump(
    lgbm_model,
    "models/hotel_demand_model.pkl"
)


print("\n================================")
print("Demand Prediction Model Saved")
print("================================")