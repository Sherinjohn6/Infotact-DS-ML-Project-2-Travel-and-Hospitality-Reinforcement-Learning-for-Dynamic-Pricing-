# ============================================
# Hotel Booking Demand
# Model Comparison
# ============================================

import os
import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


# ============================================
# Load Test Data
# ============================================

X_test = pd.read_csv(
    "processed_data/X_test.csv"
)

y_test = pd.read_csv(
    "processed_data/y_test.csv"
).squeeze()


print("Test Data Loaded")
print("X_test shape:", X_test.shape)


# ============================================
# Load Saved Models
# ============================================

lightgbm_model = joblib.load(
    "models/lightgbm_hotel_model.pkl"
)

randomforest_model = joblib.load(
    "models/random_forest_hotel_model.pkl"
)


print("Models Loaded Successfully")


# ============================================
# Evaluation Function
# ============================================

def evaluate_model(model):

    start_time = time.time()

    prediction = model.predict(X_test)

    probability = model.predict_proba(X_test)[:,1]

    prediction_time = time.time() - start_time


    results = {

        "Accuracy":
        accuracy_score(y_test, prediction),

        "Precision":
        precision_score(y_test, prediction),

        "Recall":
        recall_score(y_test, prediction),

        "F1 Score":
        f1_score(y_test, prediction),

        "ROC-AUC":
        roc_auc_score(y_test, probability),

        "Prediction Time":
        prediction_time
    }


    return results, prediction



# ============================================
# Compare Models
# ============================================

lgb_results, lgb_prediction = evaluate_model(
    lightgbm_model
)


rf_results, rf_prediction = evaluate_model(
    randomforest_model
)



comparison = pd.DataFrame(
    {
        "LightGBM": lgb_results,
        "Random Forest": rf_results
    }
)


print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(comparison)



# ============================================
# Classification Report
# ============================================

print("\nLightGBM Classification Report")
print(
    classification_report(
        y_test,
        lgb_prediction
    )
)


print("\nRandom Forest Classification Report")
print(
    classification_report(
        y_test,
        rf_prediction
    )
)



# ============================================
# Save Results
# ============================================

os.makedirs(
    "results",
    exist_ok=True
)


comparison.to_csv(
    "results/model_comparison.csv"
)



# ============================================
# Plot Comparison
# ============================================

comparison.drop(
    "Prediction Time"
).T.plot(
    kind="bar",
    figsize=(10,6)
)


plt.title(
    "Hotel Demand Model Comparison"
)

plt.ylabel(
    "Score"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()


plt.savefig(
    "results/model_comparison.png"
)


plt.show()


print("\nComparison completed successfully!")