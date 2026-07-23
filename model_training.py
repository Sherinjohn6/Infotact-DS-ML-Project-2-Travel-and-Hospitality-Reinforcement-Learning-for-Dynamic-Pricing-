# ============================================
# Hotel Booking Demand - Model Training
# ============================================

import pandas as pd
import os
import time
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


# ============================================
# 1. Load SMOTE Dataset
# ============================================

print("Loading SMOTE dataset...")

df = pd.read_csv(
    "processed_data/smote_hotel_bookings.csv"
)

print("Dataset Shape:", df.shape)



# ============================================
# 2. Split Features and Target
# ============================================

X = df.drop(
    "is_canceled",
    axis=1
)

y = df["is_canceled"]


print("Feature Shape:", X.shape)
print("Target Shape:", y.shape)



# ============================================
# 3. Train Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



# ============================================
# 4. Define Models
# ============================================

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),


    "LightGBM": LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=10,
        random_state=42
    )
}



results = {}



# ============================================
# 5. Train and Evaluate Models
# ============================================

for name, model in models.items():

    print("\nTraining:", name)

    start = time.time()

    model.fit(
        X_train,
        y_train
    )

    training_time = time.time() - start


    y_pred = model.predict(
        X_test
    )


    y_prob = model.predict_proba(
        X_test
    )[:,1]


    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )


    results[name] = [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        training_time
    ]


    print(
        classification_report(
            y_test,
            y_pred
        )
    )


# ============================================
# 6. Model Comparison
# ============================================

results_df = pd.DataFrame(
    results,
    index=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "Training Time"
    ]
)


print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(results_df)



# ============================================
# # ============================================
# 7. Save Both Models
# ============================================

os.makedirs(
    "models",
    exist_ok=True
)


# Save LightGBM
joblib.dump(
    models["LightGBM"],
    "models/lightgbm_hotel_model.pkl"
)


# Save Random Forest
joblib.dump(
    models["Random Forest"],
    "models/random_forest_hotel_model.pkl"
)


# Save feature names
joblib.dump(
    list(X.columns),
    "models/features.pkl"
)


print("\n================================")
print("BOTH MODELS SAVED SUCCESSFULLY")
print("================================")
# Save test data
os.makedirs("processed_data", exist_ok=True)

X_test.to_csv("processed_data/X_test.csv", index=False)
y_test.to_csv("processed_data/y_test.csv", index=False)