import os

print(os.path.exists("data"))
import pandas as pd

df = pd.read_csv("hotel_bookings.csv")
# ============================================
# Hotel Demand Prediction - Data Preprocessing
# ============================================

import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib


# -------------------------------
# 1. Load Dataset
# -------------------------------

print("Loading Dataset...")

df = pd.read_csv("hotel_bookings.csv")

print("Dataset Shape:", df.shape)
print(df.head())


# -------------------------------
# 2. Basic Information
# -------------------------------

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# -------------------------------
# 3. Remove Unnecessary Columns
# -------------------------------

# These columns do not help prediction

drop_columns = [
    "reservation_status",
    "reservation_status_date"
]

df.drop(columns=drop_columns, inplace=True)


# -------------------------------
# 4. Handle Missing Values
# -------------------------------

# Fill country missing values with mode

df["country"].fillna(
    df["country"].mode()[0],
    inplace=True
)


# Fill agent and company missing values

df["agent"].fillna(0, inplace=True)
df["company"].fillna(0, inplace=True)


# Fill children missing values

df["children"].fillna(
    df["children"].median(),
    inplace=True
)


print("\nMissing Values After Treatment:")
print(df.isnull().sum().sum())


# -------------------------------
# 5. Feature Engineering
# -------------------------------

print("\nCreating New Features...")


# Total stay duration

df["total_stay"] = (
    df["stays_in_weekend_nights"] +
    df["stays_in_week_nights"]
)


# Total guests

df["total_guests"] = (
    df["adults"] +
    df["children"] +
    df["babies"]
)


# Total special requests

df["total_requests"] = (
    df["total_of_special_requests"]
)


# Arrival date feature

df["arrival_date"] = (
    df["arrival_date_day_of_month"]
)


# Booking lead time category

df["lead_time_category"] = pd.cut(
    df["lead_time"],
    bins=[-1,30,90,180,365,1000],
    labels=[
        "Short",
        "Medium",
        "Long",
        "Very_Long",
        "Extreme"
    ]
)


# -------------------------------
# 6. Drop Highly Correlated / Redundant Columns
# -------------------------------

df.drop(
    columns=[
        "arrival_date_year",
        "arrival_date_week_number",
        "arrival_date_day_of_month"
    ],
    inplace=True
)


# -------------------------------
# 7. Encode Categorical Variables
# -------------------------------

print("\nEncoding Categorical Features...")


categorical_columns = df.select_dtypes(
    include="object"
).columns


label_encoders = {}


for col in categorical_columns:

    le = LabelEncoder()

    df[col] = le.fit_transform(
        df[col].astype(str)
    )

    label_encoders[col] = le


# Encode new categorical feature

df["lead_time_category"] = LabelEncoder().fit_transform(
    df["lead_time_category"]
)


# -------------------------------
# 8. Handle Outliers
# -------------------------------

numeric_columns = df.select_dtypes(
    include=np.number
).columns


for col in numeric_columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3-Q1

    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR

    df[col] = np.clip(
        df[col],
        lower,
        upper
    )


# -------------------------------
# 9. Separate Features and Target
# -------------------------------

# Target:
# is_canceled (Booking cancellation prediction)

X = df.drop(
    "is_canceled",
    axis=1
)

y = df["is_canceled"]


print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)


# -------------------------------
# 10. Train Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------------
# 11. Feature Scaling
# -------------------------------

scaler = StandardScaler()


X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)


# -------------------------------
# 12. Save Processed Data
# -------------------------------

os.makedirs(
    "processed_data",
    exist_ok=True
)


pd.DataFrame(
    X_train
).to_csv(
    "processed_data/X_train.csv",
    index=False
)


pd.DataFrame(
    X_test
).to_csv(
    "processed_data/X_test.csv",
    index=False
)


pd.DataFrame(
    y_train
).to_csv(
    "processed_data/y_train.csv",
    index=False
)


pd.DataFrame(
    y_test
).to_csv(
    "processed_data/y_test.csv",
    index=False
)


# Save preprocessing objects

joblib.dump(
    scaler,
    "processed_data/scaler.pkl"
)

joblib.dump(
    label_encoders,
    "processed_data/label_encoders.pkl"
)


print("\n================================")
print("Data Preprocessing Completed!")
print("Processed files saved successfully")
print("================================")