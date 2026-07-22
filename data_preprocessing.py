# ============================================
# Hotel Booking Demand - Data Preprocessing
# ============================================

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# ============================================
# 1. Load Dataset
# ============================================

print("Loading Dataset...")

file_path = r"hotel_bookings.csv"

df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)


# ============================================
# 2. Remove Unnecessary Columns
# ============================================

print("\nRemoving unnecessary columns...")

df.drop(
    [
        "reservation_status",
        "reservation_status_date"
    ],
    axis=1,
    inplace=True,
    errors="ignore"
)


# ============================================
# 3. Handle Missing Values
# ============================================

# ============================================
# 3. Handle Missing Values (Fixed)
# ============================================

print("\nHandling Missing Values...")

for col in df.columns:

    if df[col].isnull().sum() > 0:

        # If column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):

            df[col] = df[col].fillna(
                df[col].median()
            )

        # If column is categorical/string
        else:

            df[col] = df[col].fillna(
                df[col].mode()[0]
            )


print(
    "Remaining Missing Values:",
    df.isnull().sum().sum()
)
# ============================================
# 5. Feature Engineering
# ============================================

print("\nCreating Features...")


# Total stay duration

df["total_stay"] = (
    df["stays_in_weekend_nights"]
    +
    df["stays_in_week_nights"]
)


# Total guests

df["total_guests"] = (
    df["adults"]
    +
    df["children"]
    +
    df["babies"]
)


# Total special changes

df["total_changes"] = (
    df["booking_changes"]
    +
    df["previous_cancellations"]
)


# Average daily rate per guest

df["adr_per_guest"] = (
    df["adr"] /
    (df["total_guests"] + 1)
)



# ============================================
# 6. Encode Categorical Variables
# ============================================

print("\nEncoding categorical variables...")


categorical_columns = df.select_dtypes(
    include="object"
).columns


encoders = {}


for col in categorical_columns:

    encoder = LabelEncoder()

    df[col] = encoder.fit_transform(
        df[col].astype(str)
    )

    encoders[col] = encoder



# ============================================
# 7. Final Missing Value Check
# ============================================

print("\nFinal Missing Value Check:")

print(
    df.isnull().sum().sum()
)


# Replace infinity values if any

df.replace(
    [np.inf, -np.inf],
    0,
    inplace=True
)



# ============================================
# 8. Split Features and Target
# ============================================

print("\nSplitting Features and Target...")


X = df.drop(
    "is_canceled",
    axis=1
)

y = df["is_canceled"]



print("Feature Shape:", X.shape)
print("Target Shape:", y.shape)



# ============================================
# 9. Train Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



# ============================================
# 10. Feature Scaling
# ============================================

print("\nScaling Features...")


scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)


X_test_scaled = scaler.transform(
    X_test
)



# ============================================
# 11. Save Processed Data
# ============================================

os.makedirs(
    "processed_data",
    exist_ok=True
)



pd.DataFrame(
    X_train_scaled,
    columns=X.columns
).to_csv(
    "processed_data/X_train.csv",
    index=False
)



pd.DataFrame(
    X_test_scaled,
    columns=X.columns
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
    encoders,
    "processed_data/label_encoders.pkl"
)


joblib.dump(
    list(X.columns),
    "processed_data/features.pkl"
)



print("\n================================")
print("DATA PREPROCESSING COMPLETED")
print("All files saved successfully")
print("================================")