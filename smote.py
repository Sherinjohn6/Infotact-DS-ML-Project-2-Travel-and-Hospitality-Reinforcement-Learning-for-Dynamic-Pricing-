# ============================================
# Hotel Booking Demand - SMOTE Handling
# ============================================

import pandas as pd
import os
from collections import Counter
from imblearn.over_sampling import SMOTE


# ============================================
# 1. Load Processed Training Data
# ============================================

print("Loading processed data...")

X_train = pd.read_csv(
    "processed_data/X_train.csv"
)

y_train = pd.read_csv(
    "processed_data/y_train.csv"
).squeeze()


print("Training Shape:", X_train.shape)
print("Target Shape:", y_train.shape)



# ============================================
# 2. Check Class Distribution Before SMOTE
# ============================================

print("\nBefore SMOTE:")
print(Counter(y_train))



# ============================================
# 3. Apply SMOTE
# ============================================

print("\nApplying SMOTE...")

smote = SMOTE(
    random_state=42
)


X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)



# ============================================
# 4. Check Distribution After SMOTE
# ============================================

print("\nAfter SMOTE:")
print(Counter(y_train_smote))


print(
    "\nSMOTE Training Shape:",
    X_train_smote.shape
)



# ============================================
# 5. Save SMOTE Dataset
# ============================================

os.makedirs(
    "processed_data",
    exist_ok=True
)


smote_data = pd.DataFrame(
    X_train_smote,
    columns=X_train.columns
)


smote_data["is_canceled"] = y_train_smote



smote_data.to_csv(
    "processed_data/smote_hotel_bookings.csv",
    index=False
)



print("\n================================")
print("SMOTE COMPLETED SUCCESSFULLY")
print("Balanced dataset saved")
print("================================")