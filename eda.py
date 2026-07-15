# ==========================================
# Exploratory Data Analysis (EDA)
# Hotel Demand Prediction Project
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_data


# Load dataset
df = load_data()


if df is not None:

    print("\n========== Dataset Shape ==========")
    print(df.shape)


    print("\n========== First 5 Rows ==========")
    print(df.head())


    print("\n========== Dataset Information ==========")
    df.info()


    print("\n========== Missing Values ==========")
    print(df.isnull().sum().sort_values(ascending=False))


    print("\n========== Statistical Summary ==========")
    print(df.describe())


    print("\n========== Categorical Summary ==========")
    print(df.describe(include="object"))


    # Target variable analysis
    print("\n========== Cancellation Distribution ==========")
    print(df["is_canceled"].value_counts())


    # ==============================
    # Visualizations
    # ==============================

    # 1. Cancellation Distribution
    plt.figure(figsize=(6,4))
    sns.countplot(
        x="is_canceled",
        data=df
    )
    plt.title("Booking Cancellation Distribution")
    plt.xlabel("Is Cancelled (0 = No, 1 = Yes)")
    plt.ylabel("Count")
    plt.show()


    # 2. Hotel Type Distribution
    plt.figure(figsize=(6,4))
    sns.countplot(
        x="hotel",
        data=df
    )
    plt.title("Hotel Type Distribution")
    plt.xlabel("Hotel")
    plt.ylabel("Bookings")
    plt.show()


    # 3. Lead Time Distribution
    plt.figure(figsize=(8,4))
    sns.histplot(
        data=df,
        x="lead_time",
        bins=50,
        kde=True
    )
    plt.title("Lead Time Distribution")
    plt.xlabel("Lead Time")
    plt.show()


    # 4. Average Cancellation Rate by Hotel
    hotel_cancel = (
        df.groupby("hotel")["is_canceled"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(6,4))
    sns.barplot(
        data=hotel_cancel,
        x="hotel",
        y="is_canceled"
    )
    plt.title("Cancellation Rate by Hotel")
    plt.ylabel("Cancellation Rate")
    plt.show()


    # 5. Correlation Heatmap
    plt.figure(figsize=(14,8))

    numeric_data = df.select_dtypes(include="number")

    sns.heatmap(
        numeric_data.corr(),
        cmap="coolwarm"
    )

    plt.title("Correlation Heatmap")
    plt.show()


else:
    print("❌ Dataset loading failed")