# ==========================================
# Hotel Demand Prediction Project
# Main Execution File
# ==========================================

from data_loader import load_data


def main():

    # Load dataset
    df = load_data()

    if df is None:
        print("❌ Dataset loading failed")
        return

    print("\n✅ Dataset Loaded Successfully")

    # Display first 5 rows
    print("\n========== First 5 Rows ==========")
    print(df.head())

    # Dataset shape
    print("\n========== Dataset Shape ==========")
    print(df.shape)

    # Column names
    print("\n========== Columns ==========")
    print(df.columns.tolist())

    # Data types
    print("\n========== Data Types ==========")
    print(df.dtypes)

    # Missing values
    print("\n========== Missing Values ==========")
    print(df.isnull().sum().sort_values(ascending=False))

    # Target distribution
    if "is_canceled" in df.columns:
        print("\n========== Target Distribution ==========")
        print(df["is_canceled"].value_counts())


if __name__ == "__main__":
    main()