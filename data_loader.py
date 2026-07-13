import pandas as pd

def load_data():

    file_path = "data/hotel_bookings.csv.csv"

    df = pd.read_csv(file_path)

    print("✅ Dataset loaded successfully!")
    print("Dataset Shape:", df.shape)

    return df


if __name__ == "__main__":
    df = load_data()
    print(df.head())