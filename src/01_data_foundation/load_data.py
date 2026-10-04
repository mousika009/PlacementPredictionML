import pandas as pd
import config


def load_dataset():
    """
    Load the original placement dataset.
    """
    df = pd.read_csv(config.RAW_DATA_PATH)

    print("================================")
    print("DATASET LOADING")
    print("================================")

    print("Dataset Shape:", df.shape)
    print("\nColumns:")

    for column in df.columns:
        print("-", column)

    return df


if __name__ == "__main__":
    df = load_dataset()

    print("\nFirst 5 Records:")
    print(df.head())