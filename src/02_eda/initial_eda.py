import pandas as pd
import config


def perform_initial_eda():

    df = pd.read_csv(config.RAW_DATA_PATH)

    print("================================")
    print("INITIAL EDA")
    print("================================")

    print("\nDataset Shape:")
    print(df.shape)

    print("\nDataset Information:")
    print(df.info())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nStatistical Summary:")
    print(df.describe(include="all"))

    print("\nFirst 5 Records:")
    print(df.head())


if __name__ == "__main__":
    perform_initial_eda()