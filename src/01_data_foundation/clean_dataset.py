import pandas as pd
import os
import config


def clean_dataset():

    print("================================")
    print("DATA CLEANING")
    print("================================")

    # Load raw dataset
    df = pd.read_csv(config.RAW_DATA_PATH)

    print("Original Shape:", df.shape)

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Display missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nAfter Cleaning:")
    print("Shape:", df.shape)

    # Save cleaned dataset
    df.to_csv(
        config.CLEANED_DATA_PATH,
        index=False
    )

    print("\nCleaned dataset saved to:")
    print(config.CLEANED_DATA_PATH)

    return df


if __name__ == "__main__":
    clean_dataset()