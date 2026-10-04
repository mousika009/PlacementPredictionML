import pandas as pd
from sklearn.model_selection import train_test_split
import config


def create_split():

    print("================================")
    print("TRAIN / TEST SPLIT")
    print("================================")

    df = pd.read_csv(config.RAW_DATA_PATH)

    target = "PlacementStatus"

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Total Records :", len(df))
    print("Training Records:", len(X_train))
    print("Testing Records :", len(X_test))

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    create_split()