import pandas as pd
import numpy as np


LABEL_REMAP = {csv_label: model_label for model_label, csv_label in enumerate(
    [i for i in range(26) if i != 9 and i != 25]  
)}


def load_train_dataset(file_path: str):
    train_df = pd.read_csv(file_path)
    X_train = train_df.drop("label", axis=1)
    y_train = train_df["label"].map(LABEL_REMAP).values  
    return X_train, y_train


def load_test_dataset(file_path: str):
    test_df = pd.read_csv(file_path)
    X_test = test_df.drop("label", axis=1)
    y_test = test_df["label"].map(LABEL_REMAP).values  
    return X_test, y_test