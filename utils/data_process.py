import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_train_dataset(file_path:str):
    train_df = pd.read_csv(file_path)
    X_train = train_df.drop("label", axis=1)
    y_train = train_df["label"].values
    return X_train, y_train

def load_test_dataset(file_path:str):
    test_df = pd.read_csv(file_path)
    X_test = test_df.drop("label", axis=1)
    y_test = test_df["label"].values
    return X_test, y_test