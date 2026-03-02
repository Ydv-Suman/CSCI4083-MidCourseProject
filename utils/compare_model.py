import sys
from pathlib import Path

# Add project root to path so utils imports work when run from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from utils.data_process import load_train_dataset, load_test_dataset

def load_rf_model(path: str):
    return joblib.load(path)


def load_mlp_model(path: str):
    return tf.keras.models.load_model(path)


def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    return accuracy, precision, recall


def compare_models(
    train_path: str = "../data/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv",
    test_path: str = "../data/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv",
    rf_path: str = "../baseline_model/baseline_model.pkl",
    mlp_path: str = "../multilayer_model/multilayer_model.keras",
):
    X_train, _ = load_train_dataset(train_path)
    X_test, y_test = load_test_dataset(test_path)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = load_rf_model(rf_path)
    mlp = load_mlp_model(mlp_path)

    y_pred_rf = rf.predict(X_test_scaled)
    y_pred_mlp = np.argmax(mlp.predict(X_test), axis=1)
    rf_metrics = compute_metrics(y_test, y_pred_rf)
    mlp_metrics = compute_metrics(y_test, y_pred_mlp)

    print(f"Random Forest:\n Accuracy : {rf_metrics[0]} \nPrecision: {rf_metrics[1]} \n Recall : {rf_metrics[2]}")
    print(f"MLP:\n Accuracy : {mlp_metrics[0]} \nPrecision: {mlp_metrics[1]} \n Recall : {mlp_metrics[2]}")


if __name__ == "__main__":
    compare_models()