import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.data_process import load_test_dataset
import tensorflow as tf
import numpy as np


def evaluate(model_path, test_csv):
    model = tf.keras.models.load_model(model_path)
    x_test, y_test = load_test_dataset(test_csv)
    x_test = np.array(x_test, dtype=np.float32) / 255.0
    loss, acc = model.evaluate(x_test, y_test)
    print(f"Test loss: {loss:.4f}, Test accuracy: {acc:.4f}")
    return loss, acc

if __name__ == "__main__":
    evaluate(
        model_path="multilayer_model.keras",
        test_csv="../data/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv",
    )
