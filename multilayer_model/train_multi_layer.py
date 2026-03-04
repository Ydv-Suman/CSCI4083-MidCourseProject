import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.data_process import load_train_dataset
from multi_layer import build_cnn
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np


def train(train_csv, epochs=100, batch_size=64):
    x_data, y_data = load_train_dataset(train_csv)
    x_data = np.array(x_data, dtype=np.float32) / 255.0
    x_data = x_data.reshape(-1, 28, 28, 1)  # CNN expects (N, 28, 28, 1)
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    model = build_cnn()
    model.summary()
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True,),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
                ]

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    # Save CNN model inside the multilayer_model folder so dashboard.py can load it
    save_path = os.path.join(os.path.dirname(__file__), "multilayer_model.keras")
    model.save(save_path)
    print(f"Model saved as {save_path}")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, "data", "sign-language-mnist", "sign_mnist_train", "sign_mnist_train.csv")
    train(train_csv=csv_path)
