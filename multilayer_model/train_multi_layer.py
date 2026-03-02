import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.data_process import load_train_dataset
from multi_layer import build_mlp
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np


def train(train_csv, epochs=100, batch_size=64):
    x_data, y_data = load_train_dataset(train_csv)
    x_data = np.array(x_data, dtype=np.float32) / 255.0
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42,)
    model = build_mlp()
    model.summary()
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True,),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
                ]

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    model.save("multilayer_model.keras")
    print("Model saved as multilayer_model.keras")


if __name__ == "__main__":
    train(train_csv="../data/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv",)
