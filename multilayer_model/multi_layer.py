import tensorflow as tf
from tensorflow.keras import layers, regularizers


def build_cnn(input_shape=(28, 28, 1), n_classes=24):
    """Small CNN for Sign Language MNIST (28x28 grayscale).
    
    Dataset has 24 classes: A-I (0-8), K-Y (9-23).
    J and Z are excluded because they require motion gestures.
    """
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, padding="same", kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.1),
        layers.MaxPool2D(2),
        layers.Dropout(0.2),

        layers.Conv2D(64, 3, padding="same", kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.1),
        layers.MaxPool2D(2),
        layers.Dropout(0.25),

        layers.Conv2D(64, 3, padding="same", kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.1),
        layers.GlobalAveragePooling2D(),

        layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4)),
        layers.LeakyReLU(0.1),
        layers.Dropout(0.3),

        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_mlp(input_dim=784, n_classes=24, hidden_units=(512, 256, 128)):
    """MLP alternative for Sign Language MNIST.
    
    Dataset has 24 classes: A-I (0-8), K-Y (9-23).
    J and Z are excluded because they require motion gestures.
    """
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for units in hidden_units:
        model.add(layers.Dense(units, kernel_regularizer=regularizers.l2(1e-4)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.1))
        model.add(layers.Dropout(0.25))

    model.add(layers.Dense(n_classes, activation="softmax")) 
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=1000,
        decay_rate=0.9
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model