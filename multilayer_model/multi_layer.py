import tensorflow as tf
from tensorflow.keras import layers, regularizers


def build_mlp(input_dim=784, n_classes=25, hidden_units=(512, 256, 128)):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for units in hidden_units:
        model.add(layers.Dense(units,kernel_regularizer=regularizers.l2(1e-4)))
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