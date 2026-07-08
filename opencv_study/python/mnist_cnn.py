import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models

# MNIST load
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 28x28 -> 28x28x1, 정규화
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# model
model = models.Sequential([
    layers.Input(shape=(28, 28, 1), name="data"),

    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.MaxPooling2D((2, 2), strides=2, padding="same"),

    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.MaxPooling2D((2, 2), strides=2, padding="same"),

    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dense(10),
    layers.Softmax(name="prob")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("Start learning")

model.fit(
    x_train,
    y_train,
    epochs=20,
    batch_size=100,
    validation_data=(x_test, y_test)
)

print("Learning finished!")

# Keras 모델 저장
model.save("mnistcnn.keras")

print("Save done!")