import numpy as np
from numpy.core.numeric import binary_repr
import tensorflow as tf


def onehot(i):
    if i % 2 == 0:
        return np.array([0, 1])
    else:
        return np.array([1, 0])


x_train = np.array([np.array(list(binary_repr(i, 11))) for i in range(10, 900)])
y_train = np.array([onehot(i) for i in range(10, 900)])

x_test = np.array([np.array(list(binary_repr(i, 11))) for i in range(900, 1000)])
y_test = np.array([onehot(i) for i in range(900, 1000)])

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(11,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation="relu"),
    ]
)

loss_fn = tf.losses.CategoricalCrossentropy(from_logits=True)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
model.fit(x_train, y_train, epochs=10)


def isEvenDeep(i) -> bool:
    val = np.array(list(binary_repr(i, 11))).reshape(1, -1)
    if model.predict(val)[0][1] > 0:
        return True
    else:
        return False

for i in range(10):
  print(f"{i} is even? {isEvenDeep(i)}")
