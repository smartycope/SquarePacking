# Portions of code taken from https://www.tensorflow.org/datasets/keras_example
# Example by www.github.com/AniAggarwal

import tensorflow as tf
import tensorflow_datasets as tfds
from time import time
import os

print("TensorFlow version:", tf.__version__)

(ds_train, ds_test), ds_info = tfds.load(
    "fashion_mnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0, label


batch_size = 512

# Process data
ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.batch(batch_size)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(batch_size)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


def create_and_train_model():
    # Create model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(32, 3, input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(64, 3),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # Run training
    model.fit(
        ds_train,
        epochs=20,
        validation_data=ds_test,
    )


start = time()
create_and_train_model()
end_gpu = time() - start

with tf.device("/device:CPU:0"):
    start = time()
    create_and_train_model()
    end_cpu = time() - start


print(f"\n\n\nTime taken for GPU: {end_gpu}\nTime taken for CPU: {end_cpu}")
