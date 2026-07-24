"""
CNN architecture for handwritten digit recognition.
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)


def build_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Build and compile a CNN model.

    Args:
        input_shape (tuple): Shape of input images.
        num_classes (int): Number of output classes.

    Returns:
        tensorflow.keras.Model
    """

    model = Sequential(
        [
            Conv2D(
                filters=32,
                kernel_size=(3, 3),
                activation="relu",
                input_shape=input_shape,
            ),

            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(
                filters=64,
                kernel_size=(3, 3),
                activation="relu",
            ),

            MaxPooling2D(pool_size=(2, 2)),

            Flatten(),

            Dense(128, activation="relu"),

            Dropout(0.5),

            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    print("Building CNN model...")

    cnn = build_cnn_model()

    print("Model built successfully!")

    cnn.summary()

    print("Done!")