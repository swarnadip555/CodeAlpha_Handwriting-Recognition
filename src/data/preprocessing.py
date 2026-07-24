"""
Data preprocessing utilities for the MNIST dataset.
"""

import numpy as np

from src.data.loader import load_mnist


def preprocess_data():
    """
    Load and preprocess the MNIST dataset.

    Returns:
        tuple:
            (x_train, y_train), (x_test, y_test)
    """

    (x_train, y_train), (x_test, y_test) = load_mnist()

    # Normalize pixel values
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    # Add channel dimension
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return (x_train, y_train), (x_test, y_test)


def show_preprocessed_info():
    """
    Display information about the preprocessed dataset.
    """

    (x_train, y_train), (x_test, y_test) = preprocess_data()

    print("=" * 60)
    print("PREPROCESSED DATASET")
    print("=" * 60)

    print(f"Training Images : {x_train.shape}")
    print(f"Training Labels : {y_train.shape}")

    print(f"Testing Images  : {x_test.shape}")
    print(f"Testing Labels  : {y_test.shape}")

    print()

    print(f"Min Pixel Value : {x_train.min()}")
    print(f"Max Pixel Value : {x_train.max()}")

    print()

    print(f"Image Shape     : {x_train[0].shape}")

    print("=" * 60)


if __name__ == "__main__":
    show_preprocessed_info()