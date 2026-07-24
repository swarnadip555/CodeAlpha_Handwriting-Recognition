"""
Data loader module.

This module is responsible for loading the datasets used by the project.
"""

from tensorflow.keras.datasets import mnist


def load_mnist():
    """
    Load the MNIST handwritten digit dataset.

    Returns:
        tuple:
            (x_train, y_train), (x_test, y_test)
    """
    return mnist.load_data()