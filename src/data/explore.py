"""
Exploratory Data Analysis (EDA) for the MNIST dataset.
"""

import matplotlib.pyplot as plt

from src.data.loader import load_mnist


def show_dataset_info() -> None:
    """Display basic information about the dataset."""
    (x_train, y_train), (x_test, y_test) = load_mnist()

    print("=" * 60)
    print("MNIST DATASET")
    print("=" * 60)
    print(f"Training Images : {x_train.shape}")
    print(f"Training Labels : {y_train.shape}")
    print(f"Testing Images  : {x_test.shape}")
    print(f"Testing Labels  : {y_test.shape}")
    print(f"Image Shape     : {x_train[0].shape}")
    print(f"Number of Classes : {len(set(y_train))}")
    print("=" * 60)


def display_samples(num_samples: int = 10) -> None:
    """Display sample handwritten digits."""
    (x_train, y_train), _ = load_mnist()

    plt.figure(figsize=(12, 4))

    for index in range(num_samples):
        plt.subplot(2, 5, index + 1)
        plt.imshow(x_train[index], cmap="gray")
        plt.title(f"Label: {y_train[index]}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_dataset_info()
    display_samples()