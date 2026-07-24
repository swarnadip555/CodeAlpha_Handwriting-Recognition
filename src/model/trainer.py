"""
Model training module.
"""

import os

from src.data.preprocessing import preprocess_data
from src.model.architecture import build_cnn_model


MODEL_PATH = "saved_models/cnn_mnist.keras"


def train():

    print("=" * 60)
    print("Loading Dataset...")
    print("=" * 60)

    (x_train, y_train), (x_test, y_test) = preprocess_data()

    print("Dataset Loaded!\n")

    print("=" * 60)
    print("Building CNN...")
    print("=" * 60)

    model = build_cnn_model()

    print("CNN Built Successfully!\n")

    print("=" * 60)
    print("Training Started...")
    print("=" * 60)

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=32,
        verbose=1,
    )

    print("\nTraining Complete!")

    print("=" * 60)
    print("Evaluating Model...")
    print("=" * 60)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    print(f"\nTest Accuracy : {accuracy:.4f}")
    print(f"Test Loss     : {loss:.4f}")

    os.makedirs("saved_models", exist_ok=True)

    model.save(MODEL_PATH)

    print("\nModel Saved Successfully!")
    print(f"Location : {MODEL_PATH}")

    return history


if __name__ == "__main__":
    train()