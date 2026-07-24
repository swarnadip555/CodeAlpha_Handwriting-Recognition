"""
Test predictor module.
"""

from tensorflow.keras.datasets import mnist

from src.model.predictor import predict

(_, _), (x_test, y_test) = mnist.load_data()

image = x_test[0]
label = y_test[0]

digit, confidence, _ = predict(image)

print("=" * 50)

print("Actual Label    :", label)

print("Predicted Label :", digit)

print(f"Confidence      : {confidence:.4f}")

print("=" * 50)