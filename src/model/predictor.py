import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "saved_models/cnn_mnist.keras"

model = load_model(MODEL_PATH)


def predict(image):

    image = image.astype(np.float32)

    image /= 255.0

    image = image.reshape(1, 28, 28, 1)

    probs = model.predict(image, verbose=0)[0]

    digit = np.argmax(probs)

    confidence = probs[digit]

    return digit, confidence, probs