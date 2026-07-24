import cv2
import numpy as np


def preprocess_canvas_image(image):
    """
    Convert Streamlit canvas image into an MNIST-style image.
    """

    # Convert RGB -> Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Binary threshold
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    # Find digit
    coords = cv2.findNonZero(thresh)

    if coords is None:
        return np.zeros((28, 28), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)

    digit = thresh[y:y+h, x:x+w]

    # Make square
    size = max(w, h)

    square = np.zeros((size, size), dtype=np.uint8)

    x_offset = (size - w) // 2
    y_offset = (size - h) // 2

    square[y_offset:y_offset+h,
           x_offset:x_offset+w] = digit

    # Resize
    resized = cv2.resize(
        square,
        (20, 20),
        interpolation=cv2.INTER_AREA,
    )

    # Put inside 28x28 canvas
    final = np.zeros((28, 28), dtype=np.uint8)

    final[4:24, 4:24] = resized

    return final