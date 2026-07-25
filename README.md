# ✍️ Handwritten Digit Recognition using CNN

A deep learning-based web application that recognizes handwritten digits drawn by the user in real time. The project uses a **Convolutional Neural Network (CNN)** trained on the **MNIST** dataset and provides predictions through an interactive **Streamlit** interface.

## 🚀 Live Demo

🔗 **Try the application here:**  
https://codealphahandwriting-recognition-rttd9mkngwbfm3452gy9um.streamlit.app/


---

## 📌 Project Overview

This project demonstrates how deep learning can be used for handwritten digit recognition. Users can draw digits directly on a canvas in the web application, and the trained CNN predicts the digit along with its confidence score.

The project is designed with a modular architecture, making it easy to extend for handwritten alphabet and word recognition using datasets such as **EMNIST**.

---

## ✨ Features

- 🎨 Draw digits directly using the mouse
- 🧠 CNN-based handwritten digit recognition
- ⚡ Real-time prediction
- 📊 Confidence score for predictions
- 📈 Probability distribution visualization
- 🖼️ Image preprocessing before prediction
- 🏗️ Modular project structure
- 🌐 Streamlit web application
- 🚀 Ready for deployment

---

## 📂 Project Structure

```text
Handwriting-Recognition/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── saved_models/
│   └── cnn_mnist.keras
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── preprocessing.py
│   │   └── explore.py
│   │
│   ├── model/
│   │   ├── architecture.py
│   │   ├── trainer.py
│   │   ├── predictor.py
│   │   └── evaluator.py
│   │
│   └── utils/
│       └── image_utils.py
│
└── .venv/
```

---

## 🧠 Model Architecture

The CNN consists of:

- Conv2D (32 filters, 3×3)
- MaxPooling2D
- Conv2D (64 filters, 3×3)
- MaxPooling2D
- Flatten
- Dense (128 neurons)
- Dropout (0.5)
- Output Layer (10 classes)

---

## 📊 Dataset

**MNIST Dataset**

- 60,000 Training Images
- 10,000 Testing Images
- Image Size: 28 × 28 pixels
- Classes: 10 (Digits 0–9)

---

## 📈 Results

| Metric | Value |
|---------|-------|
| Test Accuracy | **99.34%** |
| Test Loss | **0.0240** |

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pandas
- OpenCV
- Matplotlib

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/your-username/Handwriting-Recognition.git
```

```bash
cd Handwriting-Recognition
```

---

### Create a virtual environment

```bash
uv venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

---

### Install dependencies

```bash
pip install -r requirements.txt
```

---

### Train the model

```bash
python -m src.model.trainer
```

---

### Run the application

```bash
streamlit run app.py
```

---

## 🎮 Usage

1. Launch the Streamlit application.
2. Draw a handwritten digit using the mouse.
3. Click **Predict**.
4. View:
   - Predicted digit
   - Confidence score
   - Probability distribution
   - Processed image

---

## 🚀 Future Scope

This project can be extended to support:

- Handwritten alphabet recognition using **EMNIST**
- Alphanumeric character recognition
- Handwritten word recognition using **CRNN**
- Sentence recognition using sequence models
- Mobile and cloud deployment

---

## 👨‍💻 Author

**Swarnadip**

Internship Project

---

## 📄 License

This project is intended for educational and learning purposes.
