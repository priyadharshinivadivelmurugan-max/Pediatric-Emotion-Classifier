# 👶 Pediatric Emotion Classifier using Deep Learning

> An AI-powered infant cry emotion recognition system that leverages **Deep Learning**, **Audio Signal Processing**, and **Transfer Learning** to automatically identify an infant's emotional state from cry recordings.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras)
![Flask](https://img.shields.io/badge/Flask-Backend-black?logo=flask)
![Kivy](https://img.shields.io/badge/Kivy-Frontend-green)
![License](https://img.shields.io/badge/License-Academic-blue)

---

# 📖 Overview

Infants communicate primarily through crying, making it difficult for caregivers to determine the exact reason behind their distress. Misinterpreting these cries may lead to delayed responses or inappropriate care.

This project presents an **AI-based Pediatric Emotion Classification System** that analyzes infant cry recordings and predicts the corresponding emotional state using a **MobileNetV2-based Deep Learning model**.

The system combines **audio preprocessing**, **Mel Spectrogram generation**, **Transfer Learning**, and a **real-time prediction interface** to provide fast and accurate emotion recognition. A desktop/mobile application built using **Kivy** communicates with a **Flask REST API**, enabling real-time monitoring and prediction from recorded or uploaded audio samples.

---

# ✨ Features

- 🎤 Real-time infant cry monitoring
- 📂 Upload WAV audio files for prediction
- 🧠 MobileNetV2 Deep Learning classifier
- 🎵 Automatic Mel Spectrogram generation
- ⚡ Flask REST API for inference
- 🖥️ Cross-platform Kivy application
- 📱 Android-compatible client
- 📊 Confidence score for every prediction
- 📩 Optional SMS alert system using Twilio
- 🔄 Real-time prediction pipeline

---

# 🏗️ System Architecture

```
Infant Cry Audio
        │
        ▼
Audio Preprocessing
        │
        ▼
Mel Spectrogram Generation
        │
        ▼
Image Normalization
        │
        ▼
MobileNetV2 Model
        │
        ▼
Emotion Prediction
        │
        ▼
Flask REST API
        │
        ▼
Kivy Desktop / Android Application
```

---

# 🧠 Methodology

The proposed framework follows a complete deep learning pipeline for infant cry emotion recognition.

### 1️⃣ Data Collection

The dataset was prepared by combining multiple publicly available infant cry datasets, including:

- Donate-A-Cry Corpus
- Infant Cry Dataset (Kaggle)

To improve model generalization, additional preprocessing and augmentation techniques were applied to increase the diversity of cry samples.

---

### 2️⃣ Audio Preprocessing

Every audio sample undergoes the following preprocessing pipeline before being used for training or prediction:

- Convert stereo audio to mono
- Resample to **16 kHz**
- Extract a **4-second** audio segment
- Generate Mel Spectrogram
- Convert Power Spectrogram to Decibel Scale
- Normalize pixel values
- Resize to **256 × 256**
- Convert grayscale spectrogram into a 3-channel RGB image

This preprocessing pipeline ensures that inference data exactly matches the training data.

---

### 3️⃣ Deep Learning Model

Instead of training a CNN from scratch, **MobileNetV2** was used as the backbone architecture through **Transfer Learning**.

The model learns discriminative spectral patterns from Mel Spectrogram images and classifies infant cries into multiple emotional categories.

| Property | Value |
|----------|-------|
| Architecture | MobileNetV2 |
| Framework | TensorFlow / Keras |
| Input Size | 256 × 256 × 3 |
| Sampling Rate | 16 kHz |
| Audio Duration | 4 Seconds |

---

### 4️⃣ Real-Time Deployment

The trained model is deployed using a **Flask backend server**.

The Kivy application provides:

- Audio recording
- File upload
- Live monitoring
- Prediction visualization
- Alert acknowledgement
- SMS notification support

The application communicates with the Flask server through REST APIs for real-time inference.

---

# 🎯 Supported Emotion Classes

The trained model recognizes the following infant emotional states:

- 😊 Awake
- 🤕 Belly Pain
- 🍼 Burping
- 😟 Discomfort
- 🤗 Hug
- 🥛 Hungry
- 😂 Laugh
- 😴 Silence
- 😪 Tired

---

# 📂 Project Structure

```
Pediatric-Emotion-Classifier/
│
├── client/
│   └── main.py
│
├── server/
│   └── server.py
│
├── notebook/
│   └── MobileNet_Training.ipynb
│
├── model/
│   ├── infant_cry_model.h5
│   └── README.md
│
├── sample_audio/
├── screenshots/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/Pediatric-Emotion-Classifier.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

### Start the Flask Server

```bash
cd server
python server.py
```

Server URL

```
http://localhost:5000
```

---

### Launch the Kivy Client

```bash
cd client
python main.py
```

---

# 📡 REST API

### POST `/predict`

Upload an audio (.wav) file.

Response

```json
{
  "label": "Hungry",
  "confidence": 0.98
}
```

---

# 🛠️ Technologies Used

### Programming

- Python

### Deep Learning

- TensorFlow
- Keras
- MobileNetV2

### Audio Processing

- Librosa
- NumPy
- SciPy
- SoundFile
- SoundDevice

### Backend

- Flask
- Flask-CORS

### Frontend

- Kivy
- Plyer

### Utilities

- Requests
- Twilio
- OpenCV
- Pillow

---

# 🔮 Future Enhancements

- 🌐 Cloud deployment
- 📱 Native Android APK
- 🎥 Multimodal emotion recognition
- ☁️ IoT-based smart baby monitoring
- 📈 Continuous model improvement with larger datasets
- 🩺 Clinical decision support integration

---

# 👩‍💻 Author

**Priyadharshini V**

Electronics and Communication Engineer
----
# 👩‍💻 Co-Author
**Alphin Kamernisha K N**

Electronics and Communication Engineer 

---

# 📜 License

This project was developed as part of an academic Final Year Project and is intended for educational and research purposes.
