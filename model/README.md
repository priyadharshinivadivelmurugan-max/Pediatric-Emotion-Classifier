# Model

This directory contains the trained deep learning model used for infant cry emotion recognition.

## Model Information

- **Model Architecture:** MobileNetV2
- **Framework:** TensorFlow / Keras
- **Input:** Mel Spectrogram Images (256 × 256 × 3)
- **Sampling Rate:** 16 kHz
- **Audio Duration:** 4 seconds
- **Output:** Infant Emotion Class

## Model File

```
infant_cry_model.h5
```

This model was trained using the preprocessing pipeline implemented in the training notebook and is loaded by the Flask inference server during prediction.

## Supported Emotion Classes

- Awake
- Belly Pain
- Burping
- Discomfort
- Hug
- Hungry
- Laugh
- Silence
- Tired

## Usage

Place the model file inside this directory.

The server loads the model automatically using:

```python
MODEL_PATH = "../model/mobilenet_good.h5"
```

No additional configuration is required.
