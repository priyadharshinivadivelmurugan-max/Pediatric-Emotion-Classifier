import os
import tempfile
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify
from twilio.rest import Client
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import librosa, tempfile, os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.image import resize
# ======================
# CONFIG
# ======================
# MODEL_PATH = "mobilenet_good (1).h5"  # change if your model filename differs
SMS_ENABLED = True# set True when Twilio configured

# Twilio config (replace with your credentials)
TWILIO_SID = "#Your Twilio SID"
TWILIO_TOKEN = "#your Twilio Auth Token"
TWILIO_FROM = "# your Twilio number"
TWILIO_TO = "# parent’s phone number"    

# ======================
# INIT
# ======================
app = Flask(__name__)

CORS(app)

@app.route("/")
def home():
    return "Hello! Flask server is running fine 🚀"

# --- load your model ---
MODEL_PATH = "mobilenet_good.h5"
model = load_model(MODEL_PATH)
class_names = ['Burping','Laugh','Silence','Tired','awake','belly pain','discomfort','hug','hungry']

# --- preprocessing: match your training code exactly ---
SAMPLE_RATE = 16000
OFFSET_SEC = 0
DURATION = 4.0
N_MELS = 256
HOP_LENGTH = 512
FMIN = 20
FMAX = SAMPLE_RATE // 2
TOP_DB = 80
TARGET_SIZE = (256, 256)

def audio_to_mel_image(path, sample_rate=SAMPLE_RATE, offset=OFFSET_SEC, duration=DURATION,
                       n_mels=N_MELS, hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX, 
                       top_db=TOP_DB, target_size=TARGET_SIZE):
    # Load audio with offset and duration
    y, sr = librosa.load(path, sr=sample_rate, mono=True, offset=offset, duration=duration)
    
    # Pad if shorter than desired segment
    needed = int(duration * sample_rate)
    if len(y) < needed:
        pad = needed - len(y)
        y = np.pad(y, (0, pad), mode='constant')
    
    # Mel spectrogram (power)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels,
        hop_length=hop_length, fmin=fmin, fmax=fmax
    )
    
    # dB scaling with top_db clamp
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=top_db)
    
    # Normalize to 0-1
    mel_min, mel_max = mel_db.min(), mel_db.max()
    mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-8)
    
    # Convert to tensor and resize using TensorFlow (same as training)
    mel_tensor = tf.convert_to_tensor(mel_norm, dtype=tf.float32)         # [n_mels, T]
    mel_tensor = mel_tensor[tf.newaxis, ..., tf.newaxis]                  # [1, H, W, 1]
    mel_resized = resize(mel_tensor, target_size, method='bilinear')      # [1, H, W, 1]
    
    # Repeat channels to 3 for MobileNet
    mel_rgb = tf.repeat(mel_resized, repeats=3, axis=-1)  # [1, H, W, 3]
    return mel_rgb[0].numpy().astype(np.float32)          # [H, W, 3]

def send_sms(message):
    """Send SMS via Twilio if enabled"""
    if not SMS_ENABLED:
        print("SMS not sent (disabled). Message would be:", message)
        return {"status": "disabled", "message": message}

    try:
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        msg = client.messages.create(
            body=message,
            from_=TWILIO_FROM,
            to=TWILIO_TO
        )
        return {"status": "sent", "sid": msg.sid}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def predict_file(path):
    # Preprocess audio to mel image
    img = audio_to_mel_image(path)        # [H,W,3], 0-1
    
    # Apply MobileNet preprocessing (scale to [0,255] then apply MobileNet preprocessing)
    img = img * 255.0
    img = preprocess_input(img)
    
    # Add batch dimension and predict
    inp = np.expand_dims(img, 0)          # [1,H,W,3]
    probs = model.predict(inp, verbose=0)[0]  # softmax probs
    
    idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    
    return class_names[idx], confidence, probs.tolist()

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "no file part"}), 400
    
    f = request.files['file']
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    f.save(tmp.name)
    
    try:
        label, conf, probs = predict_file(tmp.name)
        return jsonify({
            "label": label, 
            "confidence": conf, 
            #"probabilities": dict(zip(class_names, probs))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(tmp.name)
        except:
            pass

@app.route("/send_alert", methods=["POST"])
def send_alert():
    data = request.json or {}
    label = data.get("label", "unknown")
    conf = data.get("confidence", 0)
    summary = data.get("summary", False)

    if summary:
        message = f"Summary: Baby emotion is {label} ({conf*100:.1f}%)."
    else:
        message = f"ALERT: Baby crying detected as {label} ({conf*100:.1f}%). Please check."

    result = send_sms(message)
    return jsonify(result)


if __name__ == "__main__":
    print("Loading model:", MODEL_PATH)
    app.run(host="0.0.0.0", port=5000, debug=True)

# Load model once at startup
#model = tf.keras.models.load_model(MODEL_PATH)

# # Class labels (set these exactly as your model outputs)
# CLASS_LABELS = ["hungry", "pain", "burping", "discomfort", "other"]


# # ======================
# # HELPERS
# # ======================
# def extract_features(file_path):
#     """Load wav, convert to mel-spectrogram for prediction"""
#     y, sr = librosa.load(file_path, sr=16000)
#     mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
#     mel_db = librosa.power_to_db(mel, ref=np.max)
#     mel_db = np.expand_dims(mel_db, axis=-1)  # add channel
#     mel_db = np.expand_dims(mel_db, axis=0)   # add batch
#     return mel_db
# 
# 
# def predict_audio(file_path):
#     feats = extract_features(file_path)
#     preds = model.predict(feats)
#     idx = np.argmax(preds)
#     label = CLASS_LABELS[idx] if idx < len(CLASS_LABELS) else str(idx)
#     confidence = float(preds[0][idx])
#     return label, confidence


# def send_sms(message):
#     """Send SMS via Twilio if enabled"""
#     if not SMS_ENABLED:
#         print("SMS not sent (disabled). Message would be:", message)
#         return {"status": "disabled", "message": message}
# 
#     try:
#         client = Client(TWILIO_SID, TWILIO_TOKEN)
#         msg = client.messages.create(
#             body=message,
#             from_=TWILIO_FROM,
#             to=TWILIO_TO
#         )
#         return {"status": "sent", "sid": msg.sid}
#     except Exception as e:
#         return {"status": "error", "error": str(e)}
# 

# ======================
# ROUTES
# ======================
# @app.route("/predict", methods=["POST"])
# def predict():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
# 
#     file = request.files["file"]
# 
#     # Save to temp file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#         file.save(tmp.name)
#         tmp_path = tmp.name
# 
#     try:
#         label, conf = predict_audio(tmp_path)
#         return jsonify({"label": label, "confidence": conf})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#     finally:
#         os.remove(tmp_path)
# 
# 
# @app.route("/send_alert", methods=["POST"])
# def send_alert():
#     data = request.json or {}
#     label = data.get("label", "unknown")
#     conf = data.get("confidence", 0)
#     summary = data.get("summary", False)
# 
#     if summary:
#         message = f"Summary: Baby emotion is {label} ({conf*100:.1f}%)."
#     else:
#         message = f"ALERT: Baby crying detected as {label} ({conf*100:.1f}%). Please check."
# 
#     result = send_sms(message)
#     return jsonify(result)
# 
# 
# # ======================
# # MAIN
# # ======================
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)


