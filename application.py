import os
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from keras.models import load_model
from PIL import Image

# Flask app
app = Flask(__name__)
CORS(app)

# Use relative paths (EBS-safe)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model/model.keras')
# MODEL_PATH = os.path.join(BASE_DIR, 'model1.keras')
CASCADE_PATH = os.path.join(BASE_DIR, 'model/haarcascade_frontalface_default.xml')

# Load model and face detector
model = load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Emotion labels
labels = ['Angry', 'Crying', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Smirking', 'Surprise']

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_emotion():
    if 'image' not in request.files:
        return jsonify({'emotion': 'No image uploaded'})

    file = request.files['image']
    try:
        img = Image.open(file.stream).convert('RGB')
        open_cv_image = np.array(img)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return jsonify({'emotion': 'No face found'})

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi_gray, (48, 48))
            roi_normalized = roi_resized / 255.0
            roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))

            prediction = model.predict(roi_reshaped)
            emotion = labels[np.argmax(prediction)]
            return jsonify({'emotion': emotion})

        return jsonify({'emotion': 'Face not detected'})
    except Exception as e:
        return jsonify({'emotion': f'Error: {str(e)}'})

# EBS-compatible entry point
application = app

if __name__ == '__main__':
    app.run(debug=True)