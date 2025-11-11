"""
app.py - Fixed version with delayed imports
"""

import base64
import io
import sqlite3
import os
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import cv2

# ---------- Config ----------
DB_PATH = "database.db"
FER_MT_CNN = True
# ----------------------------

app = Flask(__name__)

# DON'T initialize FER here - it blocks the app startup
# detector = FER(mtcnn=FER_MT_CNN)

# ---------- Database helpers ----------
def init_db():
    """Create the predictions table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            emotion TEXT,
            confidence REAL
        )
        """
    )
    conn.commit()
    conn.close()

def save_prediction(emotion: str, confidence: float):
    """Insert a prediction row into the database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO predictions (timestamp, emotion, confidence) VALUES (?, ?, ?)",
        (datetime.utcnow().isoformat(), emotion, float(confidence)),
    )
    conn.commit()
    conn.close()

init_db()

# ---------- Utility: image conversion ----------
def data_url_to_cv2_image(data_url: str):
    """
    Convert a data URL (data:image/png;base64,...) to an OpenCV BGR image (numpy array).
    Returns None on failure.
    """
    try:
        header, encoded = data_url.split(",", 1)
    except ValueError:
        return None

    binary = base64.b64decode(encoded)
    pil_img = Image.open(io.BytesIO(binary)).convert("RGB")
    arr = np.array(pil_img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr

# ---------- Routes ----------
@app.route("/")
def index():
    """Serve the homepage (templates/index.html)."""
    return render_template("index.html")

@app.route("/health")
def health():
    """Health check endpoint that responds immediately."""
    return jsonify({"status": "healthy"})

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON payload: { "image": "<data_url>" }
    """
    # Import FER inside the route to avoid blocking app startup
    from fer import FER
    detector = FER(mtcnn=FER_MT_CNN)
    
    payload = request.get_json(silent=True)
    if not payload or "image" not in payload:
        return jsonify({"error": "Missing 'image' in JSON body"}), 400

    cv_img = data_url_to_cv2_image(payload["image"])
    if cv_img is None:
        return jsonify({"error": "Invalid image data"}), 400

    # Use FER to detect emotions
    faces = detector.detect_emotions(cv_img)
    if not faces:
        return jsonify({"emotion": None, "confidence": 0.0, "message": "No face detected"})

    emotions = faces[0].get("emotions", {})
    if not emotions:
        return jsonify({"emotion": None, "confidence": 0.0, "message": "No emotion scores"})

    emotion = max(emotions, key=emotions.get)
    confidence = float(emotions[emotion])

    save_prediction(emotion, confidence)
    return jsonify({"emotion": emotion, "confidence": confidence})

@app.route("/history")
def history():
    """
    Returns last 100 predictions as JSON
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT timestamp, emotion, confidence FROM predictions ORDER BY id DESC LIMIT 100")
    rows = c.fetchall()
    conn.close()

    data = [{"timestamp": r[0], "emotion": r[1], "confidence": r[2]} for r in rows]
    return jsonify({"history": data})

# ---------- Run ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
