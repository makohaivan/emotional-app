"""
app.py
Minimal Flask app that:
- Serves a simple webpage (templates/index.html)
- Accepts a base64 image POST at /predict
- Uses the `fer` package to detect emotion from the image
- Stores each prediction in a small SQLite database (database.db)

Run:
    python app.py
Open in browser: http://127.0.0.1:5000
"""

import base64
import io
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import cv2
from fer import FER

# ---------- Config ----------
DB_PATH = "database.db"         # SQLite file
FER_MT_CNN = True               # set to False if mtcnn causes issues
# ----------------------------

app = Flask(__name__)

# Initialize the FER detector once (reuse for all requests)
detector = FER(mtcnn=FER_MT_CNN)


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


# initialize DB on startup
init_db()


# ---------- Utility: image conversion ----------
def data_url_to_cv2_image(data_url: str):
    """
    Convert a data URL (data:image/png;base64,...) to an OpenCV BGR image (numpy array).
    Returns None on failure.
    """
    try:
        header, encoded = data_url.split(",", 1)  # split off metadata
    except ValueError:
        return None

    binary = base64.b64decode(encoded)
    pil_img = Image.open(io.BytesIO(binary)).convert("RGB")
    arr = np.array(pil_img)               # RGB order (H, W, 3)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr


# ---------- Routes ----------
@app.route("/")
def index():
    """Serve the homepage (templates/index.html)."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON payload: { "image": "<data_url>" }
    Returns JSON: { "emotion": "happy", "confidence": 0.98 } or a helpful error message.
    """
    payload = request.get_json(silent=True)
    if not payload or "image" not in payload:
        return jsonify({"error": "Missing 'image' in JSON body"}), 400

    cv_img = data_url_to_cv2_image(payload["image"])
    if cv_img is None:
        return jsonify({"error": "Invalid image data"}), 400

    # Use FER to detect emotions. Returns list of faces with emotion scores.
    faces = detector.detect_emotions(cv_img)
    if not faces:
        return jsonify({"emotion": None, "confidence": 0.0, "message": "No face detected"})

    # take the first detected face (for simple demos)
    emotions = faces[0].get("emotions", {})
    if not emotions:
        return jsonify({"emotion": None, "confidence": 0.0, "message": "No emotion scores"})

    # find the emotion with the highest score
    emotion = max(emotions, key=emotions.get)
    confidence = float(emotions[emotion])

    # record result in DB
    save_prediction(emotion, confidence)

    return jsonify({"emotion": emotion, "confidence": confidence})


@app.route("/history")
def history():
    """
    Returns last 100 predictions as JSON:
    { "history": [ { "timestamp": "...", "emotion": "...", "confidence": 0.95 }, ... ] }
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
    # debug=True is helpful while developing; set to False in production
    app.run(host="0.0.0.0", port=5000, debug=True)
