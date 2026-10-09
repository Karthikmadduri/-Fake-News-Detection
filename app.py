from flask import Flask, render_template, request
import joblib
import nltk
import re
import socket
from pathlib import Path
from nltk.corpus import stopwords

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.joblib"

# Initialize Flask app
app = Flask(__name__)

# Load trained ML model pipeline
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Make sure best_model.joblib exists in the project root.")

model = joblib.load(MODEL_PATH)

def get_stop_words():
    """Load the same NLTK English stop-word list used during training."""
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))

STOP_WORDS = get_stop_words()

def preprocess_text(text):
    """Clean and preprocess input text to match model training pipeline."""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)   # Remove punctuation
    text = re.sub(r"\d+", "", text)       # Remove digits
    words = [word for word in text.split() if word not in STOP_WORDS]
    return " ".join(words)

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form.get("message", "").strip()
    if not message:
        return render_template("home.html", error="Please enter a news article.")
    
    cleaned_message = preprocess_text(message)
    if not cleaned_message:
        return render_template("home.html", error="Please enter text containing readable words (excluding stopwords/numbers).")
    
    # Model prediction
    raw_prediction = model.predict([cleaned_message])[0]
    pred_label = str(raw_prediction).strip().lower()
    
    # Map to user-friendly label
    is_fake = (pred_label == "fake")
    result = "Fake News" if is_fake else "Real News"
    
    # Calculate confidence score if model supports predict_proba
    confidence = None
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba([cleaned_message])[0]
            classes = list(model.classes_)
            if pred_label in classes:
                idx = classes.index(pred_label)
                confidence = round(probabilities[idx] * 100, 1)
        except Exception:
            confidence = None
            
    return render_template(
        "result.html",
        prediction=result,
        is_fake=is_fake,
        confidence=confidence,
        original_text=message,
        cleaned_text=cleaned_message
    )

def is_port_in_use(port, host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

if __name__ == "__main__":
    target_port = 5000
    if is_port_in_use(target_port):
        print(f"\n[WARNING] Port {target_port} is already in use by another process.")
        target_port = 5001
        print(f"[INFO] Starting server on fallback port {target_port} instead.\n")

    print(f"\n========================================================")
    print(f" Fake News Detection Web App is running!")
    print(f" Open your browser at: http://localhost:{target_port}")
    print(f" or:                   http://127.0.0.1:{target_port}")
    print(f"========================================================\n")
    
    # host='0.0.0.0' allows connections from both localhost and 127.0.0.1
    app.run(host="0.0.0.0", port=target_port, debug=False)
