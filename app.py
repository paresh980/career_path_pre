from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import os

app = Flask(__name__)

# ✅ Your Exact API Credentials
VALID_USERNAME = "anirudhsinghsisodiya"
VALID_API_KEY = "34d06f608c95a68484d938503739eb5b"

# Paths for model and encoder
MODEL_PATH = "career_predictor.pkl"
ENCODER_PATH = "career_encoder.pkl"

def authenticate_request(request):
    """Check if the request contains a valid API username and key."""
    username = request.headers.get("Username")
    api_key = request.headers.get("API-Key")

    if username == VALID_USERNAME and api_key == VALID_API_KEY:
        return True
    return False

def train_model():
    """Loads data, trains model, and saves model + encoder."""
    try:
        # Load Dataset
        df = pd.read_csv("career_paths.csv")

        # Combine text features
        df['combined_features'] = df['Skills'] + " " + df['Interests'] + " " + df['Education Level']

        # Encode target labels
        career_encoder = {career: idx for idx, career in enumerate(df["Career Path"].unique())}
        df["Career Path"] = df["Career Path"].map(career_encoder)

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(df["combined_features"], df["Career Path"], test_size=0.2, random_state=42)

        # Model Pipeline (TF-IDF + RandomForest)
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        # Train Model
        pipeline.fit(X_train, y_train)

        # Save Model and Encoder
        joblib.dump(pipeline, MODEL_PATH)
        joblib.dump(career_encoder, ENCODER_PATH)

        return {"message": "Model trained and saved successfully!"}
    except Exception as e:
        return {"error": str(e)}

@app.route('/')
def home():
    return "Career Predictor API is running!"

@app.route('/train', methods=['POST'])
def train():
    """Endpoint to trigger model training (API key required)."""
    if not authenticate_request(request):
        return jsonify({"error": "Unauthorized! Invalid credentials."}), 401
    
    response = train_model()
    return jsonify(response)

@app.route('/predict', methods=['POST'])
def predict():
    """Predicts the career path based on user input (API key required)."""
    if not authenticate_request(request):
        return jsonify({"error": "Unauthorized! Invalid credentials."}), 401

    try:
        # Check if model exists
        if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
            return jsonify({"error": "Model not found! Train the model first using /train."})

        # Load Model and Encoder
        model = joblib.load(MODEL_PATH)
        career_encoder = joblib.load(ENCODER_PATH)

        # Reverse encoder mapping
        career_decoder = {idx: career for career, idx in career_encoder.items()}

        # Get input data
        data = request.get_json()
        skills = data.get("skills", "")
        interests = data.get("interests", "")
        education = data.get("education", "")

        # Combine features
        input_text = f"{skills} {interests} {education}"

        # Make prediction
        prediction_idx = model.predict([input_text])[0]
        predicted_career = career_decoder.get(prediction_idx, "Unknown Career")

        return jsonify({"predicted_career": predicted_career})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
