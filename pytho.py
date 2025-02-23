import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Load Dataset
df = pd.read_csv("processed_career_paths.csv")

# Combine all text columns for feature extraction
df['combined_features'] = df['Skills'] + " " + df['Interests'] + " " + df['Education Level']

# Encode target labels (Career Paths)
career_encoder = {career: idx for idx, career in enumerate(df["Career Path"].unique())}
df["Career Path"] = df["Career Path"].map(career_encoder)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(df["combined_features"], df["Career Path"], test_size=0.2, random_state=42)

# TF-IDF + RandomForest Model Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train Model
pipeline.fit(X_train, y_train)

# Save Model and Encoders
joblib.dump(pipeline, "career_predictor.pkl")
joblib.dump(career_encoder, "career_encoder.pkl")

loaded_model = joblib.load("career_predictor.pkl")
sample_input = ["Python Machine Learning AI Bachelor's"]
prediction = loaded_model.predict(sample_input)
print(prediction)

