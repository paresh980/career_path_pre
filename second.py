import joblib

# Load the trained model and encoder
model = joblib.load("career_predictor.pkl")
career_encoder = joblib.load("career_encoder.pkl")

# Reverse mapping of encoded labels
decoded_careers = {idx: career for career, idx in career_encoder.items()}

# Function to test the model
def test_model():
    print("\n🔹 Career Prediction Test 🔹")
    skills = input("Enter skills: ")
    interests = input("Enter interests: ")
    education = input("Enter education level: ")

    # Combine input for prediction
    test_input = [skills + " " + interests + " " + education]
    
    # Predict career path
    prediction = model.predict(test_input)[0]

    # Print result
    print("\n🎯 Predicted Career Path:", decoded_careers.get(prediction, "Unknown"))

# Run test
if __name__ == "__main__":
    test_model()
