import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/hygiene_model.pkl")

# New hygiene data
new_data = pd.DataFrame({
    "cleanliness_score": [5],
    "odor_score": [5],
    "waste_level": [5],
    "complaints": [4],
    "footfall": [280],
    "hygiene_score": [5]
})

# Predict risk
prediction = model.predict(new_data)

print("Predicted Hygiene Risk:", prediction[0])