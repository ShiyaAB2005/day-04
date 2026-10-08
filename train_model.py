import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_hygiene_data.csv")

print("Cleaned Dataset:")
print(df)

# Separate features and target
X = df.drop("risk", axis=1)
y = df["risk"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestClassifier(random_state=42)

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# Save model
joblib.dump(model, "models/hygiene_model.pkl")

print("\nModel trained successfully.")
print("Model saved successfully.")