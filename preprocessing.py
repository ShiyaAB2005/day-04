import pandas as pd

# Load dataset
df = pd.read_csv("dataset/hygiene_data.csv")

print("Original Dataset:")
print(df)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Save cleaned dataset
df.to_csv("dataset/cleaned_hygiene_data.csv", index=False)

print("\nData preprocessing completed successfully.")
print("Cleaned dataset saved in dataset folder.")