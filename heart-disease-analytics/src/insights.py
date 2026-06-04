# insight.py

import pandas as pd

# Load dataset
df = pd.read_csv("heart.csv")

print("===== HEART DISEASE DATASET INSIGHTS =====\n")

# Dataset shape
print(f"Total Records: {df.shape[0]}")
print(f"Total Features: {df.shape[1] - 1}\n")

# Missing values
print("Missing Values:")
print(df.isnull().sum())
print()

# Heart disease cases
if 'target' in df.columns:
    disease_count = df['target'].sum()
    no_disease_count = len(df) - disease_count

    print(f"Patients with Heart Disease: {disease_count}")
    print(f"Patients without Heart Disease: {no_disease_count}\n")

# Average age
if 'age' in df.columns:
    print(f"Average Age: {df['age'].mean():.2f} years")

# Maximum cholesterol
if 'chol' in df.columns:
    print(f"Maximum Cholesterol: {df['chol'].max()}")

# Maximum resting blood pressure
if 'trestbps' in df.columns:
    print(f"Maximum Resting BP: {df['trestbps'].max()}")

# Gender distribution
if 'sex' in df.columns:
    male = (df['sex'] == 1).sum()
    female = (df['sex'] == 0).sum()

    print(f"\nMale Patients: {male}")
    print(f"Female Patients: {female}")

print("\n===== ANALYSIS COMPLETED =====")
