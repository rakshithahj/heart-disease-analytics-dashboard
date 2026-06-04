# predict.py

import pickle
import numpy as np

# Load trained model
with open("heart_knn.pkl", "rb") as file:
    model = pickle.load(file)

# Example patient data
# Replace these values with actual inputs from your dataset
patient_data = np.array([
    [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
])

# Predict
prediction = model.predict(patient_data)

# Output result
if prediction[0] == 1:
    print("Heart Disease Detected")
else:
    print("No Heart Disease Detected")
