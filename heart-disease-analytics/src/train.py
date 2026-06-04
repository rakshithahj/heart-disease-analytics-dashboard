# train.py

from preprocessing import load_and_preprocess_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load and preprocess data
X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data("heart.csv")

# Create KNN model
knn = KNeighborsClassifier(n_neighbors=5)

# Train model
knn.fit(X_train, y_train)

# Predictions
y_pred = knn.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model
with open("heart_knn.pkl", "wb") as file:
    pickle.dump(knn, file)

print("heart_knn.pkl saved successfully!")
