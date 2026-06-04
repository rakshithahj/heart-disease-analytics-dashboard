import streamlit as st
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

st.title("📈 Model Performance")

# Load Dataset
df = pd.read_csv("data/heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load Model
with open("heart_knn.pkl", "rb") as file:
    model = pickle.load(file)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

# Confusion Matrix
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

st.write(cm)

# Classification Report
st.subheader("Classification Report")

report = classification_report(y_test, y_pred)

st.text(report)
