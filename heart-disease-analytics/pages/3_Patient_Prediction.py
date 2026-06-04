import streamlit as st
import numpy as np
import pickle

st.title("🩺 Patient Prediction")

# Load Model
with open("heart_knn.pkl", "rb") as file:
    model = pickle.load(file)

age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex", [0, 1])
cp = st.number_input("Chest Pain Type", 0, 3, 1)
trestbps = st.number_input("Resting Blood Pressure", 80, 250, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar", [0, 1])
restecg = st.number_input("Rest ECG", 0, 2, 1)
thalach = st.number_input("Max Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise Angina", [0, 1])
oldpeak = st.number_input("Old Peak", 0.0, 10.0, 1.0)
slope = st.number_input("Slope", 0, 2, 1)
ca = st.number_input("CA", 0, 4, 0)
thal = st.number_input("Thal", 0, 3, 2)

if st.button("Predict"):

    input_data = np.array([[
        age, sex, cp, trestbps, chol,
        fbs, restecg, thalach,
        exang, oldpeak, slope,
        ca, thal
    ]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")
