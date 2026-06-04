# pages/3_Patient_Prediction.py

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Patient Prediction",
    page_icon="❤️",
    layout="wide"
)

st.title("🩺 Heart Disease Risk Prediction")

# =====================================
# LOAD DATA
# =====================================

@st.cache_data
def load_data():
    return pd.read_csv("data/heart.csv")

df = load_data()

# =====================================
# TRAIN MODEL
# =====================================

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

nominal_cols = [
    "Sex",
    "ChestPainType",
    "RestingECG",
    "ST_Slope"
]

ordinal_cols = [
    "ExerciseAngina"
]

numeric_cols = [
    "Age",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "MaxHR",
    "Oldpeak"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "nominal",
            OneHotEncoder(handle_unknown="ignore"),
            nominal_cols
        ),
        (
            "ordinal",
            OrdinalEncoder(),
            ordinal_cols
        ),
        (
            "numeric",
            StandardScaler(),
            numeric_cols
        )
    ]
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", KNeighborsClassifier(n_neighbors=10))
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model.fit(X_train, y_train)

# =====================================
# USER INPUT
# =====================================

st.subheader("Enter Patient Information")

col1, col2 = st.columns(2)

with col1:

    age = st.slider(
        "Age",
        20,
        90,
        50
    )

    sex = st.selectbox(
        "Sex",
        ["M", "F"]
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "ASY", "TA"]
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure",
        80,
        250,
        120
    )

    cholesterol = st.number_input(
        "Cholesterol",
        0,
        700,
        200
    )

with col2:

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120",
        [0, 1]
    )

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "LVH", "ST"]
    )

    max_hr = st.slider(
        "Maximum Heart Rate",
        60,
        220,
        150
    )

    exercise_angina = st.selectbox(
        "Exercise Angina",
        ["N", "Y"]
    )

    oldpeak = st.slider(
        "Old Peak",
        0.0,
        6.5,
        1.0
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

# =====================================
# PREDICT BUTTON
# =====================================

if st.button("🔍 Predict Risk"):

    patient_data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "ChestPainType": [chest_pain],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs],
        "RestingECG": [resting_ecg],
        "MaxHR": [max_hr],
        "ExerciseAngina": [exercise_angina],
        "Oldpeak": [oldpeak],
        "ST_Slope": [st_slope]
    })

    prediction = model.predict(patient_data)[0]

    probability = model.predict_proba(patient_data)[0]

    risk_probability = probability[1] * 100

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            f"⚠ High Risk of Heart Disease ({risk_probability:.2f}%)"
        )

    else:

        st.success(
            f"✅ Low Risk of Heart Disease ({100-risk_probability:.2f}%)"
        )

    st.progress(float(risk_probability / 100))

    st.metric(
        "Risk Score",
        f"{risk_probability:.2f}%"
    )

    # =====================================
    # PATIENT SUMMARY
    # =====================================

    st.subheader("Patient Summary")

    st.dataframe(
        patient_data,
        use_container_width=True
    )

# =====================================
# FOOTER
# =====================================

st.divider()

st.caption(
    "Heart Disease Prediction System | Streamlit + Scikit-Learn"
)
