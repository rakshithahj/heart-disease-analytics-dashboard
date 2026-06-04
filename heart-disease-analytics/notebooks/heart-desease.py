import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# ----------------------------------
# PAGE CONFIG
# ----------------------------------

st.set_page_config(
    page_title="Heart Disease Analytics",
    page_icon="❤️",
    layout="wide"
)

# ----------------------------------
# LOAD DATA
# ----------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("data/heart.csv")

df = load_data()

# ----------------------------------
# HEADER
# ----------------------------------

st.title("❤️ Heart Disease Analytics Dashboard")
st.markdown("---")

# ----------------------------------
# KPI SECTION
# ----------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Patients",
    len(df)
)

col2.metric(
    "Heart Disease Cases",
    int(df["HeartDisease"].sum())
)

col3.metric(
    "Average Age",
    round(df["Age"].mean(), 1)
)

col4.metric(
    "Average Cholesterol",
    round(df["Cholesterol"].mean(), 1)
)

st.markdown("---")

# ----------------------------------
# SIDEBAR
# ----------------------------------

st.sidebar.title("Navigation")

section = st.sidebar.radio(
    "Select Page",
    [
        "Overview",
        "Analytics",
        "Prediction"
    ]
)

# ==================================
# OVERVIEW
# ==================================

if section == "Overview":

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.subheader("Missing Values")

    st.dataframe(
        df.isnull().sum().reset_index()
    )

# ==================================
# ANALYTICS
# ==================================

elif section == "Analytics":

    st.subheader("Heart Disease Distribution")

    fig = px.pie(
        df,
        names="HeartDisease",
        title="Heart Disease Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Age Distribution")

    fig = px.histogram(
        df,
        x="Age",
        color="HeartDisease",
        marginal="box"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Chest Pain Type Analysis")

    chart = (
        df.groupby("ChestPainType")
        ["HeartDisease"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        chart,
        x="ChestPainType",
        y="HeartDisease",
        color="HeartDisease"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Cholesterol vs Max Heart Rate")

    fig = px.scatter(
        df,
        x="Cholesterol",
        y="MaxHR",
        color="HeartDisease",
        size="Age"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Correlation Heatmap")

    numeric_df = df.select_dtypes(
        include=np.number
    )

    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==================================
# PREDICTION
# ==================================

else:

    st.subheader("Heart Disease Prediction")

    data = df.copy()

    encoders = {}

    for col in data.select_dtypes(
        include="object"
    ).columns:

        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

        encoders[col] = le

    X = data.drop(
        "HeartDisease",
        axis=1
    )

    y = data["HeartDisease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    accuracy = accuracy_score(
        y_test,
        model.predict(X_test)
    )

    st.success(
        f"Model Accuracy: {accuracy:.2%}"
    )

    age = st.slider(
        "Age",
        20,
        90,
        50
    )

    resting_bp = st.slider(
        "Resting BP",
        80,
        220,
        120
    )

    cholesterol = st.slider(
        "Cholesterol",
        100,
        600,
        200
    )

    max_hr = st.slider(
        "Max Heart Rate",
        60,
        220,
        150
    )

    oldpeak = st.slider(
        "Old Peak",
        0.0,
        10.0,
        1.0
    )

    if st.button("Predict Risk"):

        sample = np.array([
            [
                age,
                1,
                0,
                resting_bp,
                cholesterol,
                0,
                0,
                max_hr,
                0,
                oldpeak,
                1
            ]
        ])

        prediction = model.predict(sample)

        probability = model.predict_proba(sample)

        if prediction[0] == 1:

            st.error(
                f"⚠ High Risk ({probability[0][1]*100:.1f}%)"
            )

        else:

            st.success(
                f"✅ Low Risk ({probability[0][0]*100:.1f}%)"
            )

# ----------------------------------
# FOOTER
# ----------------------------------

st.markdown("---")
st.caption(
    "Heart Disease Analytics Dashboard | Streamlit + Machine Learning"
)
