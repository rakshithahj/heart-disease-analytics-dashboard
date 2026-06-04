import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Heart Disease Analytics",
    page_icon="❤️",
    layout="wide"
)

# =====================================
# LOAD DATA
# =====================================

@st.cache_data
def load_data():
    return pd.read_csv("data/heart.csv")

df = load_data()

# =====================================
# TITLE
# =====================================

st.title("📊 Heart Disease Data Analytics")
st.markdown("Comprehensive Analysis of Heart Disease Dataset")

st.divider()

# =====================================
# KPI SECTION
# =====================================

total_patients = len(df)
heart_cases = int(df["HeartDisease"].sum())
healthy_cases = total_patients - heart_cases
avg_age = round(df["Age"].mean(), 1)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Patients",
    f"{total_patients:,}"
)

col2.metric(
    "Heart Disease Cases",
    heart_cases
)

col3.metric(
    "Healthy Patients",
    healthy_cases
)

col4.metric(
    "Average Age",
    avg_age
)

st.divider()

# =====================================
# DATA PREVIEW
# =====================================

st.subheader("Dataset Preview")

st.dataframe(
    df.head(),
    use_container_width=True
)

# =====================================
# HEART DISEASE DISTRIBUTION
# =====================================

st.subheader("Heart Disease Distribution")

fig = px.pie(
    df,
    names="HeartDisease",
    hole=0.5,
    color="HeartDisease",
    title="Heart Disease vs Healthy"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================
# AGE DISTRIBUTION
# =====================================

st.subheader("Age Distribution")

fig = px.histogram(
    df,
    x="Age",
    color="HeartDisease",
    marginal="box",
    nbins=25,
    title="Age Distribution by Disease Status"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================
# CHEST PAIN ANALYSIS
# =====================================

st.subheader("Chest Pain Type Analysis")

cp = (
    df.groupby("ChestPainType")
    ["HeartDisease"]
    .mean()
    .reset_index()
)

cp["HeartDisease"] = cp["HeartDisease"] * 100

fig = px.bar(
    cp,
    x="ChestPainType",
    y="HeartDisease",
    text_auto=".1f",
    title="Heart Disease Percentage by Chest Pain Type"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================
# CHOLESTEROL ANALYSIS
# =====================================

st.subheader("Cholesterol Analysis")

fig = px.box(
    df,
    x="HeartDisease",
    y="Cholesterol",
    color="HeartDisease",
    title="Cholesterol Levels"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================
# MAX HR ANALYSIS
# =====================================

st.subheader("Maximum Heart Rate Analysis")

fig = px.scatter(
    df,
    x="Age",
    y="MaxHR",
    color="HeartDisease",
    size="Cholesterol",
    hover_data=["Sex"],
    title="Age vs Maximum Heart Rate"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================
# CORRELATION MATRIX
# =====================================

st.subheader("Correlation Heatmap")

numeric_df = df.select_dtypes(include=np.number)

corr = numeric_df.corr()

fig = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    aspect="auto"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================
# INSIGHTS
# =====================================

st.subheader("Key Insights")

disease_rate = (
    df["HeartDisease"].mean() * 100
)

avg_positive_age = (
    df[df["HeartDisease"] == 1]["Age"]
    .mean()
)

avg_negative_age = (
    df[df["HeartDisease"] == 0]["Age"]
    .mean()
)

st.success(
    f"📌 {disease_rate:.1f}% of patients have heart disease."
)

st.info(
    f"📌 Average age of heart disease patients: {avg_positive_age:.1f} years."
)

st.info(
    f"📌 Average age of healthy patients: {avg_negative_age:.1f} years."
)

highest_risk_cp = (
    cp.sort_values(
        "HeartDisease",
        ascending=False
    )
    .iloc[0]["ChestPainType"]
)

st.warning(
    f"📌 Highest-risk chest pain category: {highest_risk_cp}"
)

# =====================================
# DOWNLOAD DATA
# =====================================

st.divider()

csv = df.to_csv(index=False)

st.download_button(
    label="⬇ Download Dataset",
    data=csv,
    file_name="heart_disease_dataset.csv",
    mime="text/csv"
)
