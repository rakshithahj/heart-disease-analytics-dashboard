# pages/1_Data_Analytics.py

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Data Analytics", page_icon="📊")

st.title("📊 Heart Disease Data Analytics")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/heart.csv")

try:
    df = load_data()

    # Dataset Preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Dataset Information
    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())

    # Column Names
    st.subheader("Column Names")
    st.write(df.columns.tolist())

    # Data Types
    st.subheader("Data Types")
    st.dataframe(pd.DataFrame(df.dtypes, columns=["Data Type"]))

    # Statistical Summary
    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

    # Missing Values
    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum().reset_index().rename(
        columns={"index": "Column", 0: "Missing Values"}
    ))

    # Target Distribution
    if "target" in df
