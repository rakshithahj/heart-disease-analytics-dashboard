import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

st.title("🔍 SHAP Explainability")

# Load Data
df = pd.read_csv("data/heart.csv")

X = df.drop("target", axis=1)

# Load Model
with open("heart_knn.pkl", "rb") as file:
    model = pickle.load(file)

st.subheader("Feature Importance")

try:
    explainer = shap.Explainer(model.predict, X)

    shap_values = explainer(X)

    fig, ax = plt.subplots()
    shap.plots.bar(shap_values, show=False)

    st.pyplot(fig)

except Exception as e:
    st.warning(
        "SHAP may not support your KNN model directly."
    )
    st.write(e)
