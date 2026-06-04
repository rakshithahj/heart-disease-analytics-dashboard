import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="SHAP Explainability",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 SHAP Explainability Dashboard")
st.markdown("Understand which features influence Heart Disease predictions")

# ======================================
# LOAD DATA
# ======================================

@st.cache_data
def load_data():
    return pd.read_csv("data/heart.csv")

df = load_data()

# ======================================
# ENCODE CATEGORICAL FEATURES
# ======================================

data = df.copy()

for col in data.select_dtypes(include="object").columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

X = data.drop("HeartDisease", axis=1)
y = data["HeartDisease"]

# ======================================
# TRAIN MODEL
# ======================================

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

model.fit(X_train, y_train)

# ======================================
# SHAP EXPLAINER
# ======================================

st.subheader("Feature Importance Using SHAP")

explainer = shap.TreeExplainer(model)

sample_data = X_test.iloc[:100]

shap_values = explainer.shap_values(sample_data)

# ======================================
# SHAP SUMMARY PLOT
# ======================================

st.markdown("### Global Feature Importance")

fig, ax = plt.subplots(figsize=(10, 6))

shap.summary_plot(
    shap_values,
    sample_data,
    show=False
)

st.pyplot(fig)

# ======================================
# BAR SUMMARY
# ======================================

st.markdown("### Mean Impact of Features")

fig2, ax2 = plt.subplots(figsize=(10, 6))

shap.summary_plot(
    shap_values,
    sample_data,
    plot_type="bar",
    show=False
)

st.pyplot(fig2)

# ======================================
# SINGLE PATIENT EXPLANATION
# ======================================

st.markdown("### Individual Patient Explanation")

patient_index = st.slider(
    "Select Patient",
    0,
    len(sample_data)-1,
    0
)

fig3, ax3 = plt.subplots(figsize=(12, 6))

shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[1][patient_index],
        base_values=explainer.expected_value[1],
        data=sample_data.iloc[patient_index],
        feature_names=sample_data.columns
    ),
    show=False
)

st.pyplot(fig3)

# ======================================
# FEATURE RANKING TABLE
# ======================================

st.markdown("### Feature Ranking")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": abs(shap_values[1]).mean(axis=0)
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

st.dataframe(
    importance,
    use_container_width=True
)

# ======================================
# INTERPRETATION
# ======================================

st.markdown("### Interpretation Guide")

st.success("""
• Red values push prediction toward Heart Disease.

• Blue values push prediction toward Healthy.

• Larger SHAP values indicate stronger influence.

• Top-ranked features are the most important risk indicators.
""")
