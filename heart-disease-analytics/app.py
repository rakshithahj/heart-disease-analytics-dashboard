import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Heart Disease Analytics",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Analytics Platform")

selected = option_menu(
    menu_title=None,
    options=[
        "Dashboard",
        "Analytics",
        "Prediction",
        "Explainability"
    ],
    icons=[
        "bar-chart",
        "graph-up",
        "activity",
        "cpu"
    ],
    orientation="horizontal"
)

st.info(
    "Predict Heart Disease Risk with Deep Analytics"
)
