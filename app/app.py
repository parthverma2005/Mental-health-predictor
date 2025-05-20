import streamlit as st
import pandas as pd
from utils import preprocess_input, load_model
import plotly.express as px

# Load model once
model = load_model("models/mental_health_model.pkl")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'risk_history' not in st.session_state:
    st.session_state.risk_history = []

if not st.session_state.logged_in:
    st.set_page_config(page_title="Login | Mental Health App", layout="centered")
    st.title("🔐 Login to Access the Mental Health Risk Predictor")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    correct_username = "user"
    correct_password = "password123"

    if st.button("Login"):
        if username == correct_username and password == correct_password:
            st.session_state.logged_in = True
            st.experimental_rerun()
        else:
            st.error("Incorrect username or password. Please try again.")
else:
    st.set_page_config(page_title="🧠 Mental Health Predictor", layout="centered")
    st.title("🧠 Mental Health Risk Predictor")
    st.markdown("This tool helps assess mental health risk based on daily life factors.")

    family_history = st.selectbox("Family History of Mental Illness?", ["Yes", "No"])
    growing_stress = st.selectbox("Growing Stress?", ["Yes", "No"])
    changes_habits = st.selectbox("Changes in Habits?", ["Yes", "No"])
    mood_swings = st.selectbox("Mood Swings?", ["Yes", "No"])
    coping_struggles = st.selectbox("Coping Struggles?", ["Yes", "No"])
    work_interest = st.selectbox("Interest in Work?", ["Yes", "No"])
    social_weakness = st.selectbox("Social Weakness?", ["Yes", "No"])
    self_employed = st.selectbox("Self Employed?", ["Yes", "No"])
    days_indoors = st.slider("🕒 Days Spent Indoors", min_value=0, max_value=90, value=15, step=1)
    mental_health_history = st.selectbox("Mental Health History?", ["Yes", "No"])
    care_options = st.selectbox("Care Options Available?", ["Yes", "No"])

    if st.button("Predict"):
        features = preprocess_input(
            family_history, growing_stress, changes_habits, mood_swings,
            coping_struggles, work_interest, social_weakness, self_employed,
            days_indoors, mental_health_history, care_options, model
        )

        prediction = model.predict([features])[0]
        proba = model.predict_proba([features])[0][1]

        st.session_state.risk_history.append(proba)

        if prediction == 0:
            st.success(f"✅ Low Risk ({proba:.2%} probability of risk)")
        else:
            st.warning(f"⚠️ High Risk ({proba:.2%} probability of risk)")

        df_history = pd.DataFrame({
            "Prediction #": list(range(1, len(st.session_state.risk_history) + 1)),
            "Risk Probability": st.session_state.risk_history
        })

        fig_line = px.line(df_history, x="Prediction #", y="Risk Probability",
                           title="📈 Mental Health Risk Probability Over Time",
                           markers=True, range_y=[0, 1])
        st.plotly_chart(fig_line)
