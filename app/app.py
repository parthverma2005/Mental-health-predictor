import streamlit as st
import pandas as pd
from utils import preprocess_input, load_model
import plotly.express as px
import smtplib
from email.mime.text import MIMEText

# Load model
model = load_model("../models/xgb_mental_health_model.pkl")

# Session state for login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Login to Mental Health Risk Predictor")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        email = st.text_input("Your Gmail (to receive report)")
        submit = st.form_submit_button("Login")

    # Dummy credentials
    valid_username = "user"
    valid_password = "password123"

    if submit:
        if username == valid_username and password == valid_password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.email = email
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Incorrect username or password.")

# Main app content after login
if st.session_state.logged_in:
    st.title("🧠 Mental Health Risk Predictor")
    st.markdown("This tool helps assess mental health risk based on daily life factors.")

    # Inputs
    family_history = st.selectbox("Family History of Mental Illness?", ["Yes", "No"])
    growing_stress = st.selectbox("Growing Stress?", ["Yes", "No"])
    changes_habits = st.selectbox("Changes in Habits?", ["Yes", "No"])
    mood_swings = st.selectbox("Mood Swings?", ["Yes", "No"])
    coping_struggles = st.selectbox("Coping Struggles?", ["Yes", "No"])
    work_interest = st.selectbox("Interest in Work?", ["Yes", "No"])
    social_weakness = st.selectbox("Social Weakness?", ["Yes", "No"])
    self_employed = st.selectbox("Self Employed?", ["Yes", "No"])
    mental_health_history = st.selectbox("Mental Health History?", ["Yes", "No"])
    care_options = st.selectbox("Care Options Available?", ["Yes", "No"])

    if "risk_history" not in st.session_state:
        st.session_state.risk_history = []

    if st.button("Predict"):
        features = preprocess_input(
            family_history, growing_stress, changes_habits, mood_swings,
            coping_struggles, work_interest, social_weakness, self_employed,
            # Days Indoors removed
            mental_health_history, care_options, model
        )

        prediction = model.predict([features])[0]
        proba = model.predict_proba([features])[0][1]

        st.session_state.risk_history.append(proba)

        if prediction == 0:
            result = f"✅ Low Risk ({proba:.2%} probability of risk)"
            st.success(result)
        else:
            result = f"⚠️ High Risk ({proba:.2%} probability of risk)"
            st.warning(result)
            st.markdown("""### 💡 Helpful Resources and Recommendations

- Practice mindfulness, meditation, or other stress-relief techniques regularly.  
- Communicate openly with trusted friends, family members, or support groups.  
- Consider seeking professional guidance from mental health specialists when needed.  
- Utilize available helplines and online counseling services for immediate support.  

Remember, prioritizing your mental health is a sign of strength and resilience.""")

        # Risk line chart
        df_history = pd.DataFrame({
            "Prediction #": list(range(1, len(st.session_state.risk_history) + 1)),
            "Risk Probability": st.session_state.risk_history
        })

        fig_line = px.line(df_history, x="Prediction #", y="Risk Probability",
                           title="📈 Mental Health Risk Probability Over Time",
                           markers=True, range_y=[0, 1])
        st.plotly_chart(fig_line)

        # ✉️ Send Gmail report
        try:
            sender_email = "vermaparth2005@gmail.com"         # Replace with your Gmail
            sender_password = "slrb huyj azcz wmgz"        # App password from Google
            receiver_email = st.session_state.email

            message = MIMEText(f"Hello {st.session_state.username},\n\nYour recent mental health risk result is:\n{result}")
            message["Subject"] = "🧠 Mental Health Risk Report"
            message["From"] = sender_email
            message["To"] = receiver_email

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(sender_email, sender_password)
                server.send_message(message)

            st.info(f"📩 Report sent to {receiver_email}")
        except Exception as e:
            st.error(f"Email failed to send: {e}")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()
