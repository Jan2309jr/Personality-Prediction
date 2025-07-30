import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('randomforest_model.pkl')  # Use your actual model filename

st.set_page_config(page_title="Personality Predictor", layout="centered")
st.title("🧠 Personality Prediction App")
st.subheader("Based on your social interaction behavior")

# UI inputs
def user_input_features():
    time_spent_alone = st.slider("Time Spent Alone (hours/day)", min_value=0.0, max_value=24.0, value=3.0, step=0.5)
    stage_fear = st.selectbox("Do you fear speaking on stage?", ["Yes", "No"])
    social_event_attendance = st.slider("Social Event Attendance (per month)", min_value=0.0, max_value=30.0, value=5.0, step=1.0)
    going_outside = st.slider("Going Outside (days/week)", min_value=0.0, max_value=7.0, value=3.0, step=1.0)
    drained_after_socializing = st.selectbox("Do you feel drained after socializing?", ["Yes", "No"])
    friends_circle_size = st.slider("Friends Circle Size", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
    post_frequency = st.slider("Social Media Post Frequency (per week)", min_value=0.0, max_value=20.0, value=5.0, step=1.0)

    data = {
        "Time_spent_Alone": time_spent_alone,
        "Stage_fear": stage_fear,
        "Social_event_attendance": social_event_attendance,
        "Going_outside": going_outside,
        "Drained_after_socializing": drained_after_socializing,
        "Friends_circle_size": friends_circle_size,
        "Post_frequency": post_frequency
    }
    return pd.DataFrame([data])

# Get user inputs
input_df = user_input_features()

# Encoding categorical inputs
input_df['Stage_fear'] = input_df['Stage_fear'].map({'Yes': 1, 'No': 0})
input_df['Drained_after_socializing'] = input_df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})

# Display user inputs
st.subheader("User Input:")
st.write(input_df)

# Prediction
if st.button("Predict Personality"):
    prediction = model.predict(input_df)[0]
    st.success(f"🧬 Predicted Personality: **{prediction}**")
