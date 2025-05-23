import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📰 Fake News Detector")

# User input
text_input = st.text_area("Enter News Article Text")

if st.button("Predict"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform and predict
        transformed_text = vectorizer.transform([text_input])
        prediction = model.predict(transformed_text)

        if prediction[0] == "FAKE":
            st.error("⚠️ This news is predicted to be **FAKE**.")
        else:
            st.success("✅ This news is predicted to be **REAL**.")
