import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")

st.title("Dementia Detection App")

age = st.number_input("Age")
education = st.number_input("Education Level")
memory = st.number_input("Memory Score")

if st.button("Predict"):
    prediction = model.predict([[age, education, memory]])
    st.write("Prediction:", prediction)
