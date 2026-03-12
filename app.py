import streamlit as st
import joblib

model = joblib.load("model.pkl")

st.title("Alzheimer's Detection System")

age = st.number_input("Age")
educ = st.number_input("Education (Years)")
ses = st.number_input("SES (1-5)")
mmse = st.number_input("MMSE Score")
etiv = st.number_input("eTIV")
nwbv = st.number_input("nWBV")
asf = st.number_input("ASF")
gender = st.number_input("Gender (0 = Male, 1 = Female)")

if st.button("Predict"):
    prediction = model.predict([[age, educ, ses, mmse, etiv, nwbv, asf, gender]])
    st.success(f"Prediction: {prediction[0]}")
