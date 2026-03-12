import streamlit as st
import joblib

# Load trained model
model = joblib.load("model.pkl")

# App Title
st.title("Alzheimer's Detection System")

st.write("Enter patient details to predict dementia condition")

# Input fields
age = st.number_input("Age", min_value=50, max_value=100, value=70)
educ = st.number_input("Education (Years)", min_value=0, max_value=25, value=15)
ses = st.number_input("SES (1-5)", min_value=1, max_value=5, value=2)
mmse = st.number_input("MMSE Score", min_value=0, max_value=30, value=25)
etiv = st.number_input("eTIV", min_value=1000, max_value=2000, value=1500)
nwbv = st.number_input("nWBV", min_value=0.60, max_value=0.90, value=0.72)
asf = st.number_input("ASF", min_value=0.9, max_value=1.5, value=1.2)
gender = st.selectbox("Gender", ["Male", "Female"])

# Convert gender to numeric
if gender == "Male":
    gender_value = 0
else:
    gender_value = 1

# Prediction button
if st.button("Predict"):

    prediction = model.predict([[age, educ, ses, mmse, etiv, nwbv, asf, gender_value]])

    # Convert numeric output to text
    if prediction[0] == 1:
        result = "Demented"
        st.error(f"Prediction: {result}")
    else:
        result = "Not Demented"
        st.success(f"Prediction: {result}")
