import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("model.pkl")

st.title("Heart Disease Prediction App")

# User input
age = st.slider("Age", 20, 80, 50)
sex = st.selectbox("Sex", ['Male', 'Female'])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.slider("Cholesterol", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
thalach = st.slider("Max Heart Rate Achieved", 70, 210, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.slider("ST depression", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of ST (0-2)", [0, 1, 2])
ca = st.selectbox("Number of major vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (1=normal, 2=fixed defect, 3=reversible)", [1, 2, 3])

# Convert categorical to numerical
sex = 1 if sex == 'Male' else 0

# Format input
features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]])

# Predict
if st.button("Predict"):
    prediction = model.predict(features)[0]
    result = "Heart Disease" if prediction == 1 else "No Heart Disease"
    st.success(f"Prediction: {result}")
