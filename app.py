import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('model.pkl')

st.title("Heart Disease Prediction App")

# Input fields
age = st.slider("Age", 20, 80, 45)
sex = st.radio("Sex", ['Male', 'Female'])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure", 90, 200, 120)
chol = st.slider("Serum Cholesterol (mg/dl)", 100, 400, 200)

# Convert input
sex_val = 1 if sex == 'Male' else 0
features = np.array([[age, sex_val, cp, trestbps, chol]])

# Predict
if st.button("Predict"):
    prediction = model.predict(features)[0]
    st.success("Prediction: " + ("💔 Heart Disease Detected" if prediction == 1 else "❤️ No Heart Disease"))
