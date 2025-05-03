import streamlit as st
import pandas as pd
import joblib  # or pickle
import numpy as np

# Load trained model
model = joblib.load('model.pkl')

st.title("Heart Disease Prediction App")

# Collect user input
age = st.slider("Age", 20, 80, 45)
sex = st.selectbox("Sex", ['Male', 'Female'])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure", 90, 200, 120)
chol = st.slider("Serum Cholesterol (mg/dl)", 100, 400, 200)

# Convert inputs to numerical
sex_val = 1 if sex == 'Male' else 0

# Create feature array
input_data = np.array([[age, sex_val, cp, trestbps, chol]])

# Predict
if st.button("Predict"):
    result = model.predict(input_data)
    st.success("Prediction: " + ("Heart Disease" if result[0] == 1 else "No Heart Disease"))