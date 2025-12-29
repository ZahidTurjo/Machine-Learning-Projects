import streamlit as st
import pandas as pd
import numpy as np
import pickle

# load model
model_path="heart_disease_prediction_model.pkl"

with open(model_path,'rb') as file:
    model,scaler=pickle.load(file)


st.set_page_config(page_title="Heart Disease Prediction",
                   page_icon="🫀", layout="centered")
st.title("🩺 Heart Disease Prediction App")

st.write("This app predicts the liklihood of heart disease based on patient health data")
categorical_cols = [
    'Gender', 'ChestPainType', 'FastingBS', 'RestingECG',
    'ExerciseAngina', 'ST_Slope', 'MajorVessels', 'Thalassemia'
]
numerical_cols = ['Age', 'RestingBp', 'Cholesterol', 'MaxHR', 'ST_Depression']

col1,col2=st.columns(2)

with col1:
    Age=st.number_input("Age",10,100)
    Gender=st.selectbox("Gender",['Male','Female'])
    ChestPainType=st.selectbox(
        "Chest Pain Type (0:Typical, 1:Atypical, 2:Non-Anginal, 3:Asymptomatic)", [0, 1, 2, 3])
    RestingBp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    Cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

with col2:
    RestingECG = st.selectbox("Resting ECG Results", [0, 1, 2])

    MaxHR = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
    ExerciseAngina = st.selectbox("Exercise-Induced Angina", [0, 1])
    ST_Depression = st.number_input("ST Depression", 0.0, 6.0, 1.0, step=0.1)
    ST_Slope = st.selectbox("ST Slope", [0, 1, 2])
    MajorVessels = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3])
    Thalassemia = st.selectbox("Thalassemia (1–3)", [1, 2, 3])
    
# preprocess input
Gender=1 if Gender == "Male" else 0
input_dict = {
    'Age': Age,
    'Gender': Gender,
    'ChestPainType': ChestPainType,
    'RestingBp': RestingBp,
    'Cholesterol': Cholesterol,
    'FastingBS': FastingBS,
    'RestingECG': RestingECG,
    'MaxHR': MaxHR,
    'ExerciseAngina': ExerciseAngina,
    'ST_Depression': ST_Depression,
    'ST_Slope': ST_Slope,
    'MajorVessels': MajorVessels,
    'Thalassemia': Thalassemia
}

input_df = pd.DataFrame([input_dict])

input_encoded = pd.get_dummies(
    input_df, columns=categorical_cols, drop_first=True)
expected_features=model.feature_names_in_
input_encoded = input_encoded.reindex(columns=expected_features, fill_value=0)

input_encoded[numerical_cols] = scaler.fit_transform(
    input_encoded[numerical_cols])
if st.button("Predict Heart Disease Risk"):
    prediction=model.predict(input_encoded)[0]
    if(prediction==1):
        st.error("High risk of Heart Disease detected! Please consult a cardiologist.")
    else:
        st.success("No signs of Heart Disease")
st.markdown(" ")
st.caption("Developed By Z.I. Turjo")
    