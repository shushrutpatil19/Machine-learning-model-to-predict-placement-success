import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("placement_model_v2.pkl")
scaler = joblib.load("scaler_v2.pkl")

st.title("🎓 Placement Success Prediction App")
st.write("Enter student details below to predict placement success:")

cgpa = st.number_input("Enter CGPA", min_value=0.0, max_value=10.0, step=0.1)
comm = st.number_input("Communication Skills (0-100)", min_value=0, max_value=100, step=1)
intern_act = st.selectbox("Did Internship?", ['Yes', 'No'])
intern_type = st.selectbox("Internship Type", ['Technical', 'Non-Technical', 'None'])
intern_company = st.selectbox("Internship Company", ['Startup', 'MNC', 'None'])
extra = st.number_input("Extracurricular Activities (0-100)", min_value=0, max_value=100, step=1)

if st.button("Predict Placement"):
    input_df = pd.DataFrame([{
        'CGPA': cgpa,
        'Communication_Skills': comm,
        'Internship_Activity': intern_act,
        'Internship_Type': intern_type,
        'Internship_Company': intern_company,
        'Extracurricular_Activities': extra
    }])

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in ['Internship_Activity', 'Internship_Type', 'Internship_Company']:
        input_df[col] = le.fit_transform(input_df[col])

    input_scaled = scaler.transform(input_df)
    result = model.predict(input_scaled)[0]

    if result == 1:
        st.success("The student is likely to be Placed!")
    else:
        st.error("The student is unlikely to be Placed.")
