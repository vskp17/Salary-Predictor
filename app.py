# app.py

import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load trained model and feature names
with open("salary_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("model_features.pkl", "rb") as file:
    feature_names = pickle.load(file)

st.set_page_config(page_title="Salary Prediction App", layout="centered")
st.title("💼 Salary Prediction App")
st.markdown("Fill in the applicant details below to estimate their expected salary (CTC).")

# Input fields
experience = st.number_input("Total Experience (in years)", min_value=0, max_value=50, step=1)
field_experience = st.number_input("Experience in Applied Field (in years)", min_value=0, max_value=50, step=1)
education = st.selectbox("Education Level", ["Grad", "PG", "Doctorate"])
certifications = st.slider("Number of Certifications", 0, 10, 0)
publications = st.slider("Number of Publications", 0, 10, 0)
no_of_companies = st.slider("Number of Companies Worked", 1, 10, 1)
international_degree = st.radio("Has International Degree?", ["No", "Yes"])
current_ctc = st.number_input("Current CTC (in ₹)", min_value=0)

# Create empty input array with 0s
input_data = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

# Fill in values
input_data["Total_Experience"] = experience
input_data["Total_Experience_in_field_applied"] = field_experience
input_data["Current_CTC"] = current_ctc
input_data["No_Of_Companies_worked"] = no_of_companies
input_data["Number_of_Publications"] = publications
input_data["Certifications"] = certifications
input_data["International_degree_any"] = 1 if international_degree == "Yes" else 0

# Handle education
if f"Education_PG" in input_data.columns:
    input_data["Education_PG"] = 1 if education == "PG" else 0
if f"Education_Doctorate" in input_data.columns:
    input_data["Education_Doctorate"] = 1 if education == "Doctorate" else 0

# Predict
if st.button("💰 Predict Salary"):
    try:
        predicted_salary = model.predict(input_data)[0]
        st.success(f"✅ Predicted Expected CTC: ₹{predicted_salary:,.2f}")
    except Exception as e:
        st.error(f"❌ Error in prediction: {e}")
