import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and feature names
data = joblib.load("model_rf.pkl")
model = data['model']
feature_names = data['feature_names']

# Set title
st.title("Data Science, AI & ML Salary Prediction App 💼📊")

# Input options based on your training features
work_year = st.selectbox("Work Year", [2020, 2021, 2022, 2023, 2024, 2025])
remote_ratio = st.selectbox("Remote Ratio (%)", [0, 50, 100])

employment_type = st.selectbox("Employment Type", ['Freelance', 'Full-time', 'Part-time'])
experience_level = st.selectbox("Experience Level", ['Executive', 'Intermediate', 'Senior-level'])
company_size = st.selectbox("Company Size", ['Small(1-50)', 'Medium(51-500)'])
 
job_title = st.selectbox("Job Title", ['Data Engineer', 'Data Scientist', 'Engineer', 'Software Engineer', 'Other'])
company_location = st.selectbox("Company Location", ['Canada', 'United Kingdom', 'Netherlands', 'United States', 'Other'])
employee_residence = st.selectbox("Employee Residence", ['Canada', 'United Kingdom', 'United States', 'Other'])


input_dict = {
    'work_year': work_year,
    'remote_ratio': remote_ratio,
    'employment_type_' + employment_type: 1,
    'experience_level_' + experience_level: 1,
    'company_size_' + company_size: 1,
    'job_title_grouped_' + job_title: 1,
    'company_location_grouped_' + company_location: 1,
    'employee_residence_grouped_' + employee_residence: 1
}

# Convert to DataFrame
df_input = pd.DataFrame([input_dict])

# Reindex to match the training features, fill missing columns with 0
df_input = df_input.reindex(columns=feature_names, fill_value=0)

# Predict on click
if st.button("Predict Salary (USD)"):
    prediction = model.predict(df_input)[0]
    st.success(f"Estimated Salary: ${prediction:,.2f}")


st.markdown(
    f"""
    <style>
    html, body, .main, .stApp {{
        background-image: url("https://i.imgur.com/guUNHTg.jpeg");
        
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}






    /* Improve font and layout */
    html, body {{
        font-family: 'Merriweather';
        padding: 1rem;
    }}

    h1, h2, h3 {{
        color: #ffffff;
        text-shadow: 1px 1px 4px #000000;
    }}

    label, .st-emotion-cache-1c7y2kd, .st-emotion-cache-1v0mbdj {{
        color: #222831 !important;
        font-weight: bold !important;
        text-shadow: 1px 1px 2px rgba(255,255,255,0.3);
    }}

    /* Improve buttons */
    .stButton>button {{
        background-color: #0066cc;
        color: white;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        font-weight: bold;
        border: none;
    }}

    .stButton>button:hover {{
        background-color: #004a99;
    }}

div[data-testid="stAlert"] {{
        background-color: #2ecc71 !important;  /* Bright green */
        color: white !important;
        font-weight: bold;
        border-radius: 12px;
        padding: 1.5rem;
        border: 2px solid #27ae60;
        box-shadow: 3px 4px 12px rgba(0, 0, 0, 0.25);
        font-size: 2.0rem;
        text-align: center;
    }}

    </style>
    """,
    unsafe_allow_html=True
)
