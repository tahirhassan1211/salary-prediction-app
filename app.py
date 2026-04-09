import streamlit as st
import joblib
import pandas as pd
import requests

# Load model
model = joblib.load("salary_model.joblib")

st.title("Salary Prediction App", text_alignment="center")
job_titles = ['AI Engineer','Data Analyst','Frontend Developer', 'Business Analyst','Product Manager','Backend Developer','Machine Learning Engineer','DevOps Engineer','Software Engineer','Cybersecurity Analyst','Data Scientist','Cloud Engineer']
locations=['India','Australia','Singapore','Canada','Sweden','USA','Netherlands','Remote','Germany','UK']
industries=['Healthcare','Telecom','Media','Retail','Manufacturing','Education','Finance','Technology','Consulting','Government']

def reset_form():
    st.session_state["job_title"] = job_titles 
    st.session_state["experience"] = 0
    st.session_state["education"] = "High School"
    st.session_state["skills"] = 0 
    st.session_state["industry"] = industries
    st.session_state["company_size"] = "Small"
    st.session_state["location"] = locations
    st.session_state["remote"] = "Remote"
    st.session_state["certifications"] = 0


job_title = st.selectbox("Job Title", job_titles, key="job_title")
experience = st.slider("Experience", 0, 20, key="experience")
education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"], key="education")
skills = st.slider("Skills Count", 0, 20, key="skills")
industry = st.selectbox("Industry", industries, key="industry")
company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"], key="company_size")
location = st.selectbox("Location",locations, key="location")
remote = st.selectbox("Remote", ["Yes", "No"], key="remote")
certifications = st.slider("Certifications", 0, 10, key="certifications")


if st.button("Predict"):
    input_df = pd.DataFrame([{
        "job_title": job_title,
        "experience_years": experience,
        "education_level": education,
        "skills_count": skills,
        "industry": industry,
        "company_size": company_size,
        "location": location,
        "remote_work": remote,
        "certifications": certifications
    }])
    
    predicted_salary = model.predict(input_df)[0]
    st.success(f"Predicted Salary: ${predicted_salary:,.2f}")

st.button("Clear All Options", on_click=reset_form)