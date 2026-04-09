from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("salary_model.joblib")

class SalaryInput(BaseModel):
    job_title: str = Field(..., min_length=2)
    experience_years: float = Field(..., ge=0, le=50)
    education_level: str
    skills_count: int = Field(..., ge=0, le=50)
    industry: str
    company_size: str
    location: str
    remote_work: str
    certifications: int = Field(..., ge=0, le=20)

@app.get("/")
def home():
    return {"message": "API running"}

@app.post("/predict")
def predict(data: SalaryInput):
    df = pd.DataFrame([data.dict()])
    pred = model.predict(df)[0]
    return {"predicted_salary": float(pred)}