# 💼 Salary Prediction Using Machine Learning

A machine learning project that predicts the expected salary (CTC) for a job applicant based on their experience, education, certifications, and other features — built as part of an internship capstone project.

---

## 📌 Problem Statement

To ensure fairness in the recruitment process, Company X wants to automate salary estimation based on historical applicant data. This eliminates bias and ensures consistent salary offers for similar profiles.

---

## 🎯 Objective

Develop a robust machine learning model to:
- Predict **Expected CTC** for new applicants
- Minimize human judgment in the salary offer process
- Deploy the model using a **Streamlit web app**

---

## 🧠 Project Workflow

### ✅ Phase 1: Data Cleaning & EDA
- Removed rows with missing values
- Explored key trends using visualizations
- Saved correlation heatmap for reporting

### ✅ Phase 2: Feature Engineering & Model Training
- One-hot encoded categorical features
- Trained and evaluated:
  - Linear Regression
  - Random Forest
  - XGBoost
- Chose **XGBoost** as the final model (R² ≈ 0.9998)
- Saved model and features using `pickle`

### ✅ Phase 3: Streamlit App
- Created a user-friendly form interface
- Inputs like experience, education, certifications, etc.
- Displays predicted salary from model in real-time

---

## 📊 Model Performance

| Model             | R² Score | RMSE (₹)     |
|-------------------|----------|--------------|
| Linear Regression | 0.9967   | ₹66,117.66   |
| Random Forest     | 0.9996   | ₹23,568.88   |
| XGBoost           | **0.9998** | **₹16,243.41** |

✅ **XGBoost** was selected and used in the final deployed app.

---

## 🗃️ Dataset Info

- Original file: `expected_ctc.csv`
- Cleaned version: `cleaned_expected_ctc.csv`
- Features include:
  - Total Experience
  - Experience in Field
  - Education Level
  - Certifications
  - Number of Companies Worked
  - Publications
  - International Degree (Yes/No)
  - Current CTC
  - Expected CTC (Target)

---

## 🚀 How to Run This Project

### 🔧 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/salary-predictor.git
cd salary-predictor
