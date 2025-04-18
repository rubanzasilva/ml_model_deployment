import streamlit as st
import pandas as pd
import requests
import json
import numpy as np

def predict_single(name, gender, city, occupation_type, profession, sleep_duration, 
                  dietary_habits, degree, suicidal_thoughts, family_history,
                  age, academic_pressure, work_pressure, cgpa, 
                  study_satisfaction, job_satisfaction, work_hours, financial_stress):
    """Make single prediction using the BentoML API"""
    data = {
        "data": [{
            "Name": name,
            "Gender": gender,
            "City": city,
            "Working Professional or Student": occupation_type,
            "Profession": profession,
            "Sleep Duration": sleep_duration,
            "Dietary Habits": dietary_habits,
            "Degree": degree,
            "Have you ever had suicidal thoughts ?": suicidal_thoughts,
            "Family History of Mental Illness": family_history,
            "Age": age,
            "Academic Pressure": academic_pressure,
            "Work Pressure": work_pressure,
            "CGPA": cgpa,
            "Study Satisfaction": study_satisfaction,
            "Job Satisfaction": job_satisfaction,
            "Work/Study Hours": work_hours,
            "Financial Stress": financial_stress
        }]
    }
    
    try:
        response = requests.post(
            "http://localhost:3000/predict",
            json=data,
            headers={"content-type": "application/json"}
        )
        
        if response.status_code == 200:
            prediction = response.json()
            return "Depression Risk: Yes" if prediction[0] == 1 else "Depression Risk: No"
        else:
            return f"Error: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Error connecting to service: {str(e)}"

def clean_dataframe(df):
    """Clean dataframe to ensure JSON compatibility"""
    # Replace infinite values with None
    df = df.replace([np.inf, -np.inf], None)
    
    # Replace NaN values with None (which will be converted to null in JSON)
    df = df.replace({np.nan: None})
    
    # Ensure numeric columns are finite
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def predict_batch(file):
    """Process batch predictions from CSV file"""
    try:
        # Read CSV file
        df = pd.read_csv(file)
        
        # Clean the dataframe
        df = clean_dataframe(df)
        
        # Convert to dict with explicit handling of null values
        records = df.to_dict(orient='records')
        
        # Make prediction request to BentoML service
        response = requests.post(
            "http://localhost:3000/predict_csv",
            json={"data": records},
            headers={"content-type": "application/json"}
        )
        
        if response.status_code == 200:
            predictions = response.json()
            
            # Add predictions to dataframe
            df['Depression Risk'] = ['Yes' if pred == 1 else 'No' for pred in predictions]
            
            return df
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.error("Please ensure all numeric values are valid numbers")
        return None

def main():
    st.title("Mental Health Depression Risk Prediction System")
    
    # Create tabs for single prediction and batch prediction
    tab1, tab2 = st.tabs(["Individual Prediction", "Batch Processing"])
    
    # Single Prediction Tab
    with tab1:
        st.header("Individual Prediction")
        
        # Create input fields matching Gradio implementation
        name = st.text_input("Name")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        city = st.text_input("City")
        occupation_type = st.selectbox("Occupation Type", ["Working Professional", "Student"])
        profession = st.text_input("Profession")
        sleep_duration = st.selectbox("Sleep Duration", ["Less than 6 hours", "6-8 hours", "More than 8 hours"])
        dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
        degree = st.text_input("Degree")
        suicidal_thoughts = st.radio("Have you ever had suicidal thoughts?", ["Yes", "No"])
        family_history = st.radio("Family History of Mental Illness", ["Yes", "No"])
        age = st.number_input("Age", min_value=0, max_value=120)
        academic_pressure = st.slider("Academic Pressure", 0, 10)
        work_pressure = st.slider("Work Pressure", 0, 10)
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0)
        study_satisfaction = st.slider("Study Satisfaction", 0, 10)
        job_satisfaction = st.slider("Job Satisfaction", 0, 10)
        work_hours = st.number_input("Work/Study Hours", min_value=0)
        financial_stress = st.slider("Financial Stress", 0, 10)
        
        if st.button("Predict"):
            result = predict_single(
                name, gender, city, occupation_type, profession, sleep_duration,
                dietary_habits, degree, suicidal_thoughts, family_history,
                age, academic_pressure, work_pressure, cgpa,
                study_satisfaction, job_satisfaction, work_hours, financial_stress
            )
            st.write(result)
    
    # Batch Prediction Tab
    with tab2:
        st.header("Batch Processing")
        st.markdown("""
        Upload a CSV file containing multiple records for batch prediction. 
        The CSV should contain the following columns:
        - Name
        - Gender
        - City
        - Working Professional or Student
        - Profession
        - Sleep Duration
        - Dietary Habits
        - Degree
        - Have you ever had suicidal thoughts?
        - Family History of Mental Illness
        - Age
        - Academic Pressure
        - Work Pressure
        - CGPA
        - Study Satisfaction
        - Job Satisfaction
        - Work/Study Hours
        - Financial Stress
        """)
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                preview_df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.write(preview_df.head())
                
                if st.button("Process Batch"):
                    uploaded_file.seek(0)  # Reset file pointer
                    results_df = predict_batch(uploaded_file)
                    
                    if results_df is not None:
                        st.success("Predictions completed!")
                        st.write("Results preview:")
                        st.write(results_df)
                        
                        # Download button for results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results CSV",
                            data=csv,
                            file_name="predictions_results.csv",
                            mime="text/csv"
                        )
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")

if __name__ == "__main__":
    main()