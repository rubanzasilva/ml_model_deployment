import gradio as gr
import requests
import pandas as pd
import json

def predict_depression(name, gender, city, occupation_type, profession, sleep_duration, 
                      dietary_habits, degree, suicidal_thoughts, family_history,
                      age, academic_pressure, work_pressure, cgpa, 
                      study_satisfaction, job_satisfaction, work_hours, financial_stress):
    
    # Create the input data structure
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
    
    # Make prediction request to BentoML service
    try:
        response = requests.post(
            #"http://localhost:3000/predict",
            "https://mental-health-classifier-vl0e-63072676.mt-guc1.bentoml.ai/predict",
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

# Create the Gradio interface
demo = gr.Interface(
    fn=predict_depression,
    inputs=[
        gr.Textbox(label="Name"),
        gr.Dropdown(
            choices=["Male", "Female", "Other"],
            label="Gender"
        ),
        gr.Textbox(label="City"),
        gr.Dropdown(
            choices=["Working Professional", "Student"],
            label="Occupation Type"
        ),
        gr.Textbox(label="Profession"),
        gr.Dropdown(
            choices=["Less than 6 hours", "6-8 hours", "More than 8 hours"],
            label="Sleep Duration"
        ),
        gr.Dropdown(
            choices=["Healthy", "Moderate", "Unhealthy"],
            label="Dietary Habits"
        ),
        gr.Textbox(label="Degree"),
        gr.Radio(
            choices=["Yes", "No"],
            label="Have you ever had suicidal thoughts?"
        ),
        gr.Radio(
            choices=["Yes", "No"],
            label="Family History of Mental Illness"
        ),
        gr.Number(label="Age"),
        gr.Slider(minimum=0, maximum=10, step=1, label="Academic Pressure"),
        gr.Slider(minimum=0, maximum=10, step=1, label="Work Pressure"),
        gr.Number(label="CGPA", minimum=0, maximum=10),
        gr.Slider(minimum=0, maximum=10, step=1, label="Study Satisfaction"),
        gr.Slider(minimum=0, maximum=10, step=1, label="Job Satisfaction"),
        gr.Number(label="Work/Study Hours"),
        gr.Slider(minimum=0, maximum=10, step=1, label="Financial Stress")
    ],
    outputs=gr.Textbox(label="Prediction Result"),
    title="Mental Health Depression Risk Prediction",
    description="Enter the required information to predict depression risk."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)