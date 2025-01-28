import gradio as gr
import requests
import pandas as pd
import tempfile
import os

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

def predict_csv_file(file_obj):
    try:
        # Check if file was uploaded
        if file_obj is None:
            return None, "Please upload a CSV file"
            
        # Read the uploaded CSV file directly using pandas
        df = pd.read_csv(file_obj.name)
        
        # Create multipart form-data
        files = {
            'csv': ('input.csv', open(file_obj.name, 'rb'), 'text/csv')
        }
        
        # Make prediction request to BentoML service
        response = requests.post(
            "http://localhost:3000/predict_csv",
            files=files
        )
        
        if response.status_code == 200:
            predictions = response.json()
            
            # Create a DataFrame with original data and predictions
            df['Depression Risk'] = ['Yes' if pred == 1 else 'No' for pred in predictions]
            
            # Save results to a CSV file
            results_path = "predictions_results.csv"
            df.to_csv(results_path, index=False)
            
            # Create a formatted prediction list
            prediction_list = []
            for index, row in df.iterrows():
                prediction_text = f"Record {index + 1}: {row['Name']} - Depression Risk: {row['Depression Risk']}"
                prediction_list.append(prediction_text)
            
            # Join predictions into a single string with line breaks
            predictions_display = "\n".join(prediction_list)
            
            return results_path, predictions_display
        else:
            return None, f"Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return None, f"Error processing file: {str(e)}"

# Create the individual prediction interface
individual_inputs = [
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
]

individual_output = gr.Textbox(label="Prediction Result")

# Create the batch processing interface
batch_inputs = [
    gr.File(
        label="Upload CSV File",
        file_types=[".csv"]
    )
]

batch_outputs = [
    gr.File(label="Download Complete Predictions CSV"),
    gr.Textbox(label="Predictions", lines=10)
]

# Create the combined interface with tabs
demo = gr.Blocks(title="Mental Health Depression Risk Prediction System")

with demo:
    gr.Markdown("# Mental Health Depression Risk Prediction System")
    
    with gr.Tabs():
        with gr.TabItem("Individual Prediction"):
            gr.Markdown("Enter individual information to predict depression risk.")
            with gr.Column():
                individual_interface = gr.Interface(
                    fn=predict_depression,
                    inputs=individual_inputs,
                    outputs=individual_output,
                    title=None
                )
        
        with gr.TabItem("Batch Processing"):
            gr.Markdown("""Upload a CSV file containing multiple records for batch prediction. 
            The CSV should contain the following columns:
            Name, Gender, City, Working Professional or Student, Profession, Sleep Duration, 
            Dietary Habits, Degree, Have you ever had suicidal thoughts?, Family History of Mental Illness,
            Age, Academic Pressure, Work Pressure, CGPA, Study Satisfaction, Job Satisfaction, 
            Work/Study Hours, Financial Stress""")
            with gr.Column():
                batch_interface = gr.Interface(
                    fn=predict_csv_file,
                    inputs=batch_inputs,
                    outputs=batch_outputs,
                    title=None
                )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)