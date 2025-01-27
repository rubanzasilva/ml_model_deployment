import gradio as gr
import requests
import pandas as pd
import tempfile
import os

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
            #"https://mental-health-classifier-v2-63072676.mt-guc1.bentoml.ai/predict_csv",
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

# Create the Gradio interface
demo = gr.Interface(
    fn=predict_csv_file,
    inputs=[
        gr.File(
            label="Upload CSV File",
            file_types=[".csv"]
        )
    ],
    outputs=[
        gr.File(label="Download Complete Predictions CSV"),
        gr.Textbox(label="Predictions", lines=10)
    ],
    title="Mental Health Depression Risk Prediction - Batch Processing",
    description="""Upload a CSV file containing multiple records for batch prediction. 
    The CSV should contain the same columns as required in the individual prediction form:
    Name, Gender, City, Working Professional or Student, Profession, Sleep Duration, 
    Dietary Habits, Degree, Have you ever had suicidal thoughts?, Family History of Mental Illness,
    Age, Academic Pressure, Work Pressure, CGPA, Study Satisfaction, Job Satisfaction, 
    Work/Study Hours, Financial Stress"""
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)