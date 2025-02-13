from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
import pandas as pd
import uvicorn
import io
import plotly.graph_objects as go
import json
import matplotlib.pyplot as plt

app = FastAPI(title="Rotating Machine Fault Classifier", description="Upload a file to classify faults.")

# Load the trained model
MODEL_PATH = "best_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Set up Jinja2 templates directory
templates = Jinja2Templates(directory="templates")

# Serve static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")


# Function to process CSV or NPY files
def process_uploaded_file(file: UploadFile):
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file.file, header=None)
            input_array = df.to_numpy()
        elif file.filename.endswith(".npy"):
            input_array = np.load(io.BytesIO(file.file.read()))
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or NPY file.")

        if input_array.shape != (4, 1013):
            raise ValueError(f"Expected shape (4, 1013), but got {input_array.shape}")

        # ✅ Ensure Correct Shape: (4, 1013) should be (1013, 4)
        input_array = input_array.T  # Transpose to get (1013, 4)

        return input_array  # Now shape is (1013, 4)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Function to generate interactive Plotly plots
def generate_plotly_plots(raw_data):
    """
    Generates Plotly plots for the four vibration channels.
    """
    time_series = list(range(raw_data.shape[0]))  # Time steps from 0 to 1012
    plots = []

    for i in range(4):
        fig = go.Figure()

        # ✅ Ensure Y-axis data is explicitly converted to Python list
        fig.add_trace(go.Scatter(
            x=time_series, 
            y=raw_data[:, i].tolist(),  # Convert NumPy array to list
            mode="lines", 
            name=f"Channel {i+1}"
        ))

        fig.update_layout(
            title=f"Vibration Data - Channel {i+1}",
            xaxis_title="Time Steps",
            yaxis_title="Amplitude"
        )

        # Convert Plotly figure to JSON for frontend
        plots.append(fig.to_json())

    return plots



@app.post("/predict-file/")
async def predict_file(file: UploadFile = File(...)):
    try:
        raw_data = process_uploaded_file(file)  # ✅ Extract correctly shaped data

        # Debugging: Print the first few values of each channel
        print("DEBUG: First 5 values of each channel:")
        for i in range(4):
            print(f"Channel {i+1}: {raw_data[:5, i]}")

        # Expand dimensions before prediction
        input_array = np.expand_dims(raw_data, axis=0)  # Shape: (1, 1013, 4)

        # Make prediction
        predictions = model.predict(input_array)
        predicted_class = int(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions))

        # Generate interactive Plotly plots
        plotly_plots = generate_plotly_plots(raw_data)

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "plots": plotly_plots
        }

    except Exception as e:
        return {"error": str(e)}



@app.get("/")
async def upload_page(request: Request):
    """Render the file upload page."""
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)





