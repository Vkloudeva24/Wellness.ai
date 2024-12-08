from fastapi import FastAPI, Request
from pydantic import BaseModel
import streamlit.web.cli as stcli
import threading
import uvicorn

# Run Streamlit in a separate thread
def run_streamlit():
    stcli.main(["app.py"])

threading.Thread(target=run_streamlit, daemon=True).start()

app = FastAPI()

class InputData(BaseModel):
    Gender: str
    Occupation: str
    BMI_Category: str
    Sleep_Disorder: str
    Stress_Level: int
    Quality_of_Sleep: int
    Daily_Steps: int

@app.post("/predict")
def predict(data: InputData):
    result = {
        "health_status": "Healthy",
        "recommendation": "Maintain a healthy lifestyle!"
    }
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
