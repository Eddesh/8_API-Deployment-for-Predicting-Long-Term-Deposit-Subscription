from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import logging

# Davin Edbert Santoso Halim
# 2602067086

# Mengatur logging untuk mencatat informasi penting selama eksekusi API.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load model dan encoders
with open('dt_model.pkl', 'rb') as file:
    dt_best = pickle.load(file)
with open('OHE_encoder.pkl', 'rb') as file:
    OHE_encoder = pickle.load(file)
with open('enc_contact.pkl', 'rb') as file:
    enc_contact = pickle.load(file)
with open('enc_month.pkl', 'rb') as file:
    enc_month = pickle.load(file)
with open('enc_day.pkl', 'rb') as file:
    enc_day = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
logger.info("Models and encoders loaded successfully.")

# Mendefinisikan Model Input Data
class InputData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    duration: float
    campaign: int
    pdays: int
    previous: int
    poutcome: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Model API"}

def preprocess_data(input_df):
    OHE_col = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'poutcome']
    num_col = ['age', 'duration', 'campaign', 'pdays', 'previous']
    encoded_data = OHE_encoder.transform(input_df[OHE_col])
    encoded_data_df = pd.DataFrame(encoded_data, columns=OHE_encoder.get_feature_names_out(OHE_col))
    input_df = input_df.reset_index(drop=True)
    input_df = pd.concat([input_df.drop(OHE_col, axis=1), encoded_data_df], axis=1)

    input_df = input_df.replace(enc_contact)
    input_df = input_df.replace(enc_month)
    input_df = input_df.replace(enc_day)
    
    input_df[num_col] = scaler.transform(input_df[num_col])
    
    return input_df

@app.post('/predict')
def predict(data: InputData):
    input_df = pd.DataFrame([data.dict()])
    logger.info(f"Input data: {input_df}")
    
    processed_data = preprocess_data(input_df)
    logger.info(f"Processed data: {processed_data}")
    
    prediction = dt_best.predict(processed_data)
    prediction_label = 'yes' if prediction[0] == 1 else 'no'
    
    prediction_meaning = (
        "Individu tersebut kemungkinan besar akan berlangganan deposito berjangka."
        if prediction_label == 'yes'
        else "Individu tersebut kemungkinan besar tidak akan berlangganan deposito berjangka."
    )
    
    logger.info(f"Prediction: {prediction_label}")
    return {
        'prediction': prediction_label,
        'meaning': prediction_meaning
    }
