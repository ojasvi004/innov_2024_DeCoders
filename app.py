from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
import pandas as pd
from main import data_prep, predict_open

app = FastAPI()

# Dictionary to store LSTM models
models = {}

# Load LSTM models
symbols = ["^NSEBANK", "AXISBANK.NS", "SBIN.NS", "RBLBANK.NS", "PNB.NS", "KOTAKBANK.NS", 
          "INDUSINDBK.NS", "IDFCFIRSTB.NS", "ICICIBANK.NS", "BANDHANBNK.NS", 
          "FEDERALBNK.NS", "AUBANK.NS", "RELIANCE.NS", "TCS.NS", "GOOGL", "AMZN", "INFY.NS",
          "ADANIENT.NS", "HINDUNILVR.NS", "WIPRO.NS", "BHARTIARTL.NS", "LT.NS", "ITC.NS"]
for symbol in symbols:
    with open(f'pickle/lstm_model_{symbol}.pkl', 'rb') as f:
        models[symbol] = pickle.load(f)

class PredictionRequest:
    def __init__(self, symbol: str, date: str):
        self.symbol = symbol
        self.date = date

class PredictionResponse:
    def __init__(self, predicted_open: list):
        self.predicted_open = predicted_open

@app.post('/predict')
async def predict(request):
        data = await request.json()
        symbol = data['symbol']
        date_to_predict = pd.to_datetime(data['date'])

        # Load the LSTM model
        model = models[symbol]

        # Retrieve historical data up to the prediction date
        df = yf.Ticker(symbol).history(start="2010-01-01", end=date_to_predict.strftime('%Y-%m-%d'))

        # Data preparation
        Scale = StandardScaler()
        lookback = 30
        future = 1
        Lstm_x, _, df_train, date_train = data_prep(df, lookback, future, Scale)

        # Predict the open price
        predicted_descaled, _ = predict_open(model, date_train, Lstm_x, df_train, future, Scale)

        return {'predicted_open': predicted_descaled.tolist()}
    