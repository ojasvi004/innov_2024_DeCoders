import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
import tensorflow as tf
import yfinance as yf
import pickle
import os

def data_prep(df, lookback, future, Scale):
    date_train = pd.to_datetime(df['Date'])
    df_train = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]
    df_train = df_train.astype(float)
    
    df_train_scaled = Scale.fit_transform(df_train)

    X, y = [], []
    for i in range(lookback, len(df_train_scaled) - future + 1):
        X.append(df_train_scaled[i - lookback:i, 0:df_train.shape[1]])
        y.append(df_train_scaled[i + future - 1:i + future, 0])
        
    return np.array(X), np.array(y), df_train, date_train

def Lstm_model2(X, y):
    model = Sequential()
    
    model.add(LSTM(20, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(LSTM(15))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    
    adam = optimizers.Adam(0.001)
    model.compile(loss='mean_squared_error', optimizer=adam)
    
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
    model.fit(X, y, validation_split=0.2, epochs=100, batch_size=64, verbose=1, callbacks=[es])
    return model

def predict_open(model, date_train, Lstm_x, df_train, future, Scale):
    forecasting_dates = pd.date_range(list(date_train)[-1], periods=future, freq='1d').tolist()
    predicted = model.predict(Lstm_x[-future:])
    predicted1 = np.repeat(predicted, df_train.shape[1], axis=-1)
    predicted_descaled = Scale.inverse_transform(predicted1)[:,0]
    return predicted_descaled, forecasting_dates

def output_prep(forecasting_dates, predicted_descaled):
    dates = [] 
    for i in forecasting_dates:
        dates.append(i.date())
    df_final = pd.DataFrame(columns=['Date', 'Open'])
    df_final['Date'] = pd.to_datetime(dates)
    df_final['Open'] = predicted_descaled
    return df_final

def main():
    # Create a directory named 'pickle' if it doesn't exist
    if not os.path.exists('pickle'):
        os.makedirs('pickle')

    # List of stock symbols
    stock_symbols = ["^NSEBANK", "AXISBANK.NS", "SBIN.NS", "RBLBANK.NS", "PNB.NS", "KOTAKBANK.NS", 
          "INDUSINDBK.NS", "IDFCFIRSTB.NS", "ICICIBANK.NS", "BANDHANBNK.NS", "HDFC.NS", 
          "FEDERALBNK.NS", "AUBANK.NS", "RELIANCE.NS", "TCS.NS", "GOOGL", "AMZN", "INFY.NS",
          "ADANIENT.NS", "HINDUNILVR.NS", "WIPRO.NS", "BHARTIARTL.NS", "LT.NS", "ITC.NS"]
    
    for symbol in stock_symbols:
        try:
            # Data collection
            df = yf.Ticker(symbol).history(period='10y').reset_index()

            # Data preparation
            Scale = StandardScaler()
            lookback = 30
            future = 1
            Lstm_x, Lstm_y, df_train, date_train = data_prep(df, lookback, future, Scale)

            # Splitting data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(Lstm_x, Lstm_y, test_size=0.2, random_state=42)

            # Training LSTM model
            model = Lstm_model2(X_train, y_train)

            # Save the trained model using pickle
            with open(f'pickle/lstm_model_{symbol}.pkl', 'wb') as f:
                pickle.dump(model, f)

            # Prediction
            predicted_descaled, forecasting_dates = predict_open(model, date_train, X_test, df_train, future, Scale)
            results = output_prep(forecasting_dates, predicted_descaled)

            
            print(f"Prediction for {symbol} completed successfully.")

        except Exception as e:
            print(f"Error processing data for symbol {symbol}: {str(e)}")
            
if __name__ == "__main__":
    main()
