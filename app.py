from flask import Flask, render_template, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import LSTM

import sys

def prepare_data(data, seq_len=20):
    scaler = MinMaxScaler()
    df = data[['Close']]
    scaled_data = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(scaled_data) - seq_len):
        X.append(scaled_data[i:i + seq_len])
        y.append(scaled_data[i + seq_len, 0])
    
    return X, y, scaler

def train_model(X_train, X_val, y_train, y_val):
    input_dim = 1
    hidden_dim = 64
    num_layers = 2
    output_dim = 1
    batch_size = 64
    
    model = LSTM(input_dim, hidden_dim, num_layers, output_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    num_epochs = 10
    
    dataset_train = TensorDataset(X_train, y_train)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    dataset_val = TensorDataset(X_val, y_val)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training Loop
        for i, (inputs, targets) in enumerate(dataloader_train):
            outputs = model(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch {i + 1}, Loss: {loss.item():.4f}')

        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in dataloader_val:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(dataloader_val)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # save the best model due to validation (least val loss)
            torch.save(model.state_dict(), "lstm_model.pth")

    model.load_state_dict(torch.load("lstm_model.pth"))
    return model

def fetch_stock_data(ticker, period="5y", interval="1d"):
    stock = yf.Ticker(ticker)
    historical_data = stock.history(period=period, interval=interval)
    return historical_data


app = Flask(__name__)
CORS(app)
api = Api(app)

class StockDataResource(Resource):
    def get(self, ticker):
        data = fetch_stock_data(ticker)
        data_list = data.to_dict(orient="records")
        # Convert index if it's Datetime type
        # historical_data['Date'] = historical_data.index.strftime('%Y-%m-%d') 
        return jsonify({'data': data_list})

class PredictionResource(Resource):
    def get(self, ticker):
        try:
            data = fetch_stock_data(ticker)
    
            X, y, scaler = prepare_data(data)
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            X_train = np.array(X_train)
            X_val = np.array(X_val)
            y_train = np.array(y_train)
            y_val = np.array(y_val)
            
            X_train = X_train.astype('float32')
            X_val = X_val.astype('float32')
            y_train = y_train.astype('float32')
            y_val = y_val.astype('float32')
            
            y_train = y_train.reshape(-1, 1)  
            y_val = y_val.reshape(-1, 1)

            # Train model
            model = train_model(torch.from_numpy(X_train), torch.from_numpy(X_val), 
                                torch.from_numpy(y_train), torch.from_numpy(y_val))

            # Inference
            model.eval()
            with torch.no_grad():
                X = np.array(X)
                X = X.astype('float32')
                input_tensor = torch.tensor(X[-1:])
                # print(type(X))
                # print(type(X[-1:]))
                # print(X[-1:].shape)
                output = model(input_tensor)
                raw_prediction = output.item()

            predicted_price = scaler.inverse_transform([[raw_prediction]])[0][0]    

            
            return jsonify({'prediction': predicted_price})
        except ValueError:  # Example: catching invalid tickers
            return jsonify({'error': 'Invalid ticker symbol'}), 400 
        except Exception as e:  # Catch-all for unexpected errors
            return jsonify({'error': 'Internal server error'}), 500

api.add_resource(StockDataResource, "/stock-data/<string:ticker>")
api.add_resource(PredictionResource, "/predict/<string:ticker>")

'''
@app.route("/")
def index():
    return render_template("index.html")
'''

'''
@app.route("/forecast", methods=["POST"])

def get_forecast():
    ticker = request.form['ticker']
    data = fetch_stock_data(ticker)
    
    X, y, scaler = prepare_data(data)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = np.array(X_train)
    X_val = np.array(X_val)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    y_train = y_train.astype('float32')
    y_val = y_val.astype('float32')
    
    y_train = y_train.reshape(-1, 1)  
    y_val = y_val.reshape(-1, 1)

    # Train model
    model = train_model(torch.from_numpy(X_train), torch.from_numpy(X_val), 
                        torch.from_numpy(y_train), torch.from_numpy(y_val))

    # Inference
    model.eval()
    with torch.no_grad():
        X = np.array(X)
        X = X.astype('float32')
        input_tensor = torch.tensor(X[-1:])
        # print(type(X))
        # print(type(X[-1:]))
        # print(X[-1:].shape)
        output = model(input_tensor)
        raw_prediction = output.item()

    predicted_price = scaler.inverse_transform([[raw_prediction]])[0][0]    
    return render_template("index.html", forecast=predicted_price)
'''

if __name__ == "__main__":
    app.run(debug=True)