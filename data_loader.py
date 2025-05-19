import yfinance as yf
import pandas as pd
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from util import create_sequence

torch.manual_seed(0)

class FinDataLoader:
    @staticmethod
    def load_fin_data(window_size: int = 10) -> tuple:
        # Load the data for Apple Inc. (AAPL) from Yahoo Finance
        df = yf.download('AAPL', start='2020-01-01', end='2024-01-01')
        df = df[['Close']]

        df['Return'] = df['Close'].pct_change().fillna(0)

        # Normalize
        scaler = StandardScaler()
        df['Return'] = scaler.fit_transform(df['Return'].values.reshape(-1, 1))

        returns = df['Return'].values
        X, y = create_sequence(returns, window_size)

        train_size = int(len(X) * 0.7)
        val_size = int(len(X) * 0.15)
        test_size = len(X) - train_size - val_size
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
        X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]
        return (X_train, X_val, X_test), (y_train, y_val, y_test)
    
def create_dataloaders(model_config):
    # Load the data
    (X_train, X_val, X_test), (y_train, y_val, y_test) = FinDataLoader.load_fin_data()

    # Create DataLoader objects
    train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=model_config['batch_size'], shuffle=True)
    val_loader = DataLoader(StockDataset(X_val, y_val), batch_size=model_config['batch_size'])
    test_loader = DataLoader(StockDataset(X_test, y_test), batch_size=model_config['batch_size'])

    print("Data preprocessed and loaded.")
    return train_loader, val_loader, test_loader

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1) 
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]