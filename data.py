import yfinance as yf
import pandas as pd
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


# Load the data for Apple Inc. (AAPL) from Yahoo Finance
df = yf.download('AAPL', start='2020-01-01', end='2024-01-01')
df = df[['Close']]

df['Return'] = df['Close'].pct_change().fillna(0)

# Normalize
scaler = StandardScaler()
df['Return'] = scaler.fit_transform(df['Return'].values.reshape(-1, 1))

def create_sequence(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

window_size = 10
returns = df['Return'].values
X, y = create_sequence(returns, window_size)

train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)
test_size = len(X) - train_size - val_size
X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1) 
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(StockDataset(X_val, y_val), batch_size=32)
test_loader = DataLoader(StockDataset(X_test, y_test), batch_size=32)

print("Data preprocessed and loaded.")