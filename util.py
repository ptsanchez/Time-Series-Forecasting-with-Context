import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

def plot_losses(train_losses, val_losses, fname):
    if not os.path.isdir('plots'):
        os.mkdir('plots')
    train_losses = train_losses[1:]
    val_losses = val_losses[1:]

    epochs = np.arange(1, len(train_losses) + 1)
    plt.close('all')
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')

    plt.savefig(fname + ".png")

def plot_predictions(predictions, targets, fname):
    if not os.path.isdir('plots'):
        os.mkdir('plots')
    plt.close('all')

    plt.plot(predictions, label='Predictions', color='blue')
    plt.plot(targets, label='Targets', color='orange')
    plt.xlabel('Time Step')
    plt.ylabel('Return')
    plt.title('Predictions vs Targets')
    plt.legend()
    plt.savefig(fname + ".png")

def create_sequence(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)