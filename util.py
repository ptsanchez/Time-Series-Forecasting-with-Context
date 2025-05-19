import matplotlib as plt
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