import torch
import torch.nn as nn
import numpy as np

class LSTM(nn.Module):
    def __init__(self, hidden_size, num_layers, input_size=1, output_size=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(2)

        output, _ = self.lstm(x)
        output = np.squeeze(self.fc(output[:, -1, :])) # Extract last time step output

        return output