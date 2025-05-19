import torch
import torch.nn as nn
import numpy as np
from config import load_config
from lstm import LSTM
from util import *
from train import train, eval
from data_loader import *

def main():
    model_config_file_path = "./configs/model_config.yaml"
    eval_config_file_path = "./configs/eval_config.yaml"

    model_config = load_config(model_config_file_path)
    eval_config = load_config(eval_config_file_path)

    print("LOADED CONFIGS")

    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(model_config)

    (input, target) = train_dataloader.dataset[0]
    print(f"Example Input Shape: {input.shape}")
    print(f"Example Target Shape: {target.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if model_config['model'] == 'LSTM':
        print("Training LSTM model")
        model = LSTM(model_config['lstm_hidden_size'], model_config['lstm_num_layers']).to(device)
    else:
        print("Config Exception: Please specify a valid model in the config file.")

    # Train - will save best model weights
    train(model=model,
          device=device,
          train_dataloader=train_dataloader,
          val_dataloader=val_dataloader,
          model_config=model_config,
          eval_config=eval_config)
    
    avg_test_loss = eval(model=model,
                        device=device,
                        val_dataloader=test_dataloader,
                        model_config=model_config,
                        eval_config=eval_config)
    print(f"Avg Test Loss: {avg_test_loss}")

if __name__ == "__main__":
    main()