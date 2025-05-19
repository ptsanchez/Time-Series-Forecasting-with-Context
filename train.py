from util import *
import torch.nn as nn
import torch
import numpy as np
import os

import uuid 
import shutil
import csv

def train(model, device, train_dataloader, val_dataloader, model_config, eval_config):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config['learning_rate'], weight_decay=model_config['weight_decay'])
    
    unique_code_str = uuid.uuid4()

    models_folder = './test/'
    model_folder = models_folder+f"{model_config['model']}_{unique_code_str}/"
    try:
        os.makedirs(model_folder)
        print(f"Model folder created at {model_folder}")
    except FileExistsError:
        print(f"Folder already exists: {model_folder}")
    except Exception as e:
        print(f"Error creating folder: {e}")

    original_model_config = "./configs/model_config.yaml"
    original_eval_config = "./configs/eval_config.yaml"

    early_stopper = EarlyStopper(model_config['patience'], model_config['early_stop_epochs'])

    unique_code = uuid.uuid4()
    unique_code_str = str(unique_code)

    train_loss = []
    val_loss = []

    shutil.copyfile(original_model_config, model_folder+f"model_config_{unique_code_str}.yaml")
    shutil.copyfile(original_eval_config, model_folder+f"eval_config_{unique_code_str}.yaml")

    min_val_loss = float('inf')

    for t in range(model_config['epochs']):
        epoch_train_loss = []
        lowest_epoch_avg_loss = float('inf')

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {t}: lr = {current_lr}")
        for i, (input, target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            input, target = input.to(device), target.to(device)
            pred = model(input)
            loss = criterion(pred, target)

            # Append first train loss for plotting purposes
            #if (len(train_loss) == 0):
            #    train_loss.append(loss.detach().cpu().item())

            epoch_train_loss.append(loss.detach().cpu().item())

            if i % 100 == 0:
                if lowest_epoch_avg_loss > np.mean(epoch_train_loss):
                    print(f"\033[91mEpoch {t}, Batch {i}: Loss: {np.mean(epoch_train_loss)}\033[0m")
                else:
                    print(f"\033[92mEpoch {t}, Batch {i}: Loss: {np.mean(epoch_train_loss)}\033[0m")
                    lowest_epoch_avg_loss = np.mean(epoch_train_loss)
            
            loss.backward()
            optimizer.step()

        avg_train_loss = np.mean(epoch_train_loss)
        avg_val_loss, eval_cell_states = eval(model, device, val_dataloader, model_config, eval_config)

        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_folder+f"{model_config['model']}_seqlen{model_config['seq_len']}_{unique_code_str}.pth")

        if early_stopper.early_stop(avg_val_loss, t):
            save_train(train_loss, val_loss, model_folder, model_config, unique_code_str)

        print(f"Epoch {t}: Avg Training Loss: {avg_train_loss}, Avg Validation Loss: {avg_val_loss}")

        train_loss.append(avg_train_loss)
        val_loss.append(avg_val_loss)

    save_train(train_loss, val_loss, model_folder, model_config, unique_code_str)

def eval(model, device, val_dataloader, model_config, eval_config, criterion=nn.MSELoss()):
    val_loss = []
    for i, (input, target) in enumerate(val_dataloader):
        input, target = input.to(device), target.to(device)
        pred = model(input)
        loss = criterion(pred, target)
        val_loss.append(loss.detach().cpu().numpy())
    return np.mean(val_loss), None

def save_train(train_loss, val_loss, model_folder, model_config, unique_code_str):
    plot_losses(train_loss, val_loss, model_folder+f"loss_{model_config['model']}_seqlen{model_config['seq_len']}_{unique_code_str}")
    csv_filename = model_folder + f"train_{model_config['model']}_seqlen{model_config['seq_len']}_{unique_code_str}.csv"
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "val_loss", "validation_loss"])
        for epoch_idx in range(len(train_loss)):
            writer.writerow([epoch_idx, train_loss[epoch_idx], val_loss[epoch_idx]])
    print("CSV file saved")


class EarlyStopper():
    def __init__(self, patience, early_stop_epochs):
        self.min_val_loss = float('inf')
        self.patience = patience
        self.early_stop_epochs = early_stop_epochs
        self.increasing_loss_count = 0

    def early_stop(self, val_loss, epoch):
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.increasing_loss_count = 0
        elif epoch >= self.early_stop_epochs:
            self.increasing_loss_count += 1
            if self.increasing_loss_count >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                return True
        return False