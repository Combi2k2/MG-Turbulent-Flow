from __future__ import unicode_literals, print_function, division
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#import matplotlib.pyplot as plt
import numpy as np

from models.model import MG
from utils.dataset import rbc_data
from utils.train import run_train, run_eval

import argparse
import os

def arg_def():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type = str, default = "checkpoints")
    parser.add_argument("--batch_size",     type = int, default = 32)
    
    args = parser.parse_args()
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args.saved_checkpoint = os.path.join(args.checkpoint_dir, 'MG_checkpoint.pth')
    
    return args

# Set up hyper parameters
learning_rate = 2e-5
n_epoch = 100

if __name__ == '__main__':
    args = arg_def()
    
    
    model = MG(nb_input_chan = 2).to(args.device)                       # Set up the model
    optim = torch.optim.Adam(model.parameters(), lr = learning_rate)    # Set up optimizer

    if os.path.exists(args.saved_checkpoint):
        loaded_checkpoint = torch.load(args.saved_checkpoint, map_location = torch.device(args.device))
        start_epoch = loaded_checkpoint['epoch'] + 1

        model.load_state_dict(loaded_checkpoint['model'])
        optim.load_state_dict(loaded_checkpoint['optim'])
        
        train_mse = loaded_checkpoint['train_mse']
        valid_mse = loaded_checkpoint['valid_mse']
    else:
        os.system('mkdir ' + args.checkpoint_dir)
        
        start_epoch = 0
        train_mse = []
        valid_mse = []
    
    # Set up dataset
    data_prep = [
        torch.load('rbc_data/sample_0.pt'),
        torch.load('rbc_data/sample_1.pt'),
        torch.load('rbc_data/sample_2.pt'),
        torch.load('rbc_data/sample_3.pt')
    ]
    train_ds = rbc_data(data_prep, split = 'train')
    valid_ds = rbc_data(data_prep, split = 'valid')
    
    for i in range(start_epoch, n_epoch):
        print(f'Epoch {i + 1}: ')
        
        # run epochs:
        model.train();  train_mse.append(run_train(train_ds, model, optim, nn.MSELoss(), args))
        model.eval();   mse, preds, trues = run_eval(valid_ds, model, nn.MSELoss(), args)
        
        valid_mse.append(mse)
        
        # pocket algorithm
        if (min_mse > valid_mse[-1]):
            min_mse = valid_mse[-1] 
            best_model = model
            
        torch.save({
            'epoch': i,
            'model': model.state_dict(),
            'optim': optim.state_dict(),
            'train_mse': train_mse,
            'valid_mse': valid_mse
        }, 'TF_Net_checkpoint.pth')
        
        print(f">>  Eval MSE = {valid_mse[-1]}")