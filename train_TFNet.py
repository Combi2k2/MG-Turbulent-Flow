from __future__ import unicode_literals, print_function, division
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#import matplotlib.pyplot as plt
import numpy as np

from models.TF_Net.TFNet import TF_Net
from utils.dataset import rbc_data
from utils.train import run_train, run_eval

import warnings
warnings.filterwarnings("ignore")

import argparse
import os

def arg_def():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type = str, default = "checkpoints")
    parser.add_argument("--batch_size",     type = int, default = 32)
    
    args = parser.parse_args()
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args.saved_checkpoint = os.path.join(args.checkpoint_dir, 'TF_Net_checkpoint.pth')
    
    return args

'''
    set up model
    best_params: kernel_size 3, learning_rate 0.001, dropout_rate 0, batch_size 120, input_length 25, output_length 4
'''
min_mse = 1
time_range  = 6
output_length = 4
input_length = 26
learning_rate = 0.001
batch_size = 32

if __name__ == '__main__':
    args = arg_def()
    
    model = TF_Net(input_channels = input_length * 2,
               output_channels = 2,
               kernel_size = 3, 
               dropout_rate = 0,
               time_range = time_range).to(args.device)
    model = nn.DataParallel(model)
    
    # set up optimizer, scheduler and train history
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, betas = (0.9, 0.999), weight_decay = 4e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.9)
    
    if os.path.exists(args.saved_checkpoint):
        loaded_checkpoint = torch.load(args.saved_checkpoint, map_location = torch.device(args.device))
        start_epoch = loaded_checkpoint['epoch'] + 1

        model.load_state_dict(loaded_checkpoint['model'])
        optimizer.load_state_dict(loaded_checkpoint['optim'])
        
        train_mse = loaded_checkpoint['train_mse']
        valid_mse = loaded_checkpoint['valid_mse']
    else:
        os.system('mkdir ' + args.checkpoint_dir)
        
        start_epoch = 0
        train_mse = []
        valid_mse = []
        
    
    # set up dataset
    data_prep = [
        torch.load('rbc_data/sample_0.pt'),
        torch.load('rbc_data/sample_1.pt'),
        torch.load('rbc_data/sample_2.pt'),
        torch.load('rbc_data/sample_3.pt')
    ]
    train_indices = list(range(5500))
    valid_indices = list(range(5500, 6000))
    
    train_set = rbc_data(data_prep, train_indices, input_length + time_range - 1, output_length, True)
    valid_set = rbc_data(data_prep, valid_indices, input_length + time_range - 1, output_length, True)    
    
    # choose metrics
    loss_fun = torch.nn.MSELoss()
    
    for i in range(start_epoch, 100):
        print(f'Epoch {i + 1}: ')
        scheduler.step()
        
        # run epochs:
        model.train();  train_mse.append(run_train(train_set, model, optimizer, loss_fun, args))
        model.eval();   mse, preds, trues = run_eval(valid_set, model, loss_fun, args)
        
        valid_mse.append(mse)
        
        # pocket algorithm
        if (min_mse > valid_mse[-1]):
            min_mse = valid_mse[-1] 
            best_model = model
        
        if (i == 99 or (len(train_mse) > 50 and np.mean(valid_mse[-5:]) >= np.mean(valid_mse[-10:-5]))):
            torch.save({
                'epoch': i,
                'model': best_model.state_dict(),
                'optim': optimizer.state_dict(),
                'train_mse': train_mse,
                'valid_mse': valid_mse
            }, 'TF_Net_checkpoint.pth')
        
        print(f">>  Eval MSE = {valid_mse[-1]}")

# loss_fun = torch.nn.MSELoss()
# best_model = torch.load("model.pth")
# test_set = Dataset(test_indices, input_length + time_range - 1, 40, 60, test_direc, True)
# test_loader = data.DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 8)
# preds, trues, loss_curve = test_epoch(test_loader, best_model, loss_fun)

# torch.save({"preds": preds,
#             "trues": trues,
#             "loss_curve": loss_curve}, 
#             "results.pt")