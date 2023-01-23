from __future__ import unicode_literals, print_function, division
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from train import run_train, run_eval
from datetime import datetime

import logging.config
import logging
import argparse
import os

def arg_def(default_checkpoint_dir = 'checkpoints', model_name = 'MG_model'):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type = str, default = default_checkpoint_dir)
    parser.add_argument("--batch_size",     type = int, default = 32)
    
    args = parser.parse_args()
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args.saved_checkpoint = os.path.join(args.checkpoint_dir, model_name + '_checpoint.pth')
    
    return args

# Set up hyper parameters
learning_rate = 2e-5
n_epoch = 10

default_logging_config = {
    'filename' : 'train.log',
    'level'    : logging.INFO,
    # 'format'   : "{asctime} {levelname:<8} {message}"
}

class Trainer:
    def __init__(self, model, optim, train_ds, valid_ds, args, logging_config = default_logging_config):
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        
        self.model = model
        self.optim = optim
    
        self.args = args

        self.min_mse = 1
        self.best_model = model

        logging.basicConfig(**logging_config)
        logging.captureWarnings(True)

        if os.path.exists(args.saved_checkpoint):
            loaded_checkpoint = torch.load(args.saved_checkpoint, map_location = torch.device(args.device))
            self.start_epoch = loaded_checkpoint['epoch'] + 1

            self.model.load_state_dict(loaded_checkpoint['model'])
            self.optim.load_state_dict(loaded_checkpoint['optim'])
            self.best_model.load_state_dict(loaded_checkpoint['best_model'])
            
            self.train_mse = loaded_checkpoint['train_mse']
            self.valid_mse = loaded_checkpoint['valid_mse']

            self.min_mse = loaded_checkpoint['min_mse']
        else:
            os.system('mkdir ' + args.checkpoint_dir)
            
            self.start_epoch = 0
            self.train_mse = []
            self.valid_mse = []
    
    def train(self):
        # check if our model is Auto Regressive
        sample_input, sample_target = self.train_ds[0]
        sample_output = self.model(sample_input[None, :])
        print(sample_input.shape, sample_output.shape)

        if (len(sample_input.shape) + 1 != len(sample_output.shape)):
            one_output_frame = True
        else:
            one_output_frame = False
        print(one_output_frame)
        # done checking

        for i in range(self.start_epoch + 1, n_epoch):
            logging.info(f'Epoch {i + 1}: Start at {datetime.now()}')
            # run epochs:
            self.model.train()
            self.train_mse.append(run_train(self.train_ds, self.model, self.optim, nn.MSELoss(), self.args, one_output_frame = one_output_frame))

            self.model.eval()
            mse, _, _ = run_eval(self.valid_ds, self.model, nn.MSELoss(), self.args, one_output_frame = one_output_frame)
            
            self.valid_mse.append(mse)
            
            # pocket algorithm
            if (self.min_mse > self.valid_mse[-1]):
                self.min_mse = self.valid_mse[-1] 
                self.best_model = self.model
                
            torch.save({
                'epoch': i,
                'model': self.model.state_dict(),
                'optim': self.optim.state_dict(),
                'train_mse': self.train_mse,
                'valid_mse': self.valid_mse,
                'min_mse': self.min_mse,
                'best_model': self.best_model.state_dict()
            }, self.args.saved_checkpoint)

            logging.info(f'>>  Eval MSE = {self.valid_mse[-1]}')
            logging.info(f'Epoch {i + 1}: Finished at {datetime.now()}')

if __name__ == '__main__':
    from dataset import rbc_data
    
    class test_model(nn.Module):
        def __init__(self):
            super().__init__()

            self.conv = nn.Conv2d(32, 8, kernel_size = (3, 3), stride =  1, padding = 1)

        def forward(self, inputs):
            """
            x: (N, T, C, H, W)
            """
            N, T, C, H, W = inputs.shape
            
            x = inputs.view(N, C * T, H, W)
            x = self.conv(x)
            x = x.view(N, 4, C, H, W)
            return x

    model = test_model()
    Optim = torch.optim.Adam(model.parameters(), lr = learning_rate)

    args = arg_def(model_name = 'test')

    # Set up dataset
    data_prep = [
        torch.load('dataset/2D/CFD/Turbulent_Flow/rbc_data/sample_0.pt'),
        torch.load('dataset/2D/CFD/Turbulent_Flow/rbc_data/sample_1.pt')
    ]
    train_indices = list(range(100))
    valid_indices = list(range(100, 150))
    
    train_ds = rbc_data(data_prep, train_indices, 16, 4, stack_x = False)
    valid_ds = rbc_data(data_prep, valid_indices, 16, 4, stack_x = False)

    trainer = Trainer(model, Optim, train_ds, valid_ds, args)
    trainer.train()