from __future__ import unicode_literals, print_function, division
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

import numpy as np
from datetime import datetime

import logging.config
import logging
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_train(train_ds, model, optimizer, loss_function, batch_size, coef = 0, regularizer = None):
    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True, num_workers = 8)
    train_loss = []
    
    for batch_idx, (inputs, target) in enumerate(train_loader):        
        inputs = inputs.to(device)
        target = target.to(device)
        
        output = model(inputs)

        if coef:    loss = loss_function(output, target) + coef * regularizer(output, target)
        else:       loss = loss_function(output, target)

        train_loss.append(loss.item() / target.shape[1])
        
        optimizer.zero_grad();  loss.backward()
        optimizer.step()
        
        if (batch_idx % 20 == 19):
            logging.info(f'    Batch {batch_idx + 1}: MSE = {train_loss[-1]}')
    
    return  np.mean(train_loss)

def run_eval(valid_ds, model, loss_function, batch_size = 32):
    valid_loader = DataLoader(valid_ds, batch_size = batch_size, shuffle = False, num_workers = 8)
    valid_loss = []
    preds = []
    trues = []
    
    with torch.no_grad():
        for inputs, target in valid_loader:
            inputs = inputs.to(device)
            target = target.to(device)

            ims = output = model(inputs)
            ims = torch.transpose(ims, 0, 1).cpu().data.numpy()
            loss = loss_function(output, target)
            
            preds.append(np.array(ims).transpose(1, 0, 2, 3, 4))
            trues.append(target.cpu().data.numpy())

            valid_loss.append(loss.item() / target.shape[1])

        preds = np.concatenate(preds, axis = 0)
        trues = np.concatenate(trues, axis = 0)

    return np.mean(valid_loss), preds, trues

class CheckPointArgs:
    def __init__(self, model_name, experiment_name, checkpoint_dir = 'checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        self.saved_checkpoint = os.path.join(checkpoint_dir, f'{model_name}_{experiment_name}_checpoint.pth')
        self.model_name = model_name
        self.experiment_name = experiment_name

class TrainArgs:
    def __init__(self, num_epochs = 100, learning_rate = 2e-5, batch_size = 32):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

class Trainer:
    def __init__(self,
        model,
        train_ds,
        valid_ds,
        checkpoint_args,
        training_args,
        logging_config = {'level': logging.INFO}
    ):
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        
        self.model = model.to(device)
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr = training_args.learning_rate, betas = (0.9, 0.999), weight_decay = 4e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = 1, gamma = 0.9)
        
        self.checkpoint_args = checkpoint_args
        self.training_args = training_args

        self.min_mse = 1
        self.best_model = model

        logging.basicConfig(**logging_config)
        logging.captureWarnings(True)

        if os.path.exists(checkpoint_args.saved_checkpoint):
            loaded_checkpoint = torch.load(checkpoint_args.saved_checkpoint, map_location = device)

            self.model.load_state_dict(loaded_checkpoint['model'])
            self.best_model.load_state_dict(loaded_checkpoint['best_model'])
            
            self.train_mse = loaded_checkpoint['train_mse']
            self.valid_mse = loaded_checkpoint['valid_mse']

            self.min_mse = loaded_checkpoint['min_mse']
        else:    
            os.system('mkdir -p ' + checkpoint_args.checkpoint_dir)
            
            self.start_epoch = -1
            self.train_mse = []
            self.valid_mse = []
    
    def train(self):
        for _ in range(self.training_args.num_epochs):
            logging.info(f'Epoch {len(self.train_mse) + 1}: Start at {datetime.now()}')
            # run epochs:
            torch.cuda.empty_cache()
            
            self.model.train()
            self.train_mse.append(run_train(self.train_ds, self.model, self.optimizer, nn.MSELoss(), self.training_args.batch_size))

            self.scheduler.step()
            
            self.model.eval()
            mse, _, _ = run_eval(self.valid_ds, self.model, nn.MSELoss(), self.training_args.batch_size)
            
            self.valid_mse.append(mse)
            
            # pocket algorithm
            if (self.min_mse > self.valid_mse[-1]):
                self.min_mse = self.valid_mse[-1] 
                self.best_model = self.model
                
            torch.save({
                'model': self.model.state_dict(),
                'train_mse': self.train_mse,
                'valid_mse': self.valid_mse,
                'min_mse': self.min_mse,
                'best_model': self.best_model.state_dict()
            }, self.checkpoint_args.saved_checkpoint)

            logging.info(f'>>  Eval MSE = {self.valid_mse[-1]}')
            logging.info(f'Epoch {len(self.train_mse)}: Finish at {datetime.now()}')

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
    
    checkpoint_args = CheckPointArgs('test', 'rbc')
    training_args = TrainArgs(num_epochs = 2, batch_size = 2)

    # Set up dataset
    data_prep = [
        torch.load('../data/sample_0.pt')
    ]
    train_indices = list(range(100))
    valid_indices = list(range(100, 150))
    
    train_ds = rbc_data(data_prep, train_indices, 16, 4, stack_x = False)
    valid_ds = rbc_data(data_prep, valid_indices, 16, 4, stack_x = False)

    trainer = Trainer(model, train_ds, valid_ds, checkpoint_args, training_args)
    trainer.train()
