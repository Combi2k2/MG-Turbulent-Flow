import torch
import torch.nn as nn
import torch.nn.functional as F

from model import MG

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import argparse
import os
import sys
import pdb

# Custom Dataset about Turbulent Flow
class rbc_data(Dataset):
    def __init__(self, data_prep, split = 'train', window_size = 16):
        self.data = []
        self.len = []
        self.window_size = window_size

        for flow in data_prep:
            flow_len = flow.size(0)
            mock = int(flow_len * 0.1)

            if split == 'train':    self.data.append(flow[mock:])
            if split == 'valid':    self.data.append(flow[:mock])

            self.len.append(self.data[-1].size(0) - window_size - 1)
    
    def __len__(self):
        return  sum(self.len)
    
    def __getitem__(self, index):
        for i, flow in enumerate(self.data):
            if (index < self.len[i]):
                inputs = flow[index : index + self.window_size]
                target = flow[index + self.window_size + 1]

                return  inputs, target
            
            index -= self.len[i]
            
        raise ValueError('Index out of range.')

def arg_def():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type = str, default = "checkpoints")
    parser.add_argument("--batch_size",     type = int, default = 32)
    parser.add_argument("--regulizer",      type = int, default = 0)

    args = parser.parse_args()
    return args

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_metrics(output, target):
    MSE_loss = F.mse_loss(output, target)

    divergence = output[:,:,2:,1:-1] + output[:,:,:-2,1:-1] + output[:,:,1:-1,2:] + output[:,:,1:-1,:-2] - 4 * output[:,:,1:-1,1:-1]
    divergence = torch.mean(divergence ** 2)

    return  {
        'MSE_Loss': MSE_loss,
        'Divergence': divergence
    }

def run_train(train_ds, model, optimizer, epoch, args):
    #model.train() 
    train_loader = DataLoader(train_ds, batch_size = args.batch_size, shuffle = True)

    for batch_idx, (inputs, target) in enumerate(train_loader):
        inputs = inputs.to(args.device)
        target = target.to(args.device)

        output  = model(inputs)
        metrics = compute_metrics(output, target)

        if (batch_idx % 50 == 49):
            print('(Epoch %d) Batch %d:'%(epoch, batch_idx + 1), end = "")
            print(' MSE = %f'%metrics['MSE_Loss'], end = '')
            print(' Divergence = %f'%metrics['Divergence'])
        
        loss = metrics['MSE_Loss'] + (0.1 * metrics['Divergence'] if args.regulizer else 0)

        optimizer.zero_grad();  loss.backward()
        optimizer.step()

def run_test(test_ds, model, args):
    test_loader = DataLoader(test_ds, batch_size = args.batch_size, shuffle = False)

    test_mse = 0
    test_div = 0

    print("\n\t******** Running Evaluation ********")
    
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(test_loader):
            inputs = inputs.to(args.device)
            target = target.to(args.device)

            output  = model(inputs)
            metrics = compute_metrics(output, target)

            test_mse += metrics['MSE_Loss']
            test_div += metrics['Divergence']
    
    print(">> Eval MSE = %f"%(test_mse / (batch_idx + 1)))
    print(">> Eval DIV = %f"%(test_div / (batch_idx + 1)))
    print("")

if __name__ == '__main__':
    args = arg_def()
    args.saved_checkpoint = os.path.join(args.checkpoint_dir, "checkpoint.pth")
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    data_prep = [
        torch.load('rbc_data/sample_0.pt'),
        torch.load('rbc_data/sample_1.pt'),
        torch.load('rbc_data/sample_2.pt'),
        torch.load('rbc_data/sample_3.pt')
    ]

    # Set up dataset
    train_ds = rbc_data(data_prep, split = 'train')
    valid_ds = rbc_data(data_prep, split = 'valid')

    # Set up the model
    model = MG(nb_input_chan = 2).to(args.device)

    # Set up hyper parameters
    learning_rate = 2e-5
    batch_size = args.batch_size
    n_epoch = 100

    # Set up optimizer
    optim = torch.optim.Adam(model.parameters(), lr = learning_rate)

    start_epoch = 0

    if os.path.exists(args.saved_checkpoint):
        loaded_checkpoint = torch.load(args.saved_checkpoint, map_location = torch.device(args.device))
        start_epoch = loaded_checkpoint['epoch'] + 1

        model.load_state_dict(loaded_checkpoint['model'])
        optim.load_state_dict(loaded_checkpoint['optim'])
    else:
        os.system('mkdir ' + args.checkpoint_dir)

    print("Number of parameters: %d"%count_parameters(model))

    for epoch in range(start_epoch, n_epoch):
        run_train(train_ds, model, optim, epoch, args)
        run_test(valid_ds, model, args)

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optim': optim.state_dict()
        }, args.saved_checkpoint)