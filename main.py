import torch
from torch import nn

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
    args = parser.parse_args()
    return args

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_train(train_ds, model, criterion, optimizer, epoch, args):
    #model.train() 
    train_loader = DataLoader(train_ds, batch_size = args.batch_size, shuffle = True)

    for batch_idx, (inputs, target) in enumerate(train_loader):
        inputs = inputs.to(args.device)
        target = target.to(args.device)

        output = model(inputs)

        divergence = torch.sum(output, dim = 1)
        divergence = torch.sum(divergence ** 2) / 4096

        loss = criterion(output, target)

        if (batch_idx % 20 == 19):
            print('(Epoch %d) Batch %d: MSE = %f, Divergence = %f'%(epoch, batch_idx + 1, loss, divergence))

        loss += 0.1 * divergence

        optimizer.zero_grad();  loss.backward()
        optimizer.step()

def run_test(test_ds, model, criterion, args):
    test_loader = DataLoader(test_ds, batch_size = args.batch_size, shuffle = False)

    test_loss = 0
    test_div = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(test_loader):
            inputs = inputs.to(args.device)
            target = target.to(args.device)

            output = model(inputs)

            divergence = torch.sum(output, dim = 1)
            divergence = torch.sum(divergence ** 2) / 4096

            loss = criterion(output, target)

            test_loss += loss
            test_div += divergence

    print("\t**** Evaluation ****")
    print("Eval MSE = %f"%(test_loss / (batch_idx + 1)))
    print("Eval Div = %f"%(test_div / (batch_idx + 1)))

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
    learning_rate = 1e-3
    batch_size = args.batch_size
    n_epoch = 100

    # Set up optimizer and loss function
    criterion = nn.MSELoss().to(args.device)
    optim = torch.optim.Adam(model.parameters(), lr = 1e-3)

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
        run_train(train_ds, model, criterion, optim, epoch, args)
        run_test(valid_ds, model, criterion, args)

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optim': optim.state_dict()
        }, args.saved_checkpoint)