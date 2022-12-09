import torch
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def run_train(train_ds, model, optimizer, loss_func, args, coef = 0, regularizer = None):
    train_loader = DataLoader(train_ds, batch_size = args.batch_size, shuffle = True, num_workers = 8)
    train_mse = []
    
    for batch_idx, (inputs, target) in enumerate(train_loader):
        inputs = inputs.to(args.device)
        target = target.to(args.device)
        
        loss = 0
        
        for tgt in target.transpose(0, 1):
            out = model(inputs)
            
            if (train_ds.stack_x):  inputs = torch.cat([inputs[:, out.shape[1]:], out], 1)
            else:                   inputs = torch.cat([inputs[:, 1:], torch.unsqueeze(out, 1)], 1)
            
            if coef:    loss += loss_func(out, tgt) + coef * regularizer(out, tgt)
            else:       loss += loss_func(out, tgt)

        train_mse.append(loss.item() / target.shape[1])
        
        optimizer.zero_grad();  loss.backward()
        optimizer.step()
        
        if (batch_idx % 20 == 19):
            print(f'    Batch {batch_idx + 1}: MSE = {train_mse[-1]}')
    
    return  round(np.sqrt(np.mean(train_mse)), 5)

def run_eval(valid_ds, model, loss_function, args):
    valid_loader = DataLoader(valid_ds, batch_size = args.batch_size, shuffle = False, num_workers = 8)
    valid_mse = []
    preds = []
    trues = []
    
    with torch.no_grad():
        for inputs, target in valid_loader:
            inputs = inputs.to(args.device)
            target = target.to(args.device)
            
            loss = 0
            ims = []
            
            for tgt in target.transpose(0, 1):
                out = model(inputs)
                if (valid_ds.stack_x):  inputs = torch.cat([inputs[:, out.shape[1]:], out], 1)
                else:                   inputs = torch.cat([inputs[:, 1:], torch.unsqueeze(out, 1)], 1)
                
                loss += loss_function(out, tgt)
                ims.append(out.cpu().data.numpy())
            
            preds.append(np.array(ims).transpose(1, 0, 2, 3, 4))
            trues.append(target.cpu().data.numpy())

            valid_mse.append(loss.item() / target.shape[1])
        
        preds = np.concatenate(preds, axis = 0)
        trues = np.concatenate(trues, axis = 0)
          
        valid_mse = round(np.sqrt(np.mean(valid_mse)), 5)

    return valid_mse, preds, trues

if __name__ == '__main__':
    from models.TF_Net.TFNet import TF_Net
    from dataset import rbc_data
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    time_range  = 6
    output_length = 4
    input_length = 16
    dropout_rate = 0
    kernel_size = 3
    
    model = TF_Net(input_channels = input_length*2, output_channels = 2, kernel_size = kernel_size, 
            dropout_rate = dropout_rate, time_range = time_range).to(device)
    model = torch.nn.DataParallel(model)

    data_prep = [torch.load('data/sample_0.pt')]
    sample_dt = rbc_data(data_prep, list(range(100)), input_length + time_range - 1, output_length, True)
    
    optim = torch.optim.Adam(model.parameters(), 1e-4, betas = (0.9, 0.999), weight_decay = 4e-4)
    
    class iter_args:
        def __init__(self):
            self.device = device
            self.batch_size = 32
    
    train_mse = run_train(sample_dt, model, optim, torch.nn.MSELoss(), iter_args())
    valid_mse, preds, trues = run_eval(sample_dt, model, torch.nn.MSELoss(), iter_args())

    print(f'Train MSE: {train_mse}')
    print(f'Valid MSE: {valid_mse}')