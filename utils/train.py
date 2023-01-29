import torch
import numpy as np
import logging

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def run_train(train_ds, model, optimizer, loss_func, args, coef = 0, regularizer = None):
    train_loader = DataLoader(train_ds, batch_size = args.batch_size, shuffle = True, num_workers = 8)
    train_mse = []
    
    for batch_idx, (inputs, target) in enumerate(train_loader):
        inputs = inputs.to(args.device)
        target = target.to(args.device)
        
        output = model(inputs)

        if coef:    loss = loss_func(output, target) + coef * regularizer(output, target)
        else:       loss = loss_func(output, target)

        train_mse.append(loss.item() / target.shape[1])
        
        optimizer.zero_grad();  loss.backward()
        optimizer.step()
        
        if (batch_idx % 20 == 19):
            logging.info(f'    Batch {batch_idx + 1}: MSE = {train_mse[-1]}')
    
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

            ims = output = model(inputs)
            ims = torch.transpose(ims, 0, 1).cpu().data.numpy()
            loss = loss_function(output, target)
            
            preds.append(np.array(ims).transpose(1, 0, 2, 3, 4))
            trues.append(target.cpu().data.numpy())

            valid_mse.append(loss.item() / target.shape[1])

        preds = np.concatenate(preds, axis = 0)
        trues = np.concatenate(trues, axis = 0)
          
        valid_mse = round(np.sqrt(np.mean(valid_mse)), 5)

    return valid_mse, preds, trues

if __name__ == '__main__':
    from models import MG
    from dataset import rbc_data
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    model = MG(frame_shape = (2, 64, 64),
            mem_start_levels = [5, 4, 4, 4, 4],
            mem_end_levels = [6, 6, 6, 6, 6],
            mem_hidden_dims = [4, 4, 8, 8, 16],
            
            gen_start_levels = [4, 4, 4, 5, 6],
            gen_end_levels = [6, 6, 6, 6, 6],
            gen_hidden_dims = [8, 8, 4, 4, 2]).to(device)
    
    sample_model = torch.nn.DataParallel(model)
    sample_optim = torch.optim.Adam(sample_model.parameters(), 1e-4, betas = (0.9, 0.999), weight_decay = 4e-4)
    
    sample_ds = rbc_data(
        [torch.load('../data/sample_0.pt')],
        list(range(100)),
        input_length = 16,
        output_length = 4,
        stack_x = False)
    class iter_args:
        def __init__(self):
            self.device = device
            self.batch_size = 8
            
    train_mse = run_train(sample_ds, sample_model, sample_optim, torch.nn.MSELoss(), iter_args())
    valid_mse, preds, trues = run_eval(sample_ds, sample_model, torch.nn.MSELoss(), iter_args())

    print(f'Train MSE: {train_mse}')
    print(f'Valid MSE: {valid_mse}')

# 1000 * 21 * 2 * 128 * 128

# utils\dataset\2D\CFD\Turbulent_Flow\rbc_data
# utils\dataset\2D\CFD\2D_Train_Rand\2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train