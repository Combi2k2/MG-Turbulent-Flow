import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils import data
import itertools
import re
import random
import time
from torch.autograd import Variable

def train_epoch(train_loader, model, optimizer, loss_function, device, coef = 0, regularizer = None):
    train_mse = []
    
    for batch_idx, (inputs, target) in enumerate(train_loader):
        inputs = inputs.to(device)
        target = target.to(device)
        
        loss = 0
        
        for tgt in target.transpose(0, 1):
            out = model(inputs)
            inputs = torch.cat([inputs[:, 2:], out], 1)
            
            if coef:    loss += loss_function(out, tgt) + coef * regularizer(out, tgt)
            else:       loss += loss_function(out, tgt)

        train_mse.append(loss.item() / target.shape[1])
        
        optimizer.zero_grad();  loss.backward()
        optimizer.step()
        
        if (batch_idx % 20 == 19):
            print(f'    Batch {batch_idx}: MSE = {train_mse[-1]}')
    
    return round(np.sqrt(np.mean(train_mse)), 5)

def eval_epoch(valid_loader, model, loss_function, device):
    valid_mse = []
    preds = []
    trues = []
    
    with torch.no_grad():
        for inputs, target in valid_loader:
            inputs = inputs.to(device)
            target = target.to(device)
            
            loss = 0
            ims = []
            
            for tgt in target.transpose(0, 1):
                out = model(inputs)
                inputs = torch.cat([inputs[:, 2:], out], 1)
                
                loss += loss_function(out, tgt)
                ims.append(out.cpu().data.numpy())
            
            preds.append(np.array(ims).transpose(1, 0, 2, 3, 4))
            trues.append(target.cpu().data.numpy())

            valid_mse.append(loss.item() / target.shape[1])
        
        preds = np.concatenate(preds, axis = 0)
        trues = np.concatenate(trues, axis = 0)
          
        valid_mse = round(np.sqrt(np.mean(valid_mse)), 5)

    return valid_mse, preds, trues