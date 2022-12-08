import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils import data
import itertools
import re
import random
import time
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings("ignore")

class rbc_data(data.Dataset):
    def __init__(self, data_prep, input_length, output_length, split = 'train'):
        self.data = []
        self.len = []
        self.input_length  = input_length
        self.output_length = output_length

        for flow in data_prep:
            flow_len = flow.size(0)
            mock = int(flow_len * 0.1)

            if split == 'train':    self.data.append(flow[mock:])
            if split == 'valid':    self.data.append(flow[:mock])

            self.len.append(self.data[-1].size(0) - input_length - output_length)
    
    def __len__(self):
        return  sum(self.len)
    
    def __getitem__(self, index):
        for i, flow in enumerate(self.data):
            if (index < self.len[i]):
                inputs = flow[index : index + self.input_length];   index += self.input_length
                target = flow[index : index + self.output_length]

                inputs = inputs.reshape(-1, 64, 64)

                return  inputs, target
            
            index -= self.len[i]
            
        raise ValueError('Index out of range.')
    
def train_epoch(train_loader, model, optimizer, loss_function, coef = 0, regularizer = None):
    train_mse = []
    for batch_idx, (xx, yy) in enumerate(train_loader):
        loss = 0
        ims = []
        xx = xx.to(device)
        yy = yy.to(device)
    
        for y in yy.transpose(0,1):
            im = model(xx)
            xx = torch.cat([xx[:, 2:], im], 1)
      
            if coef != 0 :
                loss += loss_function(im, y) + coef*regularizer(im, y)
            else:
                loss += loss_function(im, y)
            ims.append(im.cpu().data.numpy())
        
        if (batch_idx % 20 == 19):
            print(f'    Batch {batch_idx}: MSE = {loss.item() / yy.shape[1]}')
            
        ims = np.concatenate(ims, axis = 1)
        train_mse.append(loss.item()/yy.shape[1])

        optimizer.zero_grad();  loss.backward()
        optimizer.step()
    train_mse = round(np.sqrt(np.mean(train_mse)),5)
    return train_mse

def eval_epoch(valid_loader, model, loss_function):
    valid_mse = []
    preds = []
    trues = []
    with torch.no_grad():
        for xx, yy in valid_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
            ims = []

            for y in yy.transpose(0,1):
                im = model(xx)
                xx = torch.cat([xx[:, 2:], im], 1)
                loss += loss_function(im, y)
                ims.append(im.cpu().data.numpy())
  
            ims = np.array(ims).transpose(1,0,2,3,4)
            preds.append(ims)
            trues.append(yy.cpu().data.numpy())
            valid_mse.append(loss.item()/yy.shape[1])
        preds = np.concatenate(preds, axis = 0)  
        trues = np.concatenate(trues, axis = 0)  
        valid_mse = round(np.sqrt(np.mean(valid_mse)), 5)
    return valid_mse, preds, trues

def test_epoch(test_loader, model, loss_function):
    valid_mse = []
    preds = []
    trues = []
    with torch.no_grad():
        loss_curve = []
        for xx, yy in test_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            
            loss = 0
            ims = []

            for y in yy.transpose(0,1):
                im = model(xx)
                xx = torch.cat([xx[:, 2:], im], 1)
                mse = loss_function(im, y)
                loss += mse
                loss_curve.append(mse.item())
                
                ims.append(im.cpu().data.numpy())

            ims = np.array(ims).transpose(1,0,2,3,4)    
            preds.append(ims)
            trues.append(yy.cpu().data.numpy())            
            valid_mse.append(loss.item()/yy.shape[1])

        preds = np.concatenate(preds, axis = 0)  
        trues = np.concatenate(trues, axis = 0)
        
        valid_mse = round(np.mean(valid_mse), 5)
        loss_curve = np.array(loss_curve).reshape(-1,60)
        loss_curve = np.sqrt(np.mean(loss_curve, axis = 0))
    return preds, trues, loss_curve