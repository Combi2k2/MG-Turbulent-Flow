import numpy as np
import torch
import math
from torch import nn
from torch.nn import functional as F

class MLP_Layer(nn.Module):
    # Conv Pooling Normalization
    # Conv Pooling 
    
    # Flatten
    # Linear
    # View (C, H, W)
    # Conv F.interpolate((H', W'), mode = '..')
    
    def __init__(self, frame_shape, num_past_frames = 16, num_future_frames = 4, hidden_channels = None):
        super().__init__()

        self.Flatten = nn.Flatten()
        self.layers = nn.ModuleList()
        
        self.frame_shape = frame_shape
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        
        if (hidden_channels is None):
            hidden_channels = [8]
        
        C, H, W = frame_shape
        
        for i in range(len(hidden_channels)):
            inp_channels = C * num_past_frames if i == 0 else hidden_channels[i]
            out_channels = hidden_channels[i + 1] if i < len(hidden_channels) - 1 else C * num_future_frames
            
            self.layers.append(nn.Linear(inp_channels * H * W, out_channels * H * W))
            self.layers.append(nn.BatchNorm1d(num_features = out_channels * H * W))
            self.layers.append(nn.ReLU(inplace=True))
        
        self.layers = nn.Sequential(*self.layers)
    
    def forward(self, inputs):
        N, T, C, H, W = inputs.size()
    
        x = inputs.view(N, -1)
        x = self.layers(x)
        x = x.view(N, self.num_future_frames, C, H, W)
        
        return x
        

if __name__ == '__main__':
    model = MLP_Layer((2, 64, 64))
    inputs = torch.randn(13, 16, 2, 64, 64)
    
    output = model(inputs)
    print(output.shape)