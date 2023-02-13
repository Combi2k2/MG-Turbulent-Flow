"""
This file is modified base on https://github.com/NeuralOperator/fourier_neural_operator/blob/master/fourier_2d.py
This contains Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        N, C, H, W = x.size()
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(N, self.out_channels,  H, W // 2 + 1, dtype = torch.cfloat, device = x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        return torch.fft.irfft2(out_ft, s = (H, W))

class FNO2d(nn.Module):
    def __init__(self, frame_shape, num_past_frames = 16, num_future_frames = 4, modes1 = 16, modes2 = 16, width = 512):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """
        
        self.frame_shape = frame_shape
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        
        C, _, _ = frame_shape

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(C + 2, self.width) # (a(x, y), x, y)
        self.padding = 9

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        self.w0 = nn.Conv2d(in_channels = self.width, out_channels = self.width, kernel_size = 5, padding = 2)
        self.w1 = nn.Conv2d(in_channels = self.width, out_channels = self.width, kernel_size = 5, padding = 2)
        self.w2 = nn.Conv2d(in_channels = self.width, out_channels = self.width, kernel_size = 3, padding = 1)
        self.w3 = nn.Conv2d(in_channels = self.width, out_channels = self.width, kernel_size = 1)

        self.fc1 = nn.Linear(num_past_frames * self.width, 512)
        self.fc2 = nn.Linear(512, num_future_frames * C)

    def forward(self, inputs):
        N, T, C, H, W = inputs.size()
        
        gridx = torch.tensor(np.linspace(0, 1, H), dtype = torch.float)
        gridy = torch.tensor(np.linspace(0, 1, W), dtype = torch.float)
        
        gridx = gridx.reshape(1, H, 1, 1).repeat([N * T, 1, W, 1]).to(inputs.device)
        gridy = gridy.reshape(1, 1, W, 1).repeat([N * T, H, 1, 1]).to(inputs.device)
        
        x = inputs.reshape(N * T, C, H, W)
        x = torch.cat([x.permute(0, 2, 3, 1), gridx, gridy], dim = -1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.padding(x, [0, self.padding, 0, self.padding])
        
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = self.conv3(x) + self.w3(x)

        # x = x[..., :-self.padding, :-self.padding]
        x = x.reshape(N, T * self.width, H, W)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return x.permute(0, 3, 1, 2).reshape(N, self.num_future_frames, C, H, W)

if __name__ == '__main__':
    model = FNO2d((2, 64, 64), modes1 = 12, modes2 = 12, width = 32)

    inputs = torch.rand(13, 16, 2, 64, 64)
    output = model(inputs)
    
    print(output.shape)