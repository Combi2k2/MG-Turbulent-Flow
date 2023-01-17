import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvFFN(nn.Module):
    """
    https://github.com/HRNet/HRFormer/blob/main/cls/models/modules/ffn_block.py
    """
    def __init__(self, frame_shape, num_past_frames = 16, num_future_frames = 4, hidden_channels = None):
        super().__init__()
        
        self.frame_shape = frame_shape
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        self.layers = nn.ModuleList()
        
        C, H, W = frame_shape
        
        if (hidden_channels is None):
            hidden_channels = [32, 64, 128, 128, 64, 32]
        
        for i in range(len(hidden_channels)):
            inp_channels = C * num_past_frames if i == 0 else hidden_channels[i]
            out_channels = hidden_channels[i + 1] if i < len(hidden_channels) - 1 else C * num_future_frames
            
            self.layers.append(nn.Conv2d(inp_channels, out_channels, kernel_size = (5, 5), stride = 1, padding = 2))
            self.layers.append(nn.BatchNorm2d(out_channels))
            
            if (i % 2 == 1):
                self.layers.append(nn.ReLU(inplace = True))
                self.layers.append(nn.Dropout(0.2))
        
        self.layers = nn.Sequential(*self.layers)

    def forward(self, inputs):
        """
        x: (N, T, C, H, W)
        """
        N, T, C, H, W = inputs.shape
        
        x = inputs.view(N, C * T, H, W)
        x = self.layers(x)
        x = x.view(N, self.num_future_frames, C, H, W)
        
        return x

if __name__ == '__main__':
    model = ConvFFN((2, 64, 64))
    inputs = torch.randn(13, 16, 2, 64, 64)
    
    output = model(inputs)
    print(output.shape)
    