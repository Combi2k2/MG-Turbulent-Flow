from torch import nn
from .Multigrid.convlstm import ConvLSTM

class CLSTM(nn.Module):
    def __init__(self, frame_shape, num_past_frames, num_future_frames, hidden_dims = [64], num_layers = 1, dropout = 0.2):
        super(CLSTM, self).__init__()
        
        self.frame_shape = frame_shape
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        
        C, _, _ = frame_shape
        
        self.clstm = ConvLSTM(input_dim = C,
                        hidden_dim = hidden_dims,
                        kernel_size = (5, 5),
                        num_layers = num_layers,
                        dropout = dropout,
                        batch_first = True,
                        bias = True,
                        return_all_layers = False)
        
        self.output_layer = nn.Conv2d(in_channels = hidden_dims[-1], out_channels = num_future_frames * C, kernel_size = 5, padding = 2)
        
    def forward(self, inputs):
        N, _, C, H, W = inputs.size()
        
        assert(H == self.frame_shape[1])
        assert(W == self.frame_shape[2])
        
        out = self.clstm(inputs)[1][0][0]
        out = self.output_layer(out)
        
        return out.view(N, self.num_future_frames, C, H, W)

if __name__ == '__main__':
    import torch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CLSTM((2, 64, 64), 16, 4).to(device)
    
    inputs = torch.rand((32, 17, 2, 64, 64)).to(device)
    output = model(inputs)
    
    print(output.shape)
    
