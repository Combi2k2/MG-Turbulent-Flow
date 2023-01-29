from .VPTR.VidHRFormer_modules import VidHRFormerBlockEnc
from .VPTR.position_encoding import PositionEmbeddding1D, PositionEmbeddding2D
from torch import nn
from torch import Tensor

import torch
import torch.nn.functional as F

class MGBlockEnc(nn.Module):
    def __init__(self, start_level, end_level, seq_len, embed_dim, num_heads, window_size = 7, dropout = 0., drop_path = 0., Spatial_FFN_hidden_ratio = 4, far = True, rpe = True):
        super().__init__()
        
        self.start_level = start_level
        self.end_level = end_level
        self.seq_len = seq_len
        
        pos1d = PositionEmbeddding1D()
        pos2d = PositionEmbeddding2D()
        
        middle = 1 << ((start_level + end_level) // 2)
        
        temporal_pos = pos1d(L = seq_len, N = 1, E = embed_dim)[:, 0, :]
        spatial_pos = pos2d(N = 1, E = embed_dim, H = middle, W = middle)[0].permute(1, 2, 0)
        
        self.register_buffer('temporal_pos', temporal_pos)
        self.register_buffer('spatial_pos', spatial_pos)
        
        args = [middle, middle, embed_dim, num_heads]
        kwargs = {
            'window_size': window_size,
            'dropout' : dropout,
            'drop_path': drop_path,
            'Spatial_FFN_hidden_ratio': Spatial_FFN_hidden_ratio,
            'far': far,
            'rpe': rpe
        }
        
        self.encoder = VidHRFormerBlockEnc(*args, **kwargs)
    
    def forward(self, grids):
        middle = 1 << ((self.start_level + self.end_level) // 2)
        grid_concat = []
        
        for grid in grids:
            _, C, H, W = grid.size()
            grid = F.interpolate(grid, size = (middle, middle), mode = 'nearest')
            grid = grid.view(-1, self.seq_len, C, middle, middle)
            grid_concat.append(grid)

        grid_concat = torch.concat(grid_concat, dim = 0)
        grid_encoded = self.encoder(grid_concat, self.spatial_pos, self.temporal_pos)
        
        output_grids = []
        
        for i in range(len(grids)):
            NT, C, H, W = grids[i].size()
            N = NT // self.seq_len
            
            output_grid = grid_encoded[i * N: i * N + N].reshape(NT, C, middle, middle)
            output_grid = F.interpolate(output_grid, size = (H, W), mode = 'nearest')
            
            output_grids.append(output_grid)
        
        return output_grids

if __name__ == '__main__':
    import torch
    
    model = MGBlockEnc(5, 7, 13, 32, 4)
    
    inputs = [
        torch.randn(130, 32, 32, 32),
        torch.randn(130, 32, 64, 64),
        torch.randn(130, 32, 128, 128)
    ]
    
    outputs = model(inputs)
    
    print("Combiiiii")
            
         