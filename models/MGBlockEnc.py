from VPTR.VidHRFormer_modules import VidHRFormerBlockEnc
from VPTR.position_encoding import PositionEmbeddding1D, PositionEmbeddding2D
from torch import nn

class MGBlockEnc(nn.Module):
    def __init__(self, start_level, end_level, seq_len, embed_dim, num_heads, window_size = 7, dropout = 0., drop_path = 0., Spatial_FFN_hidden_ratio = 4, far = True, rpe = True):
        super().__init__()
        
        self.start_level = start_level
        self.end_level = end_level
        self.seq_len = seq_len
        
        pos1d = PositionEmbeddding1D()
        pos2d = PositionEmbeddding2D()
        
        temporal_pos = pos1d(L = seq_len, N = 1, E = embed_dim)[:, 0, :]
        self.register_buffer('temporal_pos', temporal_pos)
        
        self.spatial_pos = []
        self.encoders = nn.ModuleList()
        
        for i in range(start_level, end_level + 1):
            args = [1 << i, 1 << i, embed_dim, num_heads]
            kwargs = {
                'window_size': window_size,
                'dropout' : dropout,
                'drop_path': drop_path,
                'Spatial_FFN_hidden_ratio': Spatial_FFN_hidden_ratio,
                'far': far,
                'rpe': rpe
            }
            
            self.encoders.append(VidHRFormerBlockEnc(*args, **kwargs))
            self.spatial_pos.append(pos2d(N = 1, E = embed_dim, H = (1 << i), W = (1 << i))[0, ...].permute(1, 2, 0))
    
    def forward(self, grids):
        output_grids = []
        
        for i, encoder in enumerate(self.encoders):
            _, C, H, W = grids[i].shape
            
            grids_reshaped = grids[i].view(-1, self.seq_len, C, H, W)
            grids_encoded = encoder(grids_reshaped, self.spatial_pos[i], self.temporal_pos)
            
            N, T, C, H, W = grids_encoded.shape
            
            output_grids.append(grids_encoded.reshape(N * T, C, H, W))
        
        return output_grids

if __name__ == '__main__':
    import torch
    
    model = MGBlockEnc(4, 6, 13, 20, 4)
    
    inputs = [
        torch.randn(130, 20, 16, 16),
        torch.randn(130, 20, 32, 32),
        torch.randn(130, 20, 64, 64)
    ]
    
    outputs = model(inputs)
    
    print("Combiiiii")
            
         