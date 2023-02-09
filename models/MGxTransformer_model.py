import numpy as np
import torch
import math
from torch import nn
from torch.nn import functional as F

from .VPTR.VPTR_modules import VPTREnc, VPTRDec, VPTRFormerFAR
from .Multigrid.MGconv import MGConvLayer
from .Multigrid.MGMemory import MGMemLayer

class MGxTransformer(nn.Module):
    def __init__(self, frame_shape, num_past_frames = 16, num_future_frames = 4):
        super(MGxTransformer, self).__init__()
        
        self.frame_shape = frame_shape
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        
        C, H, W = frame_shape
        exponent = math.ceil(np.log(max(H, W)) / np.log(2))
        
        self.enc = VPTREnc(C, n_downsampling = exponent - 3)
        self.dec = VPTRDec(C, n_downsampling = exponent - 3)
        
        self.transformer = VPTRFormerFAR(num_past_frames, num_encoder_layers = 5, dropout = 0.2)
        self.gen_layers = nn.ModuleList()
        
        self.gen_layers.append(MGConvLayer(exponent - 1, exponent + 1, exponent - 1, exponent + 1, num_past_frames * C, num_future_frames * C << 3, 6))
        self.gen_layers.append(MGConvLayer(exponent - 1, exponent + 1, exponent - 1, exponent + 1, num_future_frames * C << 3, num_future_frames * C << 2, 7))
        self.gen_layers.append(MGConvLayer(exponent - 1, exponent + 1, exponent - 1, exponent + 1, num_future_frames * C << 2, num_future_frames * C << 2, 8))
        self.gen_layers.append(MGConvLayer(exponent - 1, exponent + 1, exponent - 1, exponent + 1, num_future_frames * C << 2, num_future_frames * C << 1, 9))
        self.gen_layers.append(MGConvLayer(exponent - 1, exponent + 1, exponent - 1, exponent + 1, num_future_frames * C << 1, num_future_frames * C, 10))
        
        self.out_layer = MGConvLayer(exponent - 1, exponent + 1, exponent, exponent, num_future_frames * C, num_future_frames * C, 100)

    def forward(self, inputs):
        N, T, C, H, W = inputs.size()
        assert(C, H, W == self.frame_shape), "Inconsistent frame shape"
        
        exponent = math.ceil(np.log(max(H, W)) / np.log(2)) - 1
        new_H = 2 << exponent
        new_W = 2 << exponent
        
        inputs = F.interpolate(inputs.view(N * T, C, H, W), size = (new_H, new_W), mode = 'nearest')
        inputs = inputs.view(N, T, C, new_H, new_W)
        
        feats = self.enc(inputs)
        feats = self.transformer(feats)
        feats = self.dec(feats).view(N, T * C, new_H, new_W)
        
        inputs_scale = [
            F.interpolate(feats, size = (1 << exponent, 1 << exponent), mode = 'nearest'),
            F.interpolate(feats, size = (2 << exponent, 2 << exponent), mode = 'nearest'),
            F.interpolate(feats, size = (4 << exponent, 4 << exponent), mode = 'nearest')
        ]
        output_grids_list = []
        prev_grids = inputs_scale
        
        # generator layers
        for lay_ind, layer in enumerate(self.gen_layers):
            output_dims, output_grids = layer(prev_grids)
            output_grids_list.append(output_grids)
            
            if lay_ind % 2 == 0 and lay_ind > 0:
                for scale in range(len(output_grids)):
                    output_grids_list[-1][scale] = self.residual_conn(output_grids_list[-3][scale], output_grids_list[-1][scale])
            
            prev_grids = output_grids_list[-1]
        
        out = self.out_layer(prev_grids)[1][0]
        out = out.view(N, self.num_future_frames, C, H, W)
        
        return  out

    def residual_conn(self, x, y):
        _, chan_x, _, _ = x.size()
        _, chan_y, _, _ = y.size()

        if chan_x == chan_y:	return  x + y
        if chan_x <  chan_y:	return  F.pad(x,(0,0,0,0,0,chan_y - chan_x), "constant", 0) + y
        else:					return  x[:,:chan_y,:,:] + y

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MGxTransformer(frame_shape = (2, 64, 64), num_past_frames = 13, num_future_frames = 3).to(device)
    
    inputs = torch.randn(16, 13, 2, 64, 64).to(device)
    output = model(inputs)
    
    print(output.shape)
    