import numpy as np
import torch
import math
from torch import nn
from torch.nn import functional as F

from typing import Optional

from .Multigrid.MGconv import MGConvLayer
from .Multigrid.MGMemory import MGMemLayer
from .MGBlockEnc import MGBlockEnc

class MGxTransformer(nn.Module):
    def __init__(self, frame_shape, num_past_frames = 16, num_future_frames = 4,
        mem_hidden_dims = None,
        mem_start_levels = None,
        mem_end_levels = None,
        gen_hidden_dims = None,
        gen_start_levels = None,
        gen_end_levels = None
    ):
        super(MGxTransformer, self).__init__()
        
        self.frame_shape = frame_shape
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        
        C, H, W = frame_shape
        exponent = math.ceil(np.log(max(H, W)) / np.log(2))
        
        if (mem_hidden_dims is None):   mem_hidden_dims = [C << i for i in range(5)]
        if (gen_hidden_dims is None):   gen_hidden_dims = [(C * num_future_frames) << i for i in range(4, -1, -1)]
        
        if (mem_start_levels is None):  mem_start_levels = [exponent - 1] * 5
        if (gen_start_levels is None):  gen_start_levels = [exponent - 1] * 5
        
        if (mem_end_levels is None):    mem_end_levels = [exponent + 1] * 5
        if (gen_end_levels is None):    gen_end_levels = [exponent + 1] * 5
        
        assert(len(mem_hidden_dims) == len(mem_start_levels) and
               len(mem_hidden_dims) == len(mem_end_levels)), "Size of memory layers' config doens't match"

        assert(len(gen_hidden_dims) == len(gen_start_levels) and
               len(gen_hidden_dims) == len(gen_end_levels)), "Size of generating layers' config doesn't match"
        
        self.mem_layers = nn.ModuleList()
        self.gen_layers = nn.ModuleList()
        
        self.encoders = nn.ModuleList()
        
        for i in range(len(mem_hidden_dims)):
            if (i): low1, high1, dim1 = mem_start_levels[i - 1], mem_end_levels[i - 1], mem_hidden_dims[i - 1]
            else:   low1, high1, dim1 = exponent - 1, exponent, C
            
            low2, high2, dim2 = mem_start_levels[i], mem_end_levels[i], mem_hidden_dims[i]
            
            self.mem_layers.append(MGMemLayer(prev_start_level = low1,
                                              prev_end_level = high1,
                                              cur_start_level = low2,
                                              cur_end_level = high2,
                                              input_feature_chan = dim1,
                                              output_feature_chan = dim2,
                                              lay_ind = i + 1))

            if dim2 > 2:
                # add encoder
                start_level, end_level, embed_dim = low2, high2, dim2
                
                seq_len = num_past_frames
                num_heads = 1
                
                for i in range(1, int(np.sqrt(embed_dim)) + 1):
                    if (embed_dim % i == 0):
                        num_heads = i

                self.encoders.append(MGBlockEnc(start_level, end_level, seq_len, embed_dim, num_heads))
            else:
                self.encoders.append(nn.Identity())
        
        for i in range(len(gen_start_levels)):
            if (i): low1, high1, dim1 = gen_start_levels[i - 1], gen_end_levels[i - 1], gen_hidden_dims[i - 1]
            else:   low1, high1, dim1 = mem_start_levels[-1], mem_end_levels[-1], mem_hidden_dims[-1]
            
            low2, high2, dim2 = gen_start_levels[i], gen_end_levels[i], gen_hidden_dims[i]
            
            self.gen_layers.append(MGConvLayer(prev_start_level = low1,
                                               prev_end_level = high1,
                                               cur_start_level = low2,
                                               cur_end_level = high2,
                                               input_feature_chan = dim1,
                                               output_feature_chan = dim2,
                                               lay_ind = i + 6))

    def forward(self, inputs):
        N, T, C, H, W = inputs.size()
        inputs = inputs.view(N * T, C, H, W)
        exponent = math.ceil(np.log(max(H, W)) / np.log(2)) - 1
        
        inputs_scale = [
            F.interpolate(inputs, size = (1 << exponent, 1 << exponent), mode = 'nearest'),
            F.interpolate(inputs, size = (2 << exponent, 2 << exponent), mode = 'nearest')
        ]
        output_grids_list = []
        prev_grids = inputs_scale

        # mem layers
        for memory, encoder in zip(self.mem_layers, self.encoders):
            _, output_grids, _ = memory(prev_grids, T)
            output_grids_list.append(output_grids)
            
            prev_grids = output_grids_list[-1]
            prev_grids = encoder(prev_grids)
        
        for i in range(len(prev_grids)):
            _, chan, width, height = prev_grids[i].size()

            prev_grids[i] = prev_grids[i].view(N, T, chan, height, width)
            prev_grids[i] = prev_grids[i][:,-1,:,:,:]

        # generator layers
        for layer in self.gen_layers:
            _, output_grids = layer(prev_grids)
            output_grids_list.append(output_grids)
            
            prev_grids = output_grids_list[-1]
        
        out = prev_grids[-1]
        out = F.interpolate(out, size = (H, W), mode = 'nearest')
        
        out = out.reshape(N, self.num_future_frames, C, H, W)
        
        return out

    def residual_conn(self, x, y):
        _, chan_x, _, _ = x.size()
        _, chan_y, _, _ = y.size()

        if chan_x == chan_y:	return  x + y
        if chan_x <  chan_y:	return  F.pad(x,(0,0,0,0,0,chan_y - chan_x), "constant", 0) + y
        else:					return  x[:,:chan_y,:,:] + y

if __name__ == '__main__':
    model = MGxTransformer(frame_shape = (2, 64, 64), num_past_frames = 13)

    img = torch.randn(10, 13, 2, 64, 64)
    out = model(img)
    
    print(out.shape)
    