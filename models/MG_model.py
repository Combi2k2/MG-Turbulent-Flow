from torch import nn
from torch.nn import functional as F

from MGconv import MGConvLayer
from MGMemory import MGMemLayer

from typing import Optional

import torch
import numpy as np
import math
class MG(nn.Module):
    def __init__(self, frame_shape, num_past_frames = 16, num_future_frames = 4,
        mem_hidden_dims: Optional[list[int]] = None,
        mem_start_levels: Optional[list[int]] = None,
        mem_end_levels: Optional[list[int]] = None,
        gen_hidden_dims: Optional[list[int]] = None,
        gen_start_levels: Optional[list[int]] = None,
        gen_end_levels: Optional[list[int]] = None
    ):
        super(MG, self).__init__()
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
        
        inputs2scales = [
            F.interpolate(inputs, size = (1 << exponent, 1 << exponent), mode = 'nearest'),
            F.interpolate(inputs, size = (2 << exponent, 2 << exponent), mode = 'nearest')
        ]
        
        output_grids_list = []
        prev_grids = inputs2scales
        
        # mem layers
        for lay_ind, mem_layer in enumerate(self.mem_layers):
            output_dims, output_grids, lstm_states = mem_layer(prev_grids, T)
            output_grids_list.append(output_grids)
            
            if lay_ind % 2 == 0 and lay_ind > 0:
                for scale in range(len(output_grids)):
                    output_grids_list[-1][scale] = self.residual_conn(output_grids_list[-3][scale], output_grids_list[-1][scale])
                
            prev_grids = output_grids_list[-1]
        
        for i in range(len(prev_grids)):
            _, chan, width, height = prev_grids[i].size()
            
            prev_grids[i] = prev_grids[i].view(N, T, chan, height, width)
            prev_grids[i] = torch.mean(prev_grids[i], dim = 1)
            
        # generator layers
        for layer in self.gen_layers:
            output_dims, output_grids = layer(prev_grids)
            prev_grids = output_grids
        
        out = prev_grids[-1]
        out = F.interpolate(out, size = (H, W), mode = 'nearest')
        
        out = out.view(N, self.num_future_frames, C, H, W)
        
        return  out

    def residual_conn(self, x, y):
        _, chan_x, _, _ = x.size()
        _, chan_y, _, _ = y.size()
        
        if chan_x == chan_y:	return  x + y
        if chan_x <  chan_y:	return  F.pad(x,(0,0,0,0,0,chan_y - chan_x), "constant", 0) + y
        else:					return  x[:,:chan_y,:,:] + y

if __name__ == '__main__':
	model = MG(frame_shape = (2, 64, 64))

	inputs = torch.randn(13, 16, 2, 64, 64)
	output = model(inputs)

	print(output.shape)