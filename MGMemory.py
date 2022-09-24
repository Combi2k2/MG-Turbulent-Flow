import torch
from torch import nn
from torch.nn.functional import upsample

from convlstm import ConvLSTM

class MGMemLayer(nn.Module):
	def __init__(self, prev_start_level, prev_end_level, cur_start_level, cur_end_level, input_feature_chan, output_feature_chan, lay_ind):
		super(MGMemLayer, self).__init__()
		self.nb_levels = cur_end_level - cur_start_level + 1
		self.kernel_size = (3, 3)
		# Each level will be processed by a convlstm and followed by batchnorm.
		self.convlstms  = nn.ModuleList()
		self.batchnorms = nn.ModuleList()

		self.num_input_grids = prev_end_level - prev_start_level + 1

		self.prev_start_level = prev_start_level
		self.prev_end_level   = prev_end_level

		self.cur_start_level = cur_start_level
		self.cur_end_level   = cur_end_level

		self.output_feature_chan = output_feature_chan 
		self.lay_ind = lay_ind
		self.maxpool = nn.MaxPool2d(2, stride = 2)

		for i in range(cur_start_level,cur_end_level+1):
			sum_input_chan = 0
			iprev = i - prev_start_level

			if self._check_in_range(iprev - 1): sum_input_chan += input_feature_chan
			if self._check_in_range(iprev):     sum_input_chan += input_feature_chan
			if self._check_in_range(iprev + 1): sum_input_chan += input_feature_chan

			self.convlstms.append(ConvLSTM(input_dim = sum_input_chan,
										   hidden_dim = output_feature_chan,
										   kernel_size = self.kernel_size,
										   num_layers = 1,
										   batch_first = True,
										   return_all_layers = True))

			self.batchnorms.append(nn.BatchNorm2d(output_feature_chan))          

	def forward(self, prev_grids, seq_len):
		output_grids = []
		lstm_states = []
		output_dims = []

		for i in range(self.cur_start_level, self.cur_end_level + 1):
			output_dim = (self.output_feature_chan, 1 << i, 1 << i)
			output_dims.append(output_dim)
			concat_grid = None
			iprev = i - self.prev_start_level

			if self._check_in_range(iprev - 1):
				prev_spatial_dim = (prev_grids[iprev - 1].shape[2], prev_grids[iprev - 1].shape[3])
				next_spatial_dim = (prev_spatial_dim[0] * 2, prev_spatial_dim[1] * 2)
				prev_up = upsample(prev_grids[iprev - 1], size = next_spatial_dim, mode = 'nearest')
				concat_grid = prev_up
			
			if self._check_in_range(iprev):
				if concat_grid is None:	concat_grid = prev_grids[iprev]
				else:					concat_grid = torch.cat([concat_grid, prev_grids[iprev]], dim = 1)

			if self._check_in_range(iprev + 1):
				prev_down = self.maxpool(prev_grids[iprev + 1]) #tf.layers.max_pooling2d(prev_grids[iprev+1],[2,2],2)
				if concat_grid is None:	concat_grid = prev_down
				else:					concat_grid = torch.cat([concat_grid, prev_down], dim = 1)

			_, chan, height, width = concat_grid.size()
			concat_grid_reshaped = concat_grid.view(-1, seq_len, chan, height, width)

			level_ind = i - self.cur_start_level
			lstm_outputs, lstm_state = self.convlstms[level_ind](concat_grid_reshaped)
			lstm_outputs = lstm_outputs[0]
			lstm_state   = lstm_state[0]

			_, _, chan, height, width = lstm_outputs.size()
			lstm_outputs_reshaped = lstm_outputs.view(-1,chan,height,width)
			lstm_outputs_bn = self.batchnorms[level_ind](lstm_outputs_reshaped)

			output_grids.append(lstm_outputs_bn)
			lstm_states.append(lstm_state)
			#lstm_states_ph.append(initial_lstm_state)
		
		return  output_dims, output_grids, lstm_states
	
	def _check_in_range(self, i):
		return (i >= 0) and (i < self.num_input_grids)