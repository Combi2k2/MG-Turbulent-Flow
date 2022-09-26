import torch
from torch import nn
from torch.nn.functional import interpolate

class MGConvLayer(nn.Module):
	def __init__(self, prev_start_level, prev_end_level, cur_start_level, cur_end_level, input_feature_chan, output_feature_chan, lay_ind):
		super(MGConvLayer, self).__init__()
		self.nb_levels = cur_end_level - cur_start_level + 1
		self.kernel_size = (3,3)
		# Each level will be processed by a convlstm and followed by batchnorm.
		self.convs      = nn.ModuleList()
		self.batchnorms = nn.ModuleList()

		self.num_input_grids = prev_end_level - prev_start_level + 1

		self.prev_start_level = prev_start_level
		self.prev_end_level   = prev_end_level

		self.cur_start_level = cur_start_level
		self.cur_end_level   = cur_end_level

		self.output_feature_chan = output_feature_chan 
		self.lay_ind = lay_ind
		self.maxpool = nn.MaxPool2d(2, stride = 2)
		self.padding = self.kernel_size[0] // 2, self.kernel_size[1] // 2

		for i in range(cur_start_level, cur_end_level + 1):
			sum_input_chan = 0
			iprev = i - prev_start_level
			if self._check_in_range(iprev - 1): sum_input_chan += input_feature_chan
			if self._check_in_range(iprev):     sum_input_chan += input_feature_chan
			if self._check_in_range(iprev + 1): sum_input_chan += input_feature_chan

			self.convs.append(nn.Sequential(nn.Conv2d(in_channels  = sum_input_chan,
													  out_channels = output_feature_chan,
													  kernel_size  = self.kernel_size,
													  padding      = self.padding), nn.ReLU()))
		
			self.batchnorms.append(nn.BatchNorm2d(output_feature_chan))          

	def forward(self, prev_grids):
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
				prev_up = interpolate(prev_grids[iprev - 1], size = next_spatial_dim, mode = 'nearest')
				concat_grid = prev_up
			
			if self._check_in_range(iprev):
				if concat_grid is None:	concat_grid = prev_grids[iprev]
				else:					concat_grid = torch.cat([concat_grid, prev_grids[iprev]], dim = 1)

			if self._check_in_range(iprev + 1):
				prev_down = self.maxpool(prev_grids[iprev + 1]) #tf.layers.max_pooling2d(prev_grids[iprev+1],[2,2],2)
				if concat_grid is None:	concat_grid = prev_down
				else:					concat_grid = torch.cat([concat_grid, prev_down], dim = 1)
			
			level_ind = i - self.cur_start_level

			outputs = self.convs[level_ind](concat_grid)
			outputs = self.batchnorms[level_ind](outputs)

			output_grids.append(outputs)
		
		return output_dims, output_grids

	def _check_in_range(self, i):
		return (i >= 0) and (i < self.num_input_grids)

if __name__ == '__main__':
	model = MGConvLayer(2, 3, 2, 3, 1, 2, 0)

	inputs = [
		torch.randn(16, 1, 4, 4),
		torch.randn(16, 1, 8, 8)
	]
	_, output = model(inputs)

	print(output[0].shape)
	print(output[1].shape)