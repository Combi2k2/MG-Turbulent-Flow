import torch
from torch import nn
from torch.nn import functional as F

from MGconv import MGConvLayer
from MGMemory import MGMemLayer

class MG(nn.Module):
	def __init__(self, nb_input_chan):
		super(MG, self).__init__()
		self.nb_input_chan = nb_input_chan

		self.mem_layers = nn.ModuleList()
		self.gen_layers = nn.ModuleList()

		self.mem_layers.append(MGMemLayer(4, 6, 4, 6, nb_input_chan, 4, 1))
		self.mem_layers.append(MGMemLayer(4, 6, 4, 6, 4, 4, 2))
		self.mem_layers.append(MGMemLayer(4, 6, 4, 6, 4, 8, 3))
		self.mem_layers.append(MGMemLayer(4, 6, 4, 6, 8, 8, 4))
		self.mem_layers.append(MGMemLayer(4, 6, 4, 6, 8, 16, 5))

		self.gen_layers.append(MGConvLayer(4, 6, 4, 6, 16, 16, 1))
		self.gen_layers.append(MGConvLayer(4, 6, 4, 6, 16, 8, 2))
		self.gen_layers.append(MGConvLayer(4, 6, 4, 6, 8, 8, 3))
		self.gen_layers.append(MGConvLayer(4, 6, 5, 6, 8, 4, 4))
		self.gen_layers.append(MGConvLayer(5, 6, 6, 6, 4, 2, 5))

	def forward(self, inputs):
		batch, seq_len, chan, height, width = inputs.size()
		inputs = inputs.view(-1, chan, height, width)

		input3scales = [
			F.interpolate(inputs, size = (height // 4, height // 4), mode = 'nearest'),
			F.interpolate(inputs, size = (height // 2, width // 2), mode = 'nearest'),
			F.interpolate(inputs, size = (height, width), mode = 'nearest'),
		]
		output_grids_list = []
		prev_grids = input3scales

		# mem layers
		for i in range(5):
			output_dims, output_grids, lstm_states = self.mem_layers[i](prev_grids, seq_len)
			output_grids_list.append(output_grids)

			if i == 2 or i == 4:
				for scale in range(len(output_grids)):
					output_grids_list[-1][scale] = self.residual_conn(output_grids_list[-3][scale], output_grids_list[-1][scale])
				
			prev_grids = output_grids_list[-1]
		
		for i in range(len(prev_grids)):
			_, chan, width, height = prev_grids[i].size()

			prev_grids[i] = prev_grids[i].view(batch, seq_len, chan, height, width)
			prev_grids[i] = prev_grids[i][:,-1,:,:,:]

			if (i == 3):
				output_grids_list[-1][0] = self.residual_conn(output_grids_list[-3][1], output_grids_list[-1][0])
				output_grids_list[-1][1] = self.residual_conn(output_grids_list[-3][2], output_grids_list[-1][1])
		
		# generator layers
		for i in range(5):
			output_dims, output_grids = self.gen_layers[i](prev_grids)
			output_grids_list.append(output_grids)
			
			prev_grids = output_grids_list[-1]
		
		return  prev_grids[0]

	def residual_conn(self, x, y):
		_, chan_x, _, _ = x.size()
		_, chan_y, _, _ = y.size()

		if chan_x == chan_y:	return  x + y
		if chan_x <  chan_y:	return  F.pad(x,(0,0,0,0,0,chan_y - chan_x), "constant", 0) + y
		else:					return  x[:,:chan_y,:,:] + y

if __name__ == '__main__':
	model = MG(nb_input_chan = 2)

	inputs = torch.randn(16, 10, 2, 512, 512)
	output = model(inputs)

	print(output.shape)