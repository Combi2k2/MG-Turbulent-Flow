from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class rbc_data(Dataset):
    def __init__(self, data, indices, input_length, output_length, stack_x):
        self.data = data
        self.input_length = input_length
        self.output_length = output_length
        self.list_IDs = indices
        self.stack_x = stack_x
    
    def __len__(self):
        return  len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        
        for flow in self.data:
            flow_length = flow.size(0) - self.input_length - self.output_length
            
            if (index < flow_length):
                inputs = flow[index: index + self.input_length]
                target = flow[index + self.input_length: index + self.input_length + self.output_length]
                
                if (self.stack_x):
                    inputs = inputs.reshape(-1, target.shape[-2], target.shape[-1])
                
                return  inputs, target
            
            index -= flow_length
            
        raise ValueError('Index out of range.')
    
class randomFlowData(Dataset):
    def __init__(self, data, indices, input_length, output_length, stack_x):
        self.data = data
        self.input_length = input_length
        self.output_length = output_length
        self.list_IDs = indices
    
    def __len__(self):
        return  len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        
        for flow in self.data:
            print(index, flow.shape[0])
            if flow.shape[0] > index:
                return flow[index][:self.input_length], flow[index][self.input_length:self.input_length + self.output_length]

            index -= flow.shape[0]
            
        raise ValueError('Index out of range.')

if __name__ == '__main__':
    import torch
    
    data_prep = [torch.load(r'dataset\2D\CFD\2D_Train_Rand\2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train\randflow_0.pt'), 
                 torch.load(r'dataset\2D\CFD\2D_Train_Rand\2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train\randflow_1.pt')]

    dataset = randomFlowData(data_prep, range(2000), 16, 4, False)

    input, target = dataset[0]
    print(input.shape, target.shape)

    input, target = dataset[1999]
    print(input.shape, target.shape)

