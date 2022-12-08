from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class rbc_data(Dataset):
    def __init__(self, data, indices, input_length, output_length):
        self.data = data
        self.input_length = input_length
        self.output_length = output_length
        self.list_IDs = indices
    
    def __len__(self):
        return  len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        
        for flow in self.data:
            flow_length = flow.size(0) - self.input_length - self.output_length
            
            if (index < flow_length):
                inputs = flow[index: index + self.input_length]
                target = flow[index + self.input_length: index + self.input_length + self.output_length]
                
                return  inputs, target
            
            index -= flow_length
            
        raise ValueError('Index out of range.')

if __name__ == '__main__':
    import torch
    
    data_prep = [torch.load('data/sample_0.pt')]
    sample_dt = rbc_data(data_prep, list(range(1000)), 16, 4)
    
    inputs, target = sample_dt[5]
    
    print(inputs.shape)
    print(target.shape)