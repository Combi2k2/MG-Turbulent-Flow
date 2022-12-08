from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class rbc_data(Dataset):
    def __init__(self, data_prep, split = 'train', window_size = 16):
        self.data = []
        self.len = []
        self.window_size = window_size

        for flow in data_prep:
            flow_len = flow.size(0)
            mock = int(flow_len * 0.1)

            if split == 'train':    self.data.append(flow[mock:])
            if split == 'valid':    self.data.append(flow[:mock])

            self.len.append(self.data[-1].size(0) - window_size - 1)
    
    def __len__(self):
        return  sum(self.len)
    
    def __getitem__(self, index):
        for i, flow in enumerate(self.data):
            if (index < self.len[i]):
                inputs = flow[index : index + self.window_size]
                target = flow[index + self.window_size + 1]

                return  inputs, target
            
            index -= self.len[i]
            
        raise ValueError('Index out of range.')

if __name__ == '__main__':
    import torch
    
    data_prep = [torch.load('data/sample_0.pt')]
    sample_dt = rbc_data(data_prep, 'train')
    
    inputs, target = sample_dt[5]
    
    print(inputs.shape)
    print(target.shape)