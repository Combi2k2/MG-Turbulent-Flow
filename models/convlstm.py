from .Multigrid.convlstm import ConvLSTM
from .Multigrid.convlstm import CLSTM

if __name__ == '__main__':
    import torch
    
    model = CLSTM()
    
    inputs = torch.rand((32, 17, 2, 64, 64))
    output = model(inputs)
    print(output.shape) # 2 x 64 x 64