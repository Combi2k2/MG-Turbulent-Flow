from .MG_model import MG
from .MGxTransformer_model import MGxTransformer

from .ConvFFN import ConvFFN
from .Unet import UNet
from .TF_Net.TFNet import TF_Net
from .Multigrid.convlstm import CLSTM

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    model_list = []