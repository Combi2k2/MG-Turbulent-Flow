from MG_model import MG
from ConvFFN import ConvFFN
from convlstm import CLSTM
from Unet import UNet
from TF_Net.TFNet import TF_Net

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    model_list = []