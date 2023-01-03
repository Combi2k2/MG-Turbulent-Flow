import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import math

from einops import rearrange

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """https://github.com/rwightman/pytorch-image-models/blob/07379c6d5dbb809b3f255966295a4b03f23af843/timm/models/layers/drop.py#L155
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """https://github.com/rwightman/pytorch-image-models/blob/07379c6d5dbb809b3f255966295a4b03f23af843/timm/models/layers/drop.py#L155
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob = None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return "drop_prob={}".format(self.drop_prob)

class PadBlock(object):
    """ "Make the size of feature map divisible by local group size."""
    """
    https://github.com/HRNet/HRFormer/blob/main/cls/models/modules/multihead_isa_attention.py
    """
    def __init__(self, local_group_size: int = 7):
        self.lgs = local_group_size

    def pad_if_needed(self, x, size):
        """ x: (N, H, W, C)
        """
        _, H, W, _ = size
        pad_h = math.ceil(H / self.lgs) * self.lgs - H
        pad_w = math.ceil(W / self.lgs) * self.lgs - W
        
        if pad_h > 0 or pad_w > 0:  # center-pad the feature on H and W axes
            return F.pad(x, (0, 0, pad_w // 2, pad_w - pad_w // 2,
                                   pad_h // 2, pad_h - pad_h // 2))
        
        return x

    def depad_if_needed(self, x, size):
        """ x: (N, H, W, C)
        """
        _, H, W, _ = size
        pad_h = math.ceil(H / self.lgs) * self.lgs - H
        pad_w = math.ceil(W / self.lgs) * self.lgs - W
        
        if pad_h > 0 or pad_w > 0:  # remove the center-padding on feature
            return x[:, pad_h // 2 : pad_h // 2 + H,
                        pad_w // 2 : pad_w // 2 + W, :]
        
        return x

class LocalPermuteModule(object):
    """ Permute the feature map to gather pixels in local groups, and the reverse permutations
    https://github.com/HRNet/HRFormer/blob/main/cls/models/modules/multihead_isa_attention.py
    """
    def __init__(self, local_group_size: int = 7):
        self.lgs = local_group_size
        # assert len(self.lgs) == 2

    def permute(self, x, size):
        """ x: (N, H, W, C)
            return: (local_group_size*local_group_size, N*H/local_group_size*W/local_group_size, C)
        """
        N, H, W, C = size
        return rearrange(x, "n (qh ph) (qw pw) c -> (ph pw) (n qh qw) c",   
            n = N, c = C,
            qh = H // self.lgs, ph = self.lgs,
            qw = W // self.lgs, pw = self.lgs,
        )

    def rev_permute(self, x, size):
        N, H, W, C = size
        return rearrange(x, "(ph pw) (n qh qw) c -> n (qh ph) (qw pw) c",
            n = N, c = C,
            qh = H // self.lgs, ph = self.lgs,
            qw = W // self.lgs, pw = self.lgs,
        )

class TemporalLocalPermuteModule(object):
    """ Permute the feature map to gather pixels in spatial local groups, and the reverse permutation
    https://github.com/HRNet/HRFormer/blob/main/cls/models/modules/multihead_isa_attention.py
    """
    def __init__(self, local_group_size: int = 7):
        self.lgs = local_group_size

    def permute(self, x, size):
        """ x: (N, T, H, W, C)
            return: (T * local_group_size * local_group_size, N * H * W / local_group_size^2, C)
        """
        N, T, H, W, C = size
        
        return rearrange(x, "n t (qh ph) (qw pw) c -> (t ph pw) (n qh qw) c",
            n = N, t = T, c = C,
            qh = H // self.lgs, ph = self.lgs,
            qw = W // self.lgs, pw = self.lgs,
        )

    def rev_permute(self, x, size):
        N, T, H, W, C = size
        return rearrange(x, "(t ph pw) (n qh qw) c -> n t (qh ph) (qw pw) c",
            n = N, t = T, c = C,
            qh = H // self.lgs, ph = self.lgs,
            qw = W // self.lgs, pw = self.lgs,
        )

if __name__ == '__main__':
    import torch
    img = torch.randn(16, 13, 64, 64)
    
    LPM = LocalPermuteModule()
    pad = PadBlock()
    
    img_pad = pad.pad_if_needed(img, img.size())
    img_permute = LPM.permute(img_pad, img_pad.size())
    # img = spatial.permute(img, img.shape)
    
    print(img_pad.shape)
    print(img_permute.shape)
    