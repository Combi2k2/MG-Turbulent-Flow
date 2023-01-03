# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Combi2k2 from:
#   https://github.com/XiYe20/VPTR/blob/main/utils/position_encoding.py
# --------------------------------------------------------


import torch
from torch import nn
import math
import numpy as np

"""
1D position encoding and 2D postion encoding
The code is modified based on DETR of Facebook: 
https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
"""

embedding_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class PositionEmbeddding1D(nn.Module):
    """
    1D position encoding
    Based on Attetion is all you need paper and DETR PositionEmbeddingSine class
    """
    def __init__(self, temperature = 10000, normalize = False, scale = None):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, L: int, N: int, E: int):
        """
        Args:
            L for length, N for batch size, E for embedding size (dimension of transformer).
        Returns:
            pos: position encoding, with shape [L, N, E]
        """
        pos_embed = torch.ones(N, L, dtype = torch.float32).cumsum(axis = 1)
        dim_t = torch.arange(E, dtype = torch.float32)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode = 'floor') / E)
        if self.normalize:
            eps = 1e-6
            pos_embed = pos_embed / (L + eps) * self.scale

        pos_embed = pos_embed[:, :, None] / dim_t
        pos_embed = torch.stack((pos_embed[:, :, 0::2].sin(), pos_embed[:, :, 1::2].cos()), dim = 3).flatten(2)
        pos_embed = pos_embed.permute(1, 0, 2)
        pos_embed.requires_grad_(False)
        
        return pos_embed

class PositionEmbeddding2D(nn.Module):
    """
    2D position encoding, borrowed from DETR PositionEmbeddingSine class
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """
    def __init__(self, temperature=10000, normalize=False, scale=None, device = torch.device(embedding_device)):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        self.device = device
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, N: int, E: int, H: int, W: int):
        """
        Args:
            N for batch size, E for embedding size (channel of feature), H for height, W for width
        Returns:
            pos_embed: positional encoding with shape (N, E, H, W)
        """
        assert E % 2 == 0, "Embedding size should be even number"

        y_embed = torch.ones(N, H, W, dtype = torch.float32, device = self.device).cumsum(dim = 1)
        x_embed = torch.ones(N, H, W, dtype = torch.float32, device = self.device).cumsum(dim = 2)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(E//2, dtype=torch.float32, device=self.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode = 'floor') / (E//2))

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_embed = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos_embed.requires_grad_(False)
        return pos_embed