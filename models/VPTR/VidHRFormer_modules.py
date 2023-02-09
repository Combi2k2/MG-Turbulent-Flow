import torch
import torch.nn as nn
import copy

from .MultiheadAttentionRPE import MultiheadAttentionRPE
from .VidHRFormer_utils import DropPath, PadBlock, LocalPermuteModule

"""
Encoder, same for auto-regressive and non-autoregressive models
"""
class VidHRFormerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, local_window_pos_embed, temporal_pos_embed):
        output = src
        for layer in self.layers:
            output = layer(output, local_window_pos_embed, temporal_pos_embed)
        if self.norm is not None:
            output = self.norm(output)

        return output

class VidHRFormerBlockEnc(nn.Module):
    def __init__(self, encH, encW, embed_dim, num_heads, window_size = 7, dropout = 0., drop_path = 0., Spatial_FFN_hidden_ratio = 4, dim_feedforward = 1024, far = True, rpe = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.dropout = dropout
        self.Spatial_FFN_hidden_ratio = Spatial_FFN_hidden_ratio

        self.SLMHSA = SpatialLocalMultiheadAttention(embed_dim, num_heads, window_size, dropout, rpe)
        self.FAR = far
        
        self.SpatialFFN = ConvFFN(encH, encW, embed_dim, hidden_features = int(Spatial_FFN_hidden_ratio * embed_dim), out_features = embed_dim, drop = dropout, AR_model = far)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm3 = nn.LayerNorm(embed_dim)
        self.temporal_MHSA = nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout = dropout)
        
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        
        self.activation = nn.GELU()
        self.drop1 = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.drop2 = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.drop3 = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.norm4 = nn.LayerNorm(embed_dim)

    def forward(self, x, local_window_pos_embed, temporal_pos_embed):
        """
        x: (N, T, C, H, W)
        local_window_pos_embed: (window_size, window_size, C)
        temporal_pos_embed: (T, C)
        Return: (N, T, C, H, W)
        """
        N, T, C, H, W = x.shape
        x = x.permute(0, 1, 3, 4, 2)
        x = x + self.drop_path(self.SLMHSA(self.norm1(x), local_window_pos_embed)) #spatial local window self-attention, and skip connection
        
        #Conv feed-forward, different local window information interacts
        x = x + self.drop_path(self.SpatialFFN(self.norm2(x)))#(N, T, H, W, C)

        #temporal attention
        x = x.permute(1, 0, 2, 3, 4).reshape(T, N * H * W, C)
        x1 = self.norm3(x)
        
        if self.FAR:    attn_mask = torch.triu(torch.ones(T, T), diagonal = 1) == 1
        else:           attn_mask = None
        
        x = x + self.drop1(self.temporal_MHSA(x1 + temporal_pos_embed[:, None, :], 
                                              x1 + temporal_pos_embed[:, None, :], 
                                              x1, 
                                              attn_mask = attn_mask.to(x1.device))[0])
        
        x1 = self.norm4(x)
        x1 = self.linear2(self.drop2(self.activation(self.linear1(x1))))
        x = x + self.drop3(x1)

        x = x.reshape(T, N, H, W, C).permute(1, 0, 4, 2, 3)

        return x

####################################End of Transformer Encoder modules#################################

class SpatialLocalMultiheadAttention(nn.Module):
    """
    Modified based on https://github.com/HRNet/HRFormer/blob/main/cls/models/modules/multihead_isa_attention.py
    local spatial window attention with absolute positional encoding, i.e. based the standard nn.MultiheadAttention module
    Args:
        embed_dim (int): Number of input channels.
        window_size (tuple[int]): Window size.
        num_heads (int): Number of attention heads.
    """

    def __init__(self, embed_dim, num_heads, window_size = 7, dropout = 0., rpe = False):
        super().__init__()

        self.dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.dropout = dropout

        self.rpe = rpe
        
        if rpe: self.attn = MultiheadAttentionRPE(embed_dim, num_heads, dropout = dropout, rpe = True, window_size = window_size)
        else:   self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout = dropout)
        
        self.pad_helper = PadBlock(window_size)
        self.permute_helper = LocalPermuteModule(window_size)

    def forward(self, x, local_pos_embed, value = None):
        """
        x: (N, T, H, W, C)
        value: value should be None for encoder self-attention, value is not None for the Transformer decoder self-attention
        local_pos_embed: (window_size, window_size, C)
        return:
           (N, T, H, W, C)
        """
        N, T, H, W, C = x.shape
        x = x.view(N * T, H, W, C)

        # attention
        # pad
        x_pad = self.pad_helper.pad_if_needed(x, x.size())
        # permute
        x_permute = self.permute_helper.permute(x_pad, x_pad.size()) #(window_size*window_size, N*T*H/window_size*W/window_size, C)
        
        if self.rpe:    k = q = x_permute
        else:           k = q = x_permute + local_pos_embed.flatten(0, 1)[:, None, :]

        if value is not None:
            value = value.view(N * T, H, W, C)
            value_pad = self.pad_helper.pad_if_needed(value, value.size())
            value_permute = self.permute_helper.permute(value_pad, value_pad.size()) #(window_size*window_size, N*T*H/window_size*W/window_size, C)
            # attention
            out = self.attn(q, k, value = value_permute)[0]
        else:
            out = self.attn(q, k, value = x_permute)[0]
        # reverse permutation
        out = self.permute_helper.rev_permute(out, x_pad.size()) #(N*T, H, W, C)

        # de-pad, pooling with `ceil_mode=True` will do implicit padding, so we need to remove it, too
        out = self.pad_helper.depad_if_needed(out, x.size())

        return out.view(N, T, H, W, C)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"

    def flops(self, NT):
        # calculate flops for 1 window with token length of NT
        flops = 0   # qkv = self.qkv(x)
        flops += NT * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * NT * (self.dim // self.num_heads) * NT
        #  x = (attn @ v)
        flops += self.num_heads * NT * NT * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += NT * self.dim * self.dim
        return flops

class ConvFFN(nn.Module):
    """
    https://github.com/HRNet/HRFormer/blob/main/cls/models/modules/ffn_block.py
    """
    def __init__(self, encH, encW, in_features, hidden_features = None, out_features = None, act_layer = nn.GELU, dw_act_layer = nn.GELU, drop = 0.0, AR_model = True):
        super().__init__()
        
        if (out_features is None):      out_features = in_features
        if (hidden_features is None):   hidden_features = in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act1 = act_layer()
        
        if AR_model:    self.norm1 = nn.LayerNorm((hidden_features, encH, encW))
        else:           self.norm1 = nn.BatchNorm2d(hidden_features)
        
        self.dw3x3 = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            groups=hidden_features,
            padding=1,
        )
        self.act2 = dw_act_layer()
        if AR_model:    self.norm2 = nn.LayerNorm((hidden_features, encH, encW))
        else:           self.norm2 = nn.BatchNorm2d(hidden_features)
        
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.act3 = act_layer()
        if AR_model:    self.norm3 = nn.LayerNorm((out_features, encH, encW))
        else:           self.norm3 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)

        self.out_features = out_features

    def forward(self, x):
        """
        x: (N, T, H, W, C)
        """
        N, T, H, W, C = x.shape
        x = x.view(N * T, H, W, C).permute(0, 3, 1, 2)
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.dw3x3(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.norm3(x)
        x = self.act3(x)
        x = self.drop(x)
        
        return x.permute(0, 2, 3, 1).reshape(N, T, H, W, self.out_features)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

if __name__ == '__main__':
    from position_encoding import PositionEmbeddding1D
    from position_encoding import PositionEmbeddding2D
    
    temp_model = VidHRFormerBlockEnc(48, 48, 20, 4)
    
    pos1d = PositionEmbeddding1D()
    temporal_pos = pos1d(L = 13, N = 1, E = 20)[:, 0, :]
    
    pos2d = PositionEmbeddding2D()
    lw_pos = pos2d(N = 1, E = 20, H = 48, W = 48)[0, ...].permute(1, 2, 0)
    
    input = torch.randn(16, 13, 20, 48, 48)
    output = temp_model(input, lw_pos, temporal_pos)
    
    print(output.shape)
