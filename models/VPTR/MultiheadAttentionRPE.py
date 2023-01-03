# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Combi2k2 from:
#   https://github.com/XiYe20/VPTR/blob/main/model/MultiHeadAttentionRPE.py
# --------------------------------------------------------

import torch
import torch.nn.functional as F

from torch.nn.functional import linear, pad, softmax, dropout
from torch.overrides import has_torch_function, handle_torch_function

from torch import nn, Tensor
from typing import Tuple, Optional

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout = 0.0, bias = True, add_zero_attn = False):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        
        assert (self.head_dim * num_heads == self.embed_dim), "embed_dim must be divisible by num_heads"

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias = bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias = bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias = bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.add_zero_attn = add_zero_attn

    def forward(self, query : Tensor, key : Tensor, value : Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        residual_attn: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        tgt_len, bsz, embed_dim = query.size()
        out_dim = embed_dim
        
        if (key is None):   key = query
        if (value is None): value = query

        assert embed_dim == self.embed_dim
        # allow MHA to have different sizes for the feature dimension
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        scaling = float(self.head_dim) ** -0.5

        q = self.q_proj(query) * scaling
        k = self.k_proj(key)
        v = self.v_proj(value)

        if attn_mask is not None: # checking the format of attention mask
            assert (attn_mask.dtype == torch.float32
                or attn_mask.dtype == torch.float64
                or attn_mask.dtype == torch.float16
                or attn_mask.dtype == torch.uint8
                or attn_mask.dtype == torch.bool
            ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(attn_mask.dtype)
            if attn_mask.dtype == torch.uint8:
                attn_mask = attn_mask.to(torch.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError("The size of the 2D attn_mask is not correct.")
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [bsz * self.num_heads, query.size(0), key.size(0)]:
                    raise RuntimeError("The size of the 3D attn_mask is not correct.")
            else:
                raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))

        # convert ByteTensor key_padding_mask to bool
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            key_padding_mask = key_padding_mask.to(torch.bool)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1) if k is not None else None
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1) if v is not None else None

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype = k.dtype, device = k.device)], dim = 1)
            v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype = v.dtype, device = v.device)], dim = 1)
            
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        """
        Attention weight for the invalid region is -inf
        """
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len).masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2),float("-inf"))
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if residual_attn is not None:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len) + residual_attn.unsqueeze(0)
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)

        """
        Reweight the attention map before softmax().
        attn_output_weights: (b*n_head, n, hw)
        """
        attn_output_weights = softmax(attn_output_weights, dim = -1)
        attn_output_weights = dropout(attn_output_weights, p = self.dropout, training = self.training)

        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, out_dim)
        attn_output = linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        if need_weights:
            return attn_output, attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len).sum(dim = 1) / self.num_heads
        else:
            return attn_output

class MultiheadAttentionRPE(MultiheadAttention):
    """ "Multihead Attention with extra flags on the q/k/v and out projections."""
    def __init__(self, *args, rpe = False, window_size = 7, **kwargs):
        super(MultiheadAttentionRPE, self).__init__(*args, **kwargs)
        self.rpe = rpe
        if rpe:
            self.window_size = [window_size] * 2
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                    self.num_heads,
                )
            )  # 2*Wh-1 * 2*Ww-1, nH
            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing = 'ij'))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)
            nn.init.trunc_normal_(self.relative_position_bias_table, std = 0.02)

    def forward(self, query : Tensor, key : Tensor, value : Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        do_qkv_proj: bool = True,
        do_out_proj: bool = True,
        rpe = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        tgt_len, bsz, embed_dim = query.size()
        out_dim = embed_dim
        if (key is None):   key = query
        if (value is None): value = query

        assert embed_dim == self.embed_dim
        # allow MHA to have different sizes for the feature dimension
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        scaling = float(self.head_dim) ** -0.5

        # whether or not use the original query/key/value
        q = self.q_proj(query) * scaling if do_qkv_proj else query
        k = self.k_proj(key) if do_qkv_proj else key
        v = self.v_proj(value) if do_qkv_proj else value

        if attn_mask is not None:
            assert (attn_mask.dtype == torch.float32
                or attn_mask.dtype == torch.float64
                or attn_mask.dtype == torch.float16
                or attn_mask.dtype == torch.uint8
                or attn_mask.dtype == torch.bool
            ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(attn_mask.dtype)
            if attn_mask.dtype == torch.uint8:
                attn_mask = attn_mask.to(torch.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError("The size of the 2D attn_mask is not correct.")
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [bsz * self.num_heads, query.size(0), key.size(0)]:
                    raise RuntimeError("The size of the 3D attn_mask is not correct.")
            else:
                raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))

        # convert ByteTensor key_padding_mask to bool
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            key_padding_mask = key_padding_mask.to(torch.bool)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1) if k is not None else None
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1) if v is not None else None

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype = k.dtype, device = k.device)], dim = 1)
            v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype = v.dtype, device = v.device)], dim = 1)
            
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
                

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        
        """
        Add relative position embedding
        """
        if self.rpe and rpe:
            # NOTE: for simplicity, we assume the src_len == tgt_len == window_size**2 here
            assert (src_len == self.window_size[0] * self.window_size[1]
                and tgt_len == self.window_size[0] * self.window_size[1]
            ), f"src{src_len}, tgt{tgt_len}, window{self.window_size[0]}"
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len) + relative_position_bias.unsqueeze(0)
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)

        """
        Attention weight for the invalid region is -inf
        """
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len).masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)

        """
        Reweight the attention map before softmax().
        attn_output_weights: (b*n_head, n, hw)
        """
        attn_output_weights = softmax(attn_output_weights, dim=-1)
        attn_output_weights = dropout(attn_output_weights, p = self.dropout, training = self.training)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn_output = (attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, out_dim))
        if do_out_proj:
            attn_output = linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            return attn_output, q, k, attn_output_weights.sum(dim=1) / self.num_heads
        else:
            return attn_output, q, k  # additionaly return the query and key

if __name__ == '__main__':
    model = MultiheadAttentionRPE(20, 4)
    query = torch.randn(7, 16, 20)
    
    output, q, k = model(query, None, None)
    print(output.shape)
    print(q.shape)
    print(k.shape)