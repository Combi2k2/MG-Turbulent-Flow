from .VidHRFormer_modules import VidHRFormerBlockEnc, VidHRFormerEncoder
from .position_encoding import PositionEmbeddding1D, PositionEmbeddding2D
from .ResnetAutoEncoder import ResnetEncoder, ResnetDecoder

import torch.nn as nn
import torch.nn.functional as F

class VPTREnc(nn.Module):
    def __init__(self, img_channels, feat_dim = 528, n_downsampling = 3, padding_type = 'reflect'):
        super().__init__()
        self.feat_dim = feat_dim
        self.encoder = ResnetEncoder(input_nc = img_channels, out_dim = feat_dim, n_downsampling = n_downsampling, padding_type = padding_type)
        
    def forward(self, x):
        """
        Args:
            x --- (N, T, img_channels, H, W)
        Returns:
            feat --- (N, T, 256, 16, 16)
        """
        N, T, _, _, _ = x.shape
        feat = self.encoder(x.flatten(0, 1))
        #feat = self.out_proj(feat)
        _, C, H, W = feat.shape
        feat = feat.reshape(N, T, C, H, W)

        return feat

class VPTRDec(nn.Module):
    def __init__(self, img_channels, feat_dim = 528, n_downsampling = 3, padding_type = 'reflect'):
        super().__init__()
        self.decoder = ResnetDecoder(output_nc = img_channels, feat_dim = feat_dim, n_downsampling = n_downsampling, padding_type = padding_type)

    def forward(self, feat):
        """
        Args:
            feat --- (N, T, 256, 16, 16)
        """
        N, T, _, _, _ = feat.shape


        out = self.decoder(feat.flatten(0, 1))
        _, C, H, W = out.shape

        return out.reshape(N, T, C, H, W)

class VPTRDisc(nn.Module):
    """
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    Defines a PatchGAN discriminator
    """
    def __init__(self, input_nc, ndf = 64, n_layers = 3):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride = 2, padding = padw),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size = kw, stride = 1, padding=padw),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class VPTRFormerFAR(nn.Module):
    def __init__(self, num_frames, encH = 8, encW = 8, d_model = 528, 
                 nhead = 8, num_encoder_layers = 6, dropout = 0.1, 
                 window_size = 4, Spatial_FFN_hidden_ratio = 4,rpe = True):
        super().__init__()
        
        self.nhead = nhead
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers

        self.dropout = dropout
        self.window_size = window_size
        self.Spatial_FFN_hidden_ratio = Spatial_FFN_hidden_ratio 

        self.transformer = VidHRFormerEncoder(VidHRFormerBlockEnc(encH, encW, d_model, nhead, window_size, dropout, drop_path = dropout, Spatial_FFN_hidden_ratio = Spatial_FFN_hidden_ratio, dim_feedforward = d_model * Spatial_FFN_hidden_ratio, far = True, rpe = rpe), 
                                        num_encoder_layers, nn.LayerNorm(d_model))
        
        #Init all the pos_embed
        """
        local_window_pos_embed: (window_size, window_size, embed_dim)
        temporal_pos_embed: (num_past_frames, embed_dim)
        """
        pos1d = PositionEmbeddding1D()
        pos2d = PositionEmbeddding2D()
        
        temporal_pos = pos1d(L = num_frames, N = 1, E = d_model)[:, 0, :]
        spatial_pos = pos2d(N = 1, E = d_model, H = window_size, W = window_size)[0, ...].permute(1, 2, 0)
        
        self.register_buffer('temporal_pos', temporal_pos)
        self.register_buffer('spatial_pos', spatial_pos)

        self._reset_parameters()
        
    def forward(self, input_feats):
        """
        Args:
            past_gt_feats:  (N, T, 528, 8, 8)
            future_frames: (N, T, 528, 8, 8) or None
        Return:
            out: (N, T, H, W, embed_dim), for the next layer query_pos init
        """
        out = input_feats.permute(0, 1, 3, 4, 2)
        out = self.transformer(out, self.spatial_pos, self.temporal_pos)
        out = F.relu_(out.permute(0, 1, 4, 2, 3))
        
        return out
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

if __name__ == '__main__':
    model = VPTRFormerFAR(16)
    
    import torch
    
    inputs = torch.rand(13, 16, 528, 8, 8)
    output = model(inputs)
    
    print(output.shape)

