import torch.nn as nn

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, use_dropout):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        
        conv_block = []
        p = 0
        
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size = 3, padding = p),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Dropout(0.5) if use_dropout else nn.Identity()
        ]
        conv_block += conv_block[:-2]
        
        self.conv_block = nn.Sequential(*conv_block)
    
    def forward(self, x):
        return x + self.conv_block(x)

class ResnetEncoder(nn.Module):
    def __init__(self, input_nc, ngf = 64, out_dim = 528, n_downsampling = 2, use_dropout = False, padding_type = 'reflect'):
        """Construct a Resnet-based Encoder
        Parameters:
            input_nc (int)      -- the number of channels in input images
            ngf (int)           -- the number of filters in the last conv layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        
        super().__init__()
        
        moduleList = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size = 7, padding = 0),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        ]
        
        for i in range(n_downsampling):
            input_chan = ngf << i
            output_chan = input_chan * 2 if i < n_downsampling - 1 else out_dim
            
            moduleList += [
                nn.Conv2d(input_chan, output_chan, kernel_size = 3, stride = 2, padding = 1),
                nn.BatchNorm2d(output_chan),
                nn.ReLU(inplace = True)
            ]
        
        # resnet blocks
        moduleList += [ResnetBlock(out_dim, padding_type = padding_type, use_dropout = use_dropout) for _ in range(2)]
        moduleList += [nn.ReLU()]
        
        self.model = nn.Sequential(*moduleList)
    
    def forward(self, inputs):
        return self.model(inputs)

class ResnetDecoder(nn.Module):
    def __init__(self, output_nc, ngf = 64, feat_dim = 528, n_downsampling = 2, use_dropout = False, padding_type = 'reflect'):
        """Construct a Resnet-based Encoder
        Parameters:
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        super().__init__()
        
        moduleList = []
        
        for i in range(n_downsampling):
            input_chan = ngf << (n_downsampling - i) if i > 0 else feat_dim
            output_chan = ngf << (n_downsampling - i - 1)
            
            moduleList += [
                nn.ConvTranspose2d(input_chan, output_chan, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                nn.BatchNorm2d(output_chan),
                nn.ReLU(inplace = True)
            ]

        moduleList.append(nn.ReflectionPad2d(3))
        moduleList.append(nn.Conv2d(ngf, output_nc, kernel_size = 7, padding = 0))
        moduleList.append(nn.ReLU(inplace = True))

        self.model = nn.Sequential(*moduleList)
    
    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    enc = ResnetEncoder(input_nc = 32)
    dec = ResnetDecoder(output_nc = 31)
    
    import torch
    inputs = torch.randn(16, 32, 120, 120)
    
    out = enc(inputs)
    out = dec(out)
    
    print(out.shape)
            
         