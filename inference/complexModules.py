import torch.nn as nn
import torch
from complexLayers import ComplexLinear,ComplexReLU,ComplexConv1d,ComplexConvTranspose1d,ComplexConv2d
import numpy as np

def set_skip_module1(args):
    """
    Create a frequency-representation module
    """
    net = None

    if args.fr_module_type == 'fr':
        assert args.fr_size == args.fr_inner_dim * args.fr_upsampling, \
            'The desired size of the frequency representation (fr_size) must be equal to inner_dim*upsampling'
        net = FrequencyRepresentationModule_skiplayer32_1(signal_dim=args.signal_dim, n_filters=args.fr_n_filters,
                                            inner_dim=args.fr_inner_dim, n_layers=args.fr_n_layers,
                                            upsampling=args.fr_upsampling, kernel_size=args.fr_kernel_size,
                                            kernel_out=args.fr_kernel_out)
    else:
        raise NotImplementedError('Frequency representation module type not implemented')
    if args.use_cuda:
        net.cuda()

    return net

def set_skip_module2(args):
    """
    Create a frequency-representation module
    """
    net = None

    if args.fr_module_type == 'fr':
        assert args.fr_size == args.fr_inner_dim * args.fr_upsampling, \
            'The desired size of the frequency representation (fr_size) must be equal to inner_dim*upsampling'
        net = FrequencyRepresentationModule_skiplayer32_2(signal_dim=args.signal_dim, n_filters=args.fr_n_filters,
                                            inner_dim=args.fr_inner_dim, n_layers=args.fr_n_layers,
                                            upsampling=args.fr_upsampling, kernel_size=args.fr_kernel_size,
                                            kernel_out=args.fr_kernel_out)
    else:
        raise NotImplementedError('Frequency representation module type not implemented')
    if args.use_cuda:
        net.cuda()

    return net

def set_skip_module3(args):
    """
    Create a frequency-representation module
    """
    net = None

    if args.fr_module_type == 'fr':
        assert args.fr_size == args.fr_inner_dim * args.fr_upsampling, \
            'The desired size of the frequency representation (fr_size) must be equal to inner_dim*upsampling'
        net = FrequencyRepresentationModule_skiplayer32_3(signal_dim=args.signal_dim, n_filters=args.fr_n_filters,
                                            inner_dim=args.fr_inner_dim, n_layers=args.fr_n_layers,
                                            upsampling=args.fr_upsampling, kernel_size=args.fr_kernel_size,
                                            kernel_out=args.fr_kernel_out)
    else:
        raise NotImplementedError('Frequency representation module type not implemented')
    if args.use_cuda:
        net.cuda()

    return net

class FrequencyRepresentationModule_skiplayer32_1(nn.Module):
    def __init__(self, signal_dim=50, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=8, kernel_out=25):
        super().__init__()
        self.fr_size = inner_dim * upsampling

        self.n_filters = n_filters


        self.inner=inner_dim
        self.n_layers=n_layers

        self.in_layer = ComplexLinear(signal_dim, inner_dim * int(n_filters / 8))

        self.in_layer2 = ComplexConv2d(1,  int(n_filters/4), kernel_size=(1, 3), padding=(0, 3 // 2),
                                       bias=False)
        mod=[]
        for i in range(self.n_layers):
            tmp = []
            tmp += [
                nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size //2, bias=False,
                          padding_mode='circular'),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
                nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size //2, bias=False,
                          padding_mode='circular'),
                nn.BatchNorm1d(n_filters),
            ]
            mod+= [nn.Sequential(*tmp)]
        self.mod=nn.Sequential(*mod)
        activate_layer = []
        for i in range(self.n_layers):
            activate_layer+=[nn.ReLU()]
        self.activate_layer=nn.Sequential(*activate_layer)

        self.out_layer = nn.ConvTranspose1d(n_filters, 1, 3, stride=upsampling,
                                            padding=1, output_padding=0, bias=False)

    def forward(self, x):
        bsz = x.size(0)
        inp = x[:,0,:].type(torch.complex64)+1j*x[:,1,:].type(torch.complex64)

        x=self.in_layer(inp).view(bsz, 1,int(self.n_filters/8), -1)
        x=self.in_layer2(x).view(bsz,self.n_filters,-1)
        x=x.abs()

        for i in range(self.n_layers):
            res_x = self.mod[i](x)
            x = res_x + x
            x = self.activate_layer[i](x)

        x = self.out_layer(x).view(bsz, -1)

        return x

class FrequencyRepresentationModule_skiplayer32_2(nn.Module):
    def __init__(self, signal_dim=50, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=8, kernel_out=25):
        super().__init__()
        self.fr_size = inner_dim * upsampling

        self.n_filters = n_filters


        self.inner=inner_dim
        self.n_layers=n_layers

        self.in_layer = ComplexLinear(signal_dim, inner_dim * int(n_filters / 8))

        self.in_layer2 = ComplexConv2d(1,  int(n_filters/4), kernel_size=(1, 3), padding=(0, 3 // 2),
                                       bias=False)
        mod=[]
        for i in range(self.n_layers):
            tmp = []
            tmp += [
                nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size //2, bias=False,
                          padding_mode='circular'),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
                nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size //2, bias=False,
                          padding_mode='circular'),
                nn.BatchNorm1d(n_filters),
            ]
            mod+= [nn.Sequential(*tmp)]
        self.mod=nn.Sequential(*mod)
        activate_layer = []
        for i in range(self.n_layers):
            activate_layer+=[nn.ReLU()]
        self.activate_layer=nn.Sequential(*activate_layer)

        self.out_layer = nn.ConvTranspose1d(n_filters, 1, 6, stride=upsampling,
                                            padding=1, output_padding=0, bias=False)

    def forward(self, x):
        bsz = x.size(0)
        inp = x[:,0,:].type(torch.complex64)+1j*x[:,1,:].type(torch.complex64)

        x=self.in_layer(inp).view(bsz, 1,int(self.n_filters/8), -1)
        x=self.in_layer2(x).view(bsz,self.n_filters,-1)
        x=x.abs()

        for i in range(self.n_layers):
            res_x = self.mod[i](x)
            x = res_x + x
            x = self.activate_layer[i](x)

        x = self.out_layer(x).view(bsz, -1)

        return x

class FrequencyRepresentationModule_skiplayer32_3(nn.Module):
    def __init__(self, signal_dim=50, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=8, kernel_out=25):
        super().__init__()
        self.fr_size = inner_dim * upsampling

        self.n_filters = n_filters


        self.inner=inner_dim
        self.n_layers=n_layers

        self.in_layer = ComplexLinear(signal_dim, inner_dim * int(n_filters / 8))

        self.in_layer2 = ComplexConv2d(1,  int(n_filters/4), kernel_size=(1, 3), padding=(0, 3 // 2),
                                       bias=False)
        mod=[]
        for i in range(self.n_layers):
            tmp = []
            tmp += [
                nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size //2, bias=False,
                          padding_mode='circular'),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
                nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size //2, bias=False,
                          padding_mode='circular'),
                nn.BatchNorm1d(n_filters),
            ]
            mod+= [nn.Sequential(*tmp)]
        self.mod=nn.Sequential(*mod)
        activate_layer = []
        for i in range(self.n_layers):
            activate_layer+=[nn.ReLU()]
        self.activate_layer=nn.Sequential(*activate_layer)

        self.out_layer = nn.ConvTranspose1d(n_filters, 1, 18, stride=16,
                                            padding=1, output_padding=0, bias=False)

    def forward(self, x):
        bsz = x.size(0)
        inp = x[:,0,:].type(torch.complex64)+1j*x[:,1,:].type(torch.complex64)

        x=self.in_layer(inp).view(bsz, 1,int(self.n_filters/8), -1)
        x=self.in_layer2(x).view(bsz,self.n_filters,-1)
        x=x.abs()

        for i in range(self.n_layers):
            res_x = self.mod[i](x)
            x = res_x + x
            x = self.activate_layer[i](x)

        x = self.out_layer(x).view(bsz, -1)

        return x






