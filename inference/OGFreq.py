# -*- coding: utf-8 -*-

import torch 
import torch.nn as nn
import torch.nn.functional as F
from complexLayers import  ComplexConv1d,ComplexReLU

class BasicBlock(torch.nn.Module):
    def __init__(self,n_filters,reslu,fr_size):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.reslu=reslu

        self.conv_D=ComplexConv1d(1, n_filters, kernel_size=3,padding=1)
        self.conv1_forward=ComplexConv1d(n_filters, n_filters, kernel_size=3,padding=1)
        self.conv2_forward=ComplexConv1d(n_filters, n_filters, kernel_size=3,padding=1)
        self.conv1_backward =ComplexConv1d(n_filters, n_filters, kernel_size=3,padding=1)
        self.conv2_backward = ComplexConv1d(n_filters, n_filters, kernel_size=3, padding=1)
        self.conv_G=ComplexConv1d(n_filters, 1,kernel_size=3,padding=1)

        self.beta_lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.beta_soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.beta_conv_D=ComplexConv1d(1, n_filters, kernel_size=3,padding=1)
        self.beta_conv1_forward=ComplexConv1d(n_filters, n_filters, kernel_size=3,padding=1)
        self.beta_conv2_forward=ComplexConv1d(n_filters, n_filters, kernel_size=3,padding=1)
        self.beta_conv1_backward =ComplexConv1d(n_filters, n_filters, kernel_size=3,padding=1)
        self.beta_conv2_backward = ComplexConv1d(n_filters, n_filters, kernel_size=3, padding=1)
        self.beta_conv_G=ComplexConv1d(n_filters, 1,kernel_size=3,padding=1)

        self.bn=torch.nn.BatchNorm1d(n_filters)
        self.attn=torch.nn.AvgPool1d(kernel_size=fr_size)
        self.attn_conv=torch.nn.Conv1d(1, 1, kernel_size=3,padding=1,bias=False)
        self.beta_bn = torch.nn.BatchNorm1d(n_filters)
        self.beta_attn=torch.nn.AvgPool1d(kernel_size=fr_size)
        self.beta_attn_conv=torch.nn.Conv1d(1, 1, kernel_size=3,padding=1,bias=False)

    def forward(self, x, A,B,C,D,beta,y):
        ## x update
        bsz=x.shape[0]
        x = x - self.lambda_step * (A((C(x)+D(beta*x*self.reslu)).view(bsz,1,-1)).view(bsz,1,-1)+self.reslu*beta.conj()*B((C(x)+D(beta*x*self.reslu)).view(bsz,1,-1)).view(bsz,1,-1))
        x = x + self.lambda_step * (A(y).view(bsz,1,-1)+self.reslu*beta.conj()*B(y).view(bsz,1,-1))
        x_input = x

        x_D=self.conv_D(x_input)
        x=self.conv1_forward(x_D)
        x = ComplexReLU()(x)
        x_forward=self.conv2_forward(x)

        attn=self.bn(x_forward.abs())
        attn = self.attn(attn)
        x_forward = self.attn_conv(attn.transpose(-1, -2)).transpose(-1, -2) * x_forward
        x = torch.mul(torch.sgn(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x=self.conv1_backward(x)
        x = ComplexReLU()(x)
        x_backward =self.conv2_backward(x)

        x_G=self.conv_G(x_backward)

        x_pred = x_input + x_G

        x=self.conv1_backward(x_forward)
        x = ComplexReLU()(x)
        x_D_est=self.conv2_backward(x)
        symloss = x_D_est - x_D

        ## beta update
        beta_y=y-C(x_pred).view(bsz,1,-1)
        beta = beta - self.beta_lambda_step * (x_pred.conj()*self.reslu*B(D(x_pred*beta*self.reslu).view(bsz,1,-1)).view(bsz,1,-1))
        beta = beta + self.beta_lambda_step * (x_pred.conj()*self.reslu*B(beta_y).view(bsz,1,-1))
        beta_input = beta

        beta_D = self.beta_conv_D(beta_input)
        beta = self.beta_conv1_forward(beta_D)
        beta = ComplexReLU()(beta)
        beta_forward = self.beta_conv2_forward(beta)

        beta_attn=self.beta_bn(beta_forward.real)
        beta_attn = self.beta_attn(beta_attn)
        beta_forward = self.beta_attn_conv(beta_attn.transpose(-1, -2)).transpose(-1, -2) * beta_forward
        beta = torch.mul(torch.sgn(beta_forward), F.relu(torch.abs(beta_forward) - self.beta_soft_thr))

        beta = self.beta_conv1_backward(beta)
        beta = ComplexReLU()(beta)
        beta_backward = self.beta_conv2_backward(beta)

        beta_G = self.beta_conv_G(beta_backward)

        beta_pred = beta_input + beta_G
        beta_pred[torch.abs(beta_pred) > 1 / 2] = torch.sgn(
            beta_pred[torch.abs(beta_pred) > 1 / 2]) * 1 / 100
        beta[torch.abs(beta) < 1 / 100] = torch.sgn(
            beta[torch.abs(beta) < 1 / 100]) * 1 / 100
        beta = self.beta_conv1_backward(beta_forward)
        beta = ComplexReLU()(beta)
        beta_D_est = self.beta_conv2_backward(beta)
        beta_symloss = beta_D_est - beta_D

        return [x_pred,symloss,beta_pred, beta_symloss]


class ISTANetplus(torch.nn.Module):
    def __init__(self, LayerNo,signal_dim,n_filters,inner,device,fr_size):
        super(ISTANetplus, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        self.n_filter=n_filters
        self.inner=inner
        self.device=device
        self.A=ComplexConv1d(1,  self.inner, kernel_size=signal_dim)
        self.B=ComplexConv1d(1, self.inner, kernel_size= signal_dim)
        self.C= ComplexConv1d(1,  signal_dim, kernel_size=self.inner)
        self.D = ComplexConv1d(1,  signal_dim, kernel_size=self.inner)


        for i in range(LayerNo):
            onelayer.append(BasicBlock(n_filters,1/(fr_size-1),fr_size))

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, y):

        y = (y[:, 0, :] + 1j * y[:, 1, :]).type(torch.complex64).unsqueeze(1)
        bsz=y.shape[0]
        x = self.A(y).view(bsz,1,-1)
        beta = torch.rand((bsz,1,self.inner)).to(self.device).type(torch.complex64)/2

        layers_sym = []   # for computing symmetric loss
        beta_layers_sym = []
        for i in range(self.LayerNo):
            [x, layer_sym,beta,beta_layer_sym] = self.fcs[i](x, self.A,self.B,self.C,self.D,beta,y)
            layers_sym.append(layer_sym)
            beta_layers_sym.append(beta_layer_sym)
        x_final = x.squeeze(-2)
        beta_final=beta.squeeze(-2)

        return [x_final, layers_sym,beta_final,beta_layers_sym]
