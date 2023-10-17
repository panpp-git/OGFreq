#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import torch
import util
import matplotlib.pyplot as plt
from data import fr
import matplotlib.font_manager as fm
from matplotlib import rcParams


ogfreq_path="model/ogfreq.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#load models
fr_module, _, _, _, _ = util.load(ogfreq_path, 'sbl', device)
fr_module.cpu()
fr_module.eval()
weightA=(fr_module.A.conv_r.weight+1j*fr_module.A.conv_i.weight).squeeze(1)
weightB=(fr_module.B.conv_r.weight+1j*fr_module.B.conv_i.weight).squeeze(1)
weightC=(fr_module.C.conv_r.weight+1j*fr_module.C.conv_i.weight).squeeze(1)
weightD=(fr_module.D.conv_r.weight+1j*fr_module.D.conv_i.weight).squeeze(1)
weightA_fft = np.fft.fft(weightA.detach().numpy(), 64, 1)
weightB_fft = np.fft.fft(weightB.detach().numpy(), 64, 1)
weightC_fft = np.fft.fft(weightC.detach().numpy(), 64, 0)
weightD_fft = np.fft.fft(weightD.detach().numpy(), 64, 0)

fz=10
## Paper C
plt.figure(figsize=(6,8))
plt.subplots_adjust(top=0.99, bottom=0.05, left=0.1, right=0.99, hspace=0.06, wspace=0.2)
plt.subplot(4,2,2)
plt.imshow(abs(weightA_fft).T)
plt.gca().set_ylabel('Frequency Grid', size=fz,fontproperties='Times New Roman')
plt.gca().set_xlabel('Index of Sparse Bases\n'+'(e) '+r'${\bf{W}}_1$',size=fz,fontproperties='Times New Roman')
plt.tick_params(labelsize=fz)


## Paper D
plt.subplot(4,2,4)
plt.imshow(abs(weightB_fft).T)
plt.gca().set_ylabel('Frequency Grid', size=fz,fontproperties='Times New Roman')
plt.gca().set_xlabel('Index of Sparse Bases\n'+'(f) '+r'${\bf{W}}_2$',size=fz,fontproperties='Times New Roman')
plt.tick_params(labelsize=fz)

## Paper A
plt.subplot(4,2,6)
plt.imshow(abs(weightD_fft))
plt.gca().set_ylabel('Frequency Grid', size=fz,fontproperties='Times New Roman')
plt.gca().set_xlabel('Index of Sparse Bases\n'+'(g) '+r'${\bf{W}}_3$',size=fz,fontproperties='Times New Roman')
plt.tick_params(labelsize=fz)

## Paper B
plt.subplot(4,2,8)
plt.imshow(abs(weightD_fft))
plt.gca().set_ylabel('Frequency Grid', size=fz,fontproperties='Times New Roman')
plt.gca().set_xlabel('Index of Sparse Bases\n'+'(h) '+r'${\bf{W}}_4$',size=fz,fontproperties='Times New Roman')
plt.tick_params(labelsize=fz)

x=1

######################################################################################################################
fr_size=128
sig_dim=64
C = torch.exp(2j * torch.pi / fr_size * torch.matmul(torch.arange(0, sig_dim)[:, None],
                                                     torch.arange(-fr_size // 2, fr_size // 2)[None, :])) / (fr_size * sig_dim)
D = 2j * torch.pi / fr_size * torch.arange(0, sig_dim)[:, None] * C
A = torch.exp(-2j * torch.pi / fr_size * torch.matmul(torch.arange(0, sig_dim)[:, None],
                                                     torch.arange(-fr_size // 2, fr_size // 2)[None, :])) / (fr_size * sig_dim)
B = -2j * torch.pi / fr_size * torch.arange(0, sig_dim)[:, None] * A
C=C.detach().cpu().numpy()
D=D.detach().cpu().numpy()
A=A.detach().cpu().numpy().T
B=B.detach().cpu().numpy().T
weightA_fft = np.fft.fft(A, 64, 1)
weightB_fft = np.fft.fft(B, 64, 1)
weightC_fft = np.fft.fft(C, 64, 0)
weightD_fft = np.fft.fft(D, 64, 0)

## Paper C
plt.subplot(4,2,1)
plt.imshow(abs(weightA_fft).T)
plt.gca().set_ylabel('Frequency Grid', size=fz,fontproperties='Times New Roman')
plt.gca().set_xlabel('Index of Sparse Bases\n'+'(a) '+r'${\bf{A}}^{H}$',size=fz,fontproperties='Times New Roman')
plt.tick_params(labelsize=fz)

## paper D
plt.subplot(4,2,3)
plt.imshow(abs(weightB_fft).T)
plt.gca().set_ylabel('Frequency Grid', size=fz,fontproperties='Times New Roman')
plt.gca().set_xlabel('Index of Sparse Bases\n'+'(b) '+r'${\bf{B}}^{H}$',size=fz,fontproperties='Times New Roman')
plt.tick_params(labelsize=fz)

## Paper A

plt.subplot(4,2,5)
plt.imshow(abs(weightC_fft))
plt.gca().set_ylabel('Frequency Grid', size=fz,fontproperties='Times New Roman')
plt.gca().set_xlabel('Index of Sparse Bases\n'+'(c) '+r'${\bf{A}}$',size=fz,fontproperties='Times New Roman')
plt.tick_params(labelsize=fz)

## Paper B
plt.subplot(4,2,7)
plt.imshow(abs(weightD_fft))
plt.gca().set_ylabel('Frequency Grid', size=fz,fontproperties='Times New Roman')
plt.gca().set_xlabel('Index of Sparse Bases\n'+'(d) '+r'${\bf{B}}$',size=fz,fontproperties='Times New Roman')
plt.tick_params(labelsize=fz)
b=1