#!/usr/bin/env python
# coding: utf-8
import h5py
import os
import numpy as np
import torch
import util

import matplotlib.pyplot as plt
from data import fr
import matlab.engine
import h5py
import matplotlib.font_manager as fm
from matplotlib import rcParams
import os
import pickle
import time
os.environ['CUDA_VISIBLE_DEVICES']='4'

eng = matlab.engine.start_matlab()


resfreq_len3 = 'model/resfreq_len4096.pth'
ogfreq="model/ogfreq.pth"
ogfreq_fourier="model/ogfreq_fourier.pth"
ogfreq_bn="model/ogfreq_bn.pth"


data_dir = 'test_dataset'

device = torch.device('cpu')

#load models


fr_module3, _, _, _, _ = util.load(resfreq_len3, 'skip3', device)
fr_module3.cpu()
fr_module3.eval()

og_module, _, _, _, _ = util.load(ogfreq, 'sbl', device)
og_module.cpu()
og_module.eval()

og_module_fourier, _, _, _, _ = util.load(ogfreq_fourier, 'sbl_fourier', device)
og_module_fourier.cpu()
og_module_fourier.eval()

og_module_bn, _, _, _, _ = util.load(ogfreq_bn, 'sbl_bn', device)
og_module_bn.cpu()
og_module_bn.eval()




#load data
f = np.load(os.path.join(data_dir, 'f.npy'))
r = np.load(os.path.join(data_dir, 'r.npy'))
kernel_param_0 = 0.12/ 64

nfreq =  np.sum(f >= -0.5, axis=1)
db=['-10.0dB.npy','-5.0dB.npy','0.0dB.npy','5.0dB.npy','10.0dB.npy','15.0dB.npy','20.0dB.npy','25.0dB.npy','30.0dB.npy']
ITER=1000
fr_size=128
xgrid = np.linspace(-0.5, 0.5, fr_size, endpoint=False)
fr_size2=512
xgrid2 = np.linspace(-0.5, 0.5, fr_size2, endpoint=False)
fr_size3=4096
xgrid3 = np.linspace(-0.5, 0.5, fr_size3, endpoint=False)
reslu=xgrid[1]-xgrid[0]

gt_fr,gt_beta,gt_pos=fr.freq2fr(f, xgrid,  param=kernel_param_0,r=r,nfreq=nfreq)


OGFreq=np.zeros([len(db),1])
OGFreq_time=np.zeros([len(db),1])
OGSBI=np.zeros([len(db),1])
OGSBI_time=np.zeros([len(db),1])
GRSBI=np.zeros([len(db),1])
GRSBI_time=np.zeros([len(db),1])
FR3=np.zeros([len(db),1])
FR3_time=np.zeros([len(db),1])
OGFreq_fourier=np.zeros([len(db),1])
OGFreq_bn=np.zeros([len(db),1])
OGFreq_fourier_time=np.zeros([len(db),1])
OGFreq_bn_time=np.zeros([len(db),1])

OGFreq_conj=np.zeros([len(db),1])
OGFreq_conj_time=np.zeros([len(db),1])

fig = plt.figure()
for db_iter in range(len(db)):
    signal = np.load(os.path.join(data_dir, db[db_iter]))
    signal2 = np.load(os.path.join(data_dir, db[db_iter]))
    win = np.hamming(signal.shape[2]).astype('float32')
    for idx in range(ITER):
        with torch.no_grad():
            print(db_iter,idx)
            mv = np.max(np.sqrt(pow(signal[idx][0], 2) + pow(signal[idx][1], 2)))
            signal[idx][0]=signal[idx][0]/mv
            signal[idx][1] = signal[idx][1] / mv
            file = h5py.File('signal.h5', 'w')  # 创建一个h5文件，文件指针是f
            file['signal'] = signal2[idx]  # 将数据写入文件的主键data下面
            file.close()

            tic=time.time()
            ogsbl = eng.OGSBI(nargout=1)
            OGSBI_time[db_iter]+=time.time()-tic
            ogsbl_ret = np.zeros(fr_size)
            if np.array(ogsbl['I']).size==1:
                ogsbl_ret[np.array(ogsbl['I']).astype('int') - 1] = abs(np.array(ogsbl['mu']))
                ogsbl_beta = np.array(ogsbl['beta'])
            else:
                ogsbl_ret[np.array(ogsbl['I'])[0, :].astype('int') - 1] = abs(np.array(ogsbl['mu'])[:, 0])
                ogsbl_beta = np.array(ogsbl['beta'])[:, 0]

            tic = time.time()
            grsbl=eng.GRSBI(nargout=1)
            GRSBI_time[db_iter] += time.time() - tic
            grsbl_ret = np.zeros(fr_size)
            if np.array(grsbl['I']).size==1:
                grsbl_ret[np.array(grsbl['I']).astype('int') - 1] = abs(np.array(grsbl['mu']))
                grsbl_xcorr = np.array(grsbl['grid'])[:, 0]
                grsbl_beta = np.zeros(fr_size)
                grsbl_beta[np.array(grsbl['I']).astype('int') - 1] = grsbl_xcorr[np.array(grsbl['I']).astype('int') - 1] - xgrid[np.array(grsbl['I']).astype('int') - 1]
            else:
                grsbl_ret[np.array(grsbl['I'])[0, :].astype('int') - 1] = abs(np.array(grsbl['mu'])[:, 0])
                grsbl_xcorr = np.array(grsbl['grid'])[:, 0]
                grsbl_beta = np.zeros(fr_size)
                grsbl_beta[np.array(grsbl['I'])[0, :].astype('int') - 1] = grsbl_xcorr[np.array(grsbl['I'])[0, :].astype(
                    'int') - 1] - xgrid[np.array(grsbl['I'])[0, :].astype('int') - 1]

            tic = time.time()
            ogfreq,_,ogfreq_beta,_=og_module(torch.tensor(signal[idx][None]))
            OGFreq_time[db_iter] += time.time() - tic
            ogfreq = ogfreq.squeeze().cpu().abs()
            ogfreq = ogfreq.data.numpy()
            ogfreq_beta = ogfreq_beta[0, :].real
            xgrid_cor = ogfreq_beta.squeeze().cpu() * reslu + xgrid


            tic = time.time()
            fr3 = fr_module3(torch.tensor(signal[idx][None]))[0]
            FR3_time[db_iter] += time.time() - tic

            tic = time.time()
            ogfreq_fourier,_,ogfreq_fourier_beta,_=og_module_fourier(torch.tensor(signal[idx][None]))
            OGFreq_fourier_time[db_iter] += time.time() - tic
            ogfreq_fourier = ogfreq_fourier.squeeze().cpu().abs()
            ogfreq_fourier = ogfreq_fourier.data.numpy()
            ogfreq_fourier_beta = ogfreq_fourier_beta[0, :].real
            xgrid_fourier_cor = ogfreq_fourier_beta.squeeze().cpu() * reslu + xgrid

            tic = time.time()
            ogfreq_bn,_,ogfreq_bn_beta,_=og_module_bn(torch.tensor(signal[idx][None]))
            OGFreq_bn_time[db_iter] += time.time() - tic
            ogfreq_bn = ogfreq_bn.squeeze().cpu().abs()
            ogfreq_bn = ogfreq_bn.data.numpy()
            ogfreq_bn_beta = ogfreq_bn_beta[0, :].real
            xgrid_bn_cor = ogfreq_bn_beta.squeeze().cpu() * reslu + xgrid




        plt.tight_layout()
        plt.subplots_adjust(top=0.96,bottom=0.1,left=0.06,right=0.96,hspace=0.1,wspace=0.15)

        pos = fr.find_freq_idx(abs(ogfreq_fourier), nfreq[idx], xgrid)-1
        if len(pos) == 0:
            pos = np.argsort(-abs(ogfreq_fourier))[:nfreq[idx]]
        est=xgrid[pos]+ogfreq_fourier_beta.squeeze().cpu().numpy()[pos]*reslu
        gt=(f[idx][:nfreq[idx]])
        for k in range(nfreq[idx]):
            dist_i_direct=np.min(np.abs(est-gt[k]))
            dist_i_rshift = np.min(np.abs(est+1 - gt[k]))
            dist_i_lshift = np.min(np.abs(est-1 - gt[k]))
            dist_i=np.min([dist_i_rshift,dist_i_direct,dist_i_lshift,reslu])
            OGFreq_fourier[db_iter]+=np.power(dist_i,2.0)

        pos = fr.find_freq_idx(abs(ogfreq_bn), nfreq[idx], xgrid)-1
        if len(pos) == 0:
            pos = np.argsort(-abs(ogfreq_bn))[:nfreq[idx]]
        est=xgrid[pos]+ogfreq_bn_beta.squeeze().cpu().numpy()[pos]*reslu
        gt=(f[idx][:nfreq[idx]])
        for k in range(nfreq[idx]):
            dist_i_direct=np.min(np.abs(est-gt[k]))
            dist_i_rshift = np.min(np.abs(est+1 - gt[k]))
            dist_i_lshift = np.min(np.abs(est-1 - gt[k]))
            dist_i=np.min([dist_i_rshift,dist_i_direct,dist_i_lshift,reslu])
            OGFreq_bn[db_iter]+=np.power(dist_i,2.0)


        # OGFreq
        pos = fr.find_freq_idx(abs(ogfreq), nfreq[idx], xgrid)-1
        if len(pos) == 0:
            pos = np.argsort(-abs(ogfreq))[:nfreq[idx]]
        est=xgrid[pos]+ogfreq_beta.squeeze().cpu().numpy()[pos]*reslu
        gt=(f[idx][:nfreq[idx]])
        for k in range(nfreq[idx]):
            dist_i_direct=np.min(np.abs(est-gt[k]))
            dist_i_rshift = np.min(np.abs(est+1 - gt[k]))
            dist_i_lshift = np.min(np.abs(est-1 - gt[k]))
            dist_i=np.min([dist_i_rshift,dist_i_direct,dist_i_lshift,reslu])
            OGFreq[db_iter]+=np.power(dist_i,2.0)


        ## OGSBI
        pos = fr.find_freq_idx(abs(ogsbl_ret), nfreq[idx], xgrid)-1
        if len(pos) == 0:
            pos = np.argsort(-abs(ogsbl_ret))[:nfreq[idx]]
        est = xgrid[pos] + ogsbl_beta[pos]
        gt=(f[idx][:nfreq[idx]])
        for k in range(nfreq[idx]):
            dist_i_direct=np.min(np.abs(est-gt[k]))
            dist_i_rshift = np.min(np.abs(est+1 - gt[k]))
            dist_i_lshift = np.min(np.abs(est-1 - gt[k]))
            dist_i=np.min([dist_i_rshift,dist_i_direct,dist_i_lshift,reslu])
            OGSBI[db_iter]+=np.power(dist_i,2.0)

        ## GRSBI
        pos = fr.find_freq_idx(abs(grsbl_ret), nfreq[idx], xgrid)-1
        if len(pos) == 0:
            pos = np.argsort(-abs(grsbl_ret))[:nfreq[idx]]
        est=xgrid[pos]+grsbl_beta[pos]
        gt=(f[idx][:nfreq[idx]])
        for k in range(nfreq[idx]):
            dist_i_direct=np.min(np.abs(est-gt[k]))
            dist_i_rshift = np.min(np.abs(est+1 - gt[k]))
            dist_i_lshift = np.min(np.abs(est-1 - gt[k]))
            dist_i=np.min([dist_i_rshift,dist_i_direct,dist_i_lshift,reslu])
            GRSBI[db_iter]+=np.power(dist_i,2.0)


        # Res3
        pos = fr.find_freq_idx(abs(fr3), nfreq[idx], xgrid3)-1
        est = xgrid3[pos]
        gt=(f[idx][:nfreq[idx]])
        for k in range(nfreq[idx]):
            dist_i_direct=np.min(np.abs(est-gt[k]))
            dist_i_rshift = np.min(np.abs(est+1 - gt[k]))
            dist_i_lshift = np.min(np.abs(est-1 - gt[k]))
            dist_i=np.min([dist_i_rshift,dist_i_direct,dist_i_lshift,reslu])
            FR3[db_iter]+=np.power(dist_i,2.0)



target_num=np.sum(nfreq[:ITER])
db=list(range(-10,35,5))

fig = plt.figure(figsize=(11,7))
plt.subplots_adjust(left=None,bottom=0.1,right=None,top=0.98,wspace=None,hspace=None)
plt.tick_params(labelsize=16)
labels = plt.gca().get_xticklabels() + plt.gca().get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

plt.xlabel('SNR / dB',size=20,fontproperties='Times New Roman')
plt.ylabel('MSE',size=20)

plt.semilogy(db,OGSBI/target_num,'--',c='m',marker='o',label='OGSBI',linewidth=3,markersize=10)
plt.semilogy(db,GRSBI/target_num,'--',c='g',marker='o',label='GRSBI',linewidth=3,markersize=10)
plt.semilogy(db,FR3/target_num,'--',marker='o',label='cResFreq',linewidth=3,markersize=10)
plt.semilogy(db,OGFreq_fourier/target_num,'--',marker='o',label='OGFreq_fourier',linewidth=3,markersize=10)
plt.semilogy(db,OGFreq_bn/target_num,'--',marker='o',label='OGFreq_attn',linewidth=3,markersize=10)
plt.semilogy(db,OGFreq/target_num,'--',c='r',marker='o',label='OGFreq',linewidth=3,markersize=10)
# plt.semilogy(db,OGFreq_conj/target_num,'--',marker='o',label='OGFreq_conj',linewidth=3,markersize=10)
plt.grid(linestyle='-.')
labelss =plt.legend(frameon=True,prop={'size':16}).get_texts()
[label.set_fontname('Times New Roman') for label in labelss]


#######################################################################TIME################################################################

time=[np.sum(OGSBI_time)/ITER,np.sum(GRSBI_time)/ITER,np.sum(FR3_time)/ITER,np.sum(OGFreq_fourier_time)/ITER,np.sum(OGFreq_bn_time)/ITER,np.sum(OGFreq_time)/ITER]
a=1









