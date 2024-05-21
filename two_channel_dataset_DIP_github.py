import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Sequence, Tuple, Union
import math
import time
from torch.utils.data.dataset import Dataset
from torch.nn import init
import math
import scipy
import scipy.linalg
#import read_ocmr as read
import h5py
import sys
import os
import torch
from util.util import generate_mask_alpha, generate_mask_beta
import scipy.ndimage
from util.util import fft2, ifft2, cplx_to_tensor, complex_conj, complex_matmul, absolute
import h5py
import glob
from models import networks

def convert_2chan_into_abs_2(img):
    img_real = img[0][0]
    img_imag = img[0][1]
    img_complex = torch.complex(img_real, img_imag)
    return img_complex

def make_data_list(file_path,file_array):
    file_data = []
    for i in range(len(file_array)):
        data_file = file_array[i]
        data_from_file = np.load(os.path.join(file_path,data_file),'r')
        file_data.append(data_from_file)
    return file_data

def make_vdrs_mask(N1,N2,nlines,init_lines,seed=0):
    mask_vdrs=np.zeros((N1,N2),dtype='bool')
    low1=(N2-init_lines)//2
    low2=(N2+init_lines)//2
    mask_vdrs[:,low1:low2]=True
    nlinesout=(nlines-init_lines)//2
    rng = np.random.default_rng(seed)
    t1 = rng.choice(low1-1, size=nlinesout, replace=False)
    t2 = rng.choice(np.arange(low2+1, N2), size=nlinesout, replace=False)
    mask_vdrs[:,t1]=True; mask_vdrs[:,t2]=True
    return mask_vdrs

Kspace_data_name = '/mnt/DataA/NEW_KSPACE'
kspace_array = os.listdir(Kspace_data_name)
kspace_array = sorted(kspace_array)

kspace_data = []

number =0


index = 350+number                
kspace_file = kspace_array[index]
kspace_data_from_file = np.load(os.path.join(Kspace_data_name,kspace_file),'r')
kspace_data.append(kspace_data_from_file)

mask_vali =[]
mask_data_name = '/mnt/DataA/MRI_sampling/4_accerlation_mask'
mask_array = os.listdir(mask_data_name)
mask_array = sorted(mask_array)
mask_file = mask_array[0]
mask_from_file = np.load(os.path.join(mask_data_name,mask_file),'r')
mask_vali.append(mask_from_file)

class nyumultidataset(Dataset): # model data loader
    def  __init__(self ,kspace_data,mask_data):
        self.A_paths = kspace_data
        self.A_size = len(self.A_paths)
        self.mask_path = mask_data
        self.nx = 640
        self.ny = 368

    def __getitem__(self, index):
        A_temp = self.A_paths[index]
        s_r = A_temp['s_r']/ 32767.0 
        s_i = A_temp['s_i']/ 32767.0 
        k_r = A_temp['k_r']/ 32767.0
        k_i = A_temp['k_i']/ 32767.0 
        ncoil, nx, ny = s_r.shape
        mask_in = make_vdrs_mask(nx,ny,np.int(ny*0.25),np.int(ny*0.08))
        k_np = np.stack((k_r, k_i), axis=0)
        s_np = np.stack((s_r[:, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160],
                         s_i[:, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160]), axis=0)
        s_np_no_crop = np.stack((s_r,
                         s_i), axis=0)                 
        mask = torch.tensor(np.repeat(mask_in[np.newaxis, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160], 2, axis=0), dtype=torch.float32)
        A_k = torch.tensor(k_np, dtype=torch.float32).permute(1, 0, 2, 3)
        A_I = ifft2(A_k.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        A_I = A_I[:, :, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160]
        ##A_s is the sensitive map 
        A_s = torch.tensor(s_np, dtype=torch.float32).permute(1, 0, 2, 3)
        A_s_no_crop = torch.tensor(s_np_no_crop, dtype=torch.float32).permute(1, 0, 2, 3)
        SOS = torch.sum(complex_matmul(A_I, complex_conj(A_s)),dim=0)
        A_I = A_I/torch.max(torch.abs(SOS)[:])
        A_k2 = fft2(A_I.permute(0,2,3,1)).permute(0,3,1,2)
        kreal = A_k2
        AT = networks.OPAT2(A_s)
        Iunder = AT(kreal, mask)
        Ireal = AT(kreal, torch.ones_like(mask))
        return  Iunder, Ireal, A_s, mask, mask_in,A_s_no_crop, A_k
     
       
    def __len__(self):
        return len(self.A_paths)

    
test_clean_paths = kspace_data
mask_test_paths = mask_vali
test_dataset = nyumultidataset(test_clean_paths,mask_test_paths)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,shuffle=False)




    

    

    
    

    

    
    
    
    
