from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import sys
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from models import *
import torch
import torch.optim
import time
from utils.denoising_utils import *
import _pickle as cPickle
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
import torch.optim as optim
from tqdm.notebook import tqdm
from unet import UNet, FullUNet, MediumUNet, OneLayerUNet, ExtraDeepUNet, SuperDeepUNet
#from skimage.metrics import peak_signal_noise_ratio as compute_psnr
## display images
def np_plot(np_matrix, title, opt = 'RGB'):
    plt.figure(2)
    plt.clf()
    if opt == 'RGB':
        fig = plt.imshow(np_matrix.transpose(1, 2, 0), interpolation = 'nearest')
    elif opt == 'map':
        fig = plt.imshow(np_matrix, interpolation = 'bilinear', cmap = cm.RdYlGn)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.title(title)
    plt.axis('off')
    plt.pause(0.05) 



def compare_psnr(img1, img2):
    MSE = np.mean(np.abs(img1-img2)**2)
    psnr=10*np.log10(np.max(np.abs(img1))**2/MSE)
    #psnr = 10*math.log10(float(1.**2)/MSE)
    return psnr



gt_path = '/../result/label/randomffhq00000test.npy'
input_path = '/../result/input/randomffhq00000test.npy'
mask_path = '/../result/mask/randomffhq00000test.npy'
# Load the image
gt = np.load(os.path.join(gt_path))input_mask = np.load(os.path.join(input_path))
mask = np.load(os.path.join(mask_path))



def normalize_np(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= np.min(img)
    img /= np.max(img)
    return img



img_np      = normalize_np(gt[0])
img_mask_np = normalize_np(input_mask[0])




# ### Hyper-parameters

# In[7]:


#learning_rate = LR = 0.01
exp_weight = 0.99
input_depth = 3
output_depth = 3
INPUT = 'noise'
show_every = 500

## Loss
mse = torch.nn.MSELoss().type(dtype)
img_var = np_to_torch(img_np).type(dtype)
img_var_noisy = np_to_torch(img_mask_np).type(dtype)
#mask_var = np_to_torch(mask[0]).type(dtype)
def compare_psnr(img1, img2):
    MSE = np.mean(np.abs(img1-img2)**2)
    psnr=10*np.log10(np.max(np.abs(img1))**2/MSE)
    #psnr = 10*math.log10(float(1.**2)/MSE)
    return psnr


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


net = UNet(n_channels=3, n_classes=3).to(device)
#net= nn.DataParallel(net).to(device)
init_weights(net, init_type='normal',init_gain=0.02)
num_epochs = 500
learning_rate = 1e-4
show_every = 50



optimizer = optim.Adam(net.parameters(), lr = learning_rate)



net_input = torch.tensor(img_mask_np).unsqueeze(0).to(device)
noise = get_noise(input_depth, INPUT, img_np.shape[1:]).type(dtype)
net_input_saved = net_input.detach().clone()


img_var = torch.tensor(img_np).unsqueeze(0).to(device)



img_noisy_torch = np_to_torch(img_mask_np).type(dtype)


mse = torch.nn.MSELoss().type(dtype)
img_var = np_to_torch(img_np).type(dtype)
mask_var = np_to_torch(mask[0]).type(dtype)


INPUT = 'noise'
pad = 'reflection'
OPT_OVER = 'net' # optimize over the net parameters only
reg_noise_std = 1./3.
learning_rate = LR = 0.01
exp_weight=0.99
input_depth = 3 
roll_back = True # to prevent numerical issues
num_iter = 5000 # max iterations
burnin_iter = 7000 # burn-in iteration for SGLD
weight_decay = 5e-8
show_every =  500


losses = []
psnrs = []
avg_psnrs = []

out_avg = torch.zeros_like(torch.abs(img_var)).to(device)
#parameters_dict = {}
for epoch in tqdm(range(5000)):
    optimizer.zero_grad()
    out = net(net_input)
    loss_input = mse(net_input,out)
    loss = mse(out * mask_var, img_var * mask_var)+0.001*loss_input
    for i in range(2):
        optimizer.step()
        loss.backward(retain_graph=True)
    net_input = (1 - mask_var) *out.detach() + mask_var * img_var
    with torch.no_grad():
        psnr = compare_psnr(np.array(img_var.cpu()), np.array(out.cpu()))
        psnrs.append(psnr)
    
        losses.append(loss.item())
   
    
    
        avg_psnr = compare_psnr(np.array(img_var.cpu()), np.array(out.cpu()))
        avg_psnrs.append(avg_psnr)

        if epoch%show_every == 0:
            plt.figure(figsize=(12,12))
            plt.subplot(131)
            plt.imshow(out.cpu()[0].permute(1,2,0))
            plt.axis('off')
            plt.title('Sliding Average\nPSNR = ' + str(round(avg_psnr, 2)))
            #plt.colorbar(shrink=0.5, pad=0.05)
            plt.subplot(132)
            plt.imshow(img_var[0].cpu().permute(1,2,0))
            plt.axis('off')
            plt.title('Ground Truth')
            #plt.colorbar(shrink=0.5, pad=0.05) 
                            
            plt.show()





