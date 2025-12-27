from __future__ import print_function
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn.init as init
import torch.nn.functional as F
from PIL import Image
import math
import cv2
from models import *
from unet import UNet

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

class BlurOperator(torch.nn.Module):
    """
    Forward operator A.

    This implementation is taken from the following repository:
    https://github.com/VinAIResearch/blur-kernel-space-exploring/tree/main

    """


# Auxiliary functions
def compare_psnr(img1, img2):
    MSE = np.mean(np.abs(img1-img2)**2)
    if MSE == 0: return 100.0
    psnr = 10 * np.log10(np.max(np.abs(img1))**2 / MSE)
    return psnr

def load_and_preprocess(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    img_pil = Image.open(path).convert('RGB')
    
    w, h = img_pil.size
    d = 32
    new_w = (w // d) * d
    new_h = (h // d) * d
    img_pil = img_pil.crop((0, 0, new_w, new_h))
    
    img_np = np.array(img_pil).transpose(2, 0, 1) / 255.0
    return img_np

def np_to_torch(img_np):
    return torch.from_numpy(img_np)[None, :]

def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m): 
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
        elif classname.find('BatchNorm2d') != -1: 
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
    print('initialize network with %s' % init_type)

# Path and parameter configuration
fname_gt = '/egr/research-slim/sunyongl/fixed_the_code/Aseq_DIP-main/ffhq_dataset/00043.png'
fname_blur = '/egr/research-slim/sunyongl/fixed_the_code/Aseq_DIP-main/ffhq_dataset/replicated_blur_00043.png'

N_iter = 2
K_iter = 5000
learning_rate = 1e-4
lambda_reg = 5.0
show_every = 100

save_dir = 'results_deblurring'
images_dir = os.path.join(save_dir, 'images')
os.makedirs(images_dir, exist_ok=True)

log_path = os.path.join(save_dir, 'psnr_log.txt')
best_psnr_path = os.path.join(save_dir, 'best_psnr.txt')

with open(log_path, 'w') as f:
    f.write("Epoch,PSNR,Loss\n")

print(f"Loading GT: {fname_gt}")
img_gt_np = load_and_preprocess(fname_gt)  # (3, H, W)

print(f"Loading Blurred Input: {fname_blur}")
img_blur_np = load_and_preprocess(fname_blur) # (3, H, W)

img_gt_torch = np_to_torch(img_gt_np).float().to(device)
img_blur_torch = np_to_torch(img_blur_np).float().to(device)

A = BlurOperator(device, kernel_size=64, seed=0).to(device)

net_input = img_blur_torch.detach().clone()

print(f"Input Shape: {net_input.shape}")

net = UNet(n_channels=3, n_classes=3).to(device)
init_weights(net, init_type='normal', init_gain=0.02)

mse = torch.nn.MSELoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# aSeqDIP
losses = []
psnrs = []
best_psnr = 0  
best_epoch = 0

print("Starting aSeqDIP deblurring...")
pbar = tqdm(range(K_iter))

for epoch in pbar:
    for i in range(N_iter):
        optimizer.zero_grad()
        out = net(net_input)
    
        out_blurred = A(out)
        loss_dc = mse(out_blurred, img_blur_torch) 
        
        loss_reg = mse(out, net_input)
        
        # loss
        loss = loss_dc + lambda_reg * loss_reg
        
        loss.backward()
        optimizer.step()

    net_input = out.detach().clone()

    with torch.no_grad():
        out_np = out.cpu().numpy()[0]
        gt_np = img_gt_torch.cpu().numpy()[0]
        
        psnr = compare_psnr(gt_np, out_np)
        psnrs.append(psnr)
        losses.append(loss.item())

        if psnr > best_psnr:
            best_psnr = psnr
            best_epoch = epoch
            with open(best_psnr_path, 'w') as f:
                f.write(f"Best Epoch: {best_epoch}, PSNR: {best_psnr:.4f}\n")

        with open(log_path, 'a') as f:
            f.write(f"{epoch},{psnr:.4f},{loss.item():.6f}\n")

        pbar.set_postfix({'Loss': f'{loss.item():.5f}', 'PSNR': f'{psnr:.2f}'})
        
        if epoch % show_every == 0:
            plt.figure(figsize=(18, 6))
            
            plt.subplot(1, 3, 1)
            gt_img = np.clip(gt_np.transpose(1, 2, 0), 0, 1)
            plt.imshow(gt_img)
            plt.axis('off')
            plt.title('Ground Truth')

            plt.subplot(1, 3, 2)
            blur_img_show = np.clip(img_blur_np.transpose(1, 2, 0), 0, 1)
            plt.imshow(blur_img_show)
            plt.axis('off')
            plt.title('Blurred Input (y)')

            plt.subplot(1, 3, 3)
            out_img = np.clip(out_np.transpose(1, 2, 0), 0, 1)
            plt.imshow(out_img)
            plt.axis('off')
            plt.title(f'Result Epoch {epoch}\nPSNR = {psnr:.2f} dB')

            fig_save_path = os.path.join(images_dir, f'epoch_{epoch:04d}.png')
            plt.savefig(fig_save_path, bbox_inches='tight')
            plt.close()

plt.figure(figsize=(10, 5))
plt.plot(psnrs, label='PSNR')
plt.axhline(best_psnr, color='r', linestyle='--', label=f'Best: {best_psnr:.2f}')
plt.title('PSNR Curve during Deblurring')
plt.xlabel('Epoch')
plt.ylabel('PSNR (dB)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'psnr_curve.png'))
plt.close()

print(f"\nDeblurring Finished!")
print(f"Results saved to: {save_dir}")
print(f"Best PSNR: {best_psnr:.4f} at Epoch {best_epoch}")