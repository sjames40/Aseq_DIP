## import libs
from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn.init as init
from PIL import Image

from models import *
from unet import UNet

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

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

fname_gt = 'CBSD68/original_png/0010.png'

show_every = 50
learning_rate = 1e-4
sigma = 15 / 255.0

save_dir = 'results_denoising'
images_dir = os.path.join(save_dir, 'images')
os.makedirs(images_dir, exist_ok=True)

log_path = os.path.join(save_dir, 'psnr_log.txt')
best_psnr_path = os.path.join(save_dir, 'best_psnr.txt')

with open(log_path, 'w') as f:
    f.write("Epoch,PSNR,Loss\n")


print(f"Loading GT: {fname_gt}")
img_np = load_and_preprocess(fname_gt)     # (3, H, W), [0,1]


noise = np.random.normal(0.0, sigma, img_np.shape).astype(np.float32) 
img_noisy_np = np.clip(img_np + noise, 0.0, 1.0)                       # (3, H, W), [0,1]

noisy_save_path = os.path.join(save_dir, 'noisy_input.png')

noisy_img_to_save = (img_noisy_np.transpose(1, 2, 0) * 255.0).astype(np.uint8)

Image.fromarray(noisy_img_to_save).save(noisy_save_path)

print(f"Noisy image saved to: {noisy_save_path}")

img_var = np_to_torch(img_np).float().to(device)     
img_noisy_torch = np_to_torch(img_noisy_np).float().to(device)
net_input = torch.tensor(img_noisy_np).unsqueeze(0).float().to(device)

print(f"Input Shape: {net_input.shape}")  # (1, 3, H, W)

net = UNet(n_channels=3, n_classes=3).to(device)
init_weights(net, init_type='normal', init_gain=0.02)

mse = torch.nn.MSELoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

losses = []
psnrs = []
best_psnr = 0  
best_epoch = 0

print("Starting training...")
pbar = tqdm(range(2000))

for epoch in pbar:
    for i in range(2):
        optimizer.zero_grad()
        out = net(net_input)
        
        loss_input = mse(net_input, out)
        loss = mse(out, img_noisy_torch) + 1 * loss_input
        
        loss.backward()
        optimizer.step()

    net_input = out.detach().clone()

    with torch.no_grad():
        out_np = out.cpu().numpy()[0]
        gt_np = img_var.cpu().numpy()[0]
        
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

        pbar.set_postfix({'Loss': f'{loss.item():.5f}', 'PSNR': f'{psnr:.2f}', 'Best': f'{best_psnr:.2f}'})
        
        if epoch % show_every == 0:
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            out_img = np.clip(out_np.transpose(1, 2, 0), 0, 1)
            plt.imshow(out_img)
            plt.axis('off')
            plt.title(f'Epoch {epoch}\nOutput PSNR = {psnr:.2f} dB')
            
            plt.subplot(1, 2, 2)
            gt_img = np.clip(gt_np.transpose(1, 2, 0), 0, 1)
            plt.imshow(gt_img)
            plt.axis('off')
            plt.title('Ground Truth')

            fig_save_path = os.path.join(images_dir, f'epoch_{epoch:04d}.png')
            plt.savefig(fig_save_path, bbox_inches='tight')
            plt.close()

plt.figure(figsize=(10, 5))
plt.plot(psnrs, label='PSNR')
plt.axhline(best_psnr, color='r', linestyle='--', label=f'Best: {best_psnr:.2f}')
plt.title('PSNR Curve during Training')
plt.xlabel('Epoch')
plt.ylabel('PSNR (dB)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'psnr_curve.png'))
plt.close()

print(f"\nTraining Finished!")
print(f"Results saved to: {save_dir}")
print(f"Best PSNR: {best_psnr:.4f} at Epoch {best_epoch}")
