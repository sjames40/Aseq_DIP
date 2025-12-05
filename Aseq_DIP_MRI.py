import numpy as np
import matplotlib.pyplot as plt
import torch
from unet import UNet, FullUNet, MediumUNet, OneLayerUNet, ExtraDeepUNet, SuperDeepUNet
import torch.optim as optim
import torch.fft as fft
import torch.nn as nn
from tqdm import tqdm 
#import sigpy as sp
#import sigpy.mri as mr
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
# from models import skip
#from DIP_UNET_models.skip import skip
import glob
import two_channel_dataset_DIP_github
from torch.nn import init
from torch.autograd import Variable
import os


for test_direct,test_target,test_smap,test_mask,test_mask_in,no_crop_smap,test_kspace in two_channel_dataset_DIP_github.test_loader:
    k_np =test_kspace
    A_k_ref = k_np[:,:, 0, :, :] + 1j * k_np[:,:, 1, :, :]
    sense_maps_ref = no_crop_smap[:,:, 0, :, :] + 1j *no_crop_smap[:,:, 1, :, :]
    mask_from_file = test_mask_in[0].float()
    print(mask_from_file.shape)

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
    net.apply(init_func)  # apply the initialization function <init_func>

def fft_with_shifts(img):
    return fft.fftshift(fft.fft2(fft.ifftshift(img)))

def ifft_with_shifts(ksp):
    return fft.fftshift(fft.ifft2(fft.ifftshift(ksp)))

def ksp_and_mps_to_gt(ksp, mps):
    gt = mps.conj() * ifft_with_shifts(ksp)
    gt = torch.sum(gt, axis=0)
    return gt

def mps_and_gt_to_ksp(mps, gt):
    ksp = fft_with_shifts(mps * gt)
    return ksp

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

save_dir = "recon_results"
os.makedirs(save_dir, exist_ok=True)

net =MediumUNet(n_channels=2, n_classes=2).to(device)
#net= nn.DataParallel(net).to(device)
init_weights(net, init_type='normal',init_gain=0.02)
num_epochs = 500
learning_rate = 1e-4
show_every = 500

optimizer = optim.Adam(net.parameters(), lr = learning_rate)

criterion = nn.L1Loss()

ksp1 = A_k_ref[0]

mps1 = sense_maps_ref[0]

new_ref = ksp_and_mps_to_gt(mask_from_file * ksp1, mps1)

nx, ny = new_ref.shape
ref = torch.zeros(1, 2, nx, ny).to(device)
ref[:,0,:,:] = new_ref.real
ref[:,1,:,:] = new_ref.imag


with torch.no_grad():
    scale_factor = torch.linalg.norm(net(ref.to(device)))/torch.linalg.norm(ksp_and_mps_to_gt(ksp1, mps1).to(device))
    target_ksp = scale_factor * ksp1.to(device)
    print('K-space scaled by: ', scale_factor)



gt1 = ksp_and_mps_to_gt(ksp1, mps1)
gt1 = torch.abs(gt1)/torch.max(torch.abs(gt1))
img_map = torch.sum(torch.abs(mps1), axis=0) > 0
img_map = img_map.to(device)
eplision = 1e-4 * torch.rand((640, 372), dtype=torch.complex64).to(device)

# MSE = nn.MSELoss()

mask_from_file = torch.tensor(mask_from_file, dtype=torch.complex64).to(device)

mps1 = mps1.to(device)
ksp1 = ksp1.to(device)

alpha = 0.5
avg_ksp = torch.zeros_like(ksp1)
avg_ksp = avg_ksp.to(device)

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


losses = []
psnrs = []
avg_psnrs = []
exp_weight = .99
out_avg = torch.zeros_like(torch.abs(gt1)).to(device)

best_psnr = 0.0

for epoch in tqdm(range(6000)):
    # print("Epoch:", epoch)
    for i in range(10):
        optimizer.zero_grad()
        net_output = net(ref.to(device)).squeeze()
        net_output = torch.view_as_complex(net_output.permute(1, 2, 0).contiguous())
        
        pred_ksp = mps_and_gt_to_ksp(mps1.to(device), net_output)
        
        new_pred_ksp = (1 - mask_from_file).to(device) * pred_ksp.detach() / scale_factor + mask_from_file * ksp1
        
        # Continue with your loss computation and optimization steps
        loss = torch.linalg.norm(mask_from_file * target_ksp - mask_from_file * pred_ksp.squeeze()) + 1 * torch.linalg.norm(ref.to(device) - net_output)

        loss.backward()
        optimizer.step()

    new_ref = ksp_and_mps_to_gt(new_pred_ksp, mps1)#+eplision    
    ref[:, 0, :, :] = new_ref.real
    ref[:, 1, :, :] = new_ref.imag


    with torch.no_grad():
        out = img_map.to(device) * torch.abs(net_output)
        out /= torch.max(out)
        out = out.detach().squeeze()

        losses.append(loss.item())
        avg_psnr = compute_psnr(np.array(torch.abs(gt1)), np.array(out.cpu())/float(out.max().item()))
        avg_psnrs.append(avg_psnr)
        
        tqdm.write(f"Epoch {epoch}: Loss = {loss.item():.6f} | PSNR = {avg_psnr:.4f} dB | Best: {best_psnr:.4f} dB")

        if epoch%show_every == 0:
            plt.figure(figsize=(12, 6))

            plt.subplot(132)
            plt.imshow(out.cpu(), cmap='gray')
            plt.title(f'Current Output (Epoch {epoch})')
            plt.axis('off')

            plt.subplot(133)
            plt.imshow(np.abs(gt1.cpu().numpy()), cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')

            save_path = os.path.join(save_dir, f'epoch_{epoch}_psnr_{avg_psnr:.2f}.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close() 
            
            tqdm.write(f"Type: Image Saved >>> {save_path}")


nx, ny = out_avg.shape
reshape_out_avg = out_avg[ nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160].cpu()