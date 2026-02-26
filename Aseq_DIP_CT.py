import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from torch.nn import init
import pydicom

from unet import UNet, FullUNet, MediumUNet, OneLayerUNet, ExtraDeepUNet, SuperDeepUNet
from physics.radon import Radon, IRadon

# =========================
# Device Configuration
# =========================
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# Init weights
# =========================
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

# =========================
# data-path
# =========================
img_dir = 'your_data_path'

dicom_data = pydicom.dcmread(img_dir)
image = dicom_data.pixel_array.astype(np.float32)
slope = getattr(dicom_data, 'RescaleSlope', 1)
intercept = getattr(dicom_data, 'RescaleIntercept', 0)
hu_image = image * slope + intercept

img_torch = torch.tensor(hu_image).float().unsqueeze(0).unsqueeze(0).to(device)
img_min = img_torch.min()
img_max = img_torch.max()
img_norm = (img_torch - img_min) / (img_max - img_min + 1e-8)

num_angles = 30
img_width = hu_image.shape[0]
theta = np.linspace(0, 180, num_angles, endpoint=False)
circle = False

torch_radon = Radon(img_width, theta, circle).to(device)
torch_iradon = IRadon(img_width, theta, circle).to(device)

target_out = torch_radon(img_norm).detach()

recon_out = torch_iradon(target_out).detach()

ref_min = recon_out.min()
ref_max = recon_out.max()
ref = (recon_out - ref_min) / (ref_max - ref_min + 1e-8)
ref = ref.clone().detach().to(device)

net = MediumUNet(n_channels=1, n_classes=1).to(device)
init_weights(net, init_type='normal', init_gain=0.02)
learning_rate = 2e-4
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

MSE = nn.MSELoss()
alpha = 1.0
show_every = 50

losses = []
psnrs = []
avg_psnrs = []
exp_weight = 0.99
out_avg = torch.zeros_like(img_norm).to(device)

best_psnr = -1e18
best_epoch = -1

save_dir = "result_path"
os.makedirs(save_dir, exist_ok=True)

save_path_png = os.path.join(save_dir, "recon_final.png")
save_path_npy = os.path.join(save_dir, "recon_final.npy")


save_path_out_tpl = os.path.join(save_dir, "recon_out_epoch_{:04d}.png")
save_path_avg_tpl = os.path.join(save_dir, "recon_avg_epoch_{:04d}.png")

img_np = img_norm.cpu().numpy()

for epoch in tqdm(range(2000)):
    for i in range(10):
        optimizer.zero_grad()
        net_output = net(ref)
        pred_out = torch_radon(net_output)
        
        loss = torch.linalg.norm(target_out - pred_out) + alpha * torch.linalg.norm(ref - net_output)
        
        loss.backward()
        optimizer.step()

    net_output = net_output.detach()
    ref = net_output
    
    with torch.no_grad():
        out_eval = net_output.clone()
        out_np = out_eval.cpu().numpy()

        psnr = compute_psnr(img_np, out_np, data_range=1.0)
        psnrs.append(psnr)
        losses.append(loss.item())

        out_avg = out_avg * exp_weight + out_eval * (1 - exp_weight)
        avg_psnr = compute_psnr(img_np, out_avg.cpu().numpy(), data_range=1.0)
        avg_psnrs.append(avg_psnr)

        if epoch % 10 == 0:
            print(f"[Epoch {epoch:04d}] PSNR={psnr:.4f}  AvgPSNR={avg_psnr:.4f}  Loss={loss.item():.6e}")

        if psnr > best_psnr:
            best_psnr = float(psnr)
            best_epoch = int(epoch)

        if epoch % show_every == 0:
            out_img = np.clip(out_eval[0][0].cpu().numpy(), 0, 1)
            avg_img = np.clip(out_avg[0][0].cpu().numpy(), 0, 1)

            # save OUT
            plt.figure(figsize=(6, 6))
            plt.imshow(out_img, cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_path_out_tpl.format(epoch), dpi=300, bbox_inches='tight', pad_inches=0.0)
            plt.close()

            # save AVG OUT
            plt.figure(figsize=(6, 6))
            plt.imshow(avg_img, cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_path_avg_tpl.format(epoch), dpi=300, bbox_inches='tight', pad_inches=0.0)
            plt.close()

            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            plt.imshow(out_img, cmap='gray')
            plt.title('Reconstruction\nPSNR = ' + str(round(psnr, 2)))
            plt.colorbar(fraction=0.046, pad=0.04)

            plt.subplot(122)
            plt.imshow(img_np[0][0], cmap='gray')
            plt.title('Ground Truth')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.show()

print("\n========== Training Summary ==========")
print(f"Best PSNR = {best_psnr:.6f} at epoch = {best_epoch}")
print(f"Best AvgPSNR (max over epochs) = {float(np.max(avg_psnrs)):.6f}")

final_out_np = out_eval[0][0].cpu().numpy()

plt.figure(figsize=(6, 6))
plt.imshow(np.clip(final_out_np, 0, 1), cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.savefig(save_path_png, dpi=300, bbox_inches='tight', pad_inches=0.0)
plt.close()

np.save(save_path_npy, final_out_np)

print(f"Saved final reconstruction PNG to: {save_path_png}")
print(f"Saved final reconstruction NPY to: {save_path_npy}")

plt.imshow(np.clip(final_out_np, 0, 1), cmap='gray')
plt.show()