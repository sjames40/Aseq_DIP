import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compute_psnr

class DownsampleModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(in_ch, out_ch, 3, stride=2), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(1), nn.Conv2d(out_ch, out_ch, 3, stride=1), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, True)
        )
    def forward(self, x): return self.conv(x)

class UpsampleModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_ch), nn.ReflectionPad2d(1), nn.Conv2d(in_ch, out_ch, 3, stride=1), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(1), nn.Conv2d(out_ch, out_ch, 3, stride=1), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, True)
        )
    def forward(self, x):
        x = self.conv(x)
        return F.interpolate(x, scale_factor=2, mode='nearest')

class SkipConnection(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(0), nn.Conv2d(in_ch, out_ch, 1, stride=1), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, True)
        )
    def forward(self, x): return self.conv(x)

class SkipArchitecture(nn.Module):
    def __init__(self, in_ch=32, out_ch=3, chs_down=[128]*5, chs_up=[128]*5, chs_skip=[4]*5):
        super().__init__()
        
        self.down = nn.ModuleList()
        current_ch = in_ch
        for ch in chs_down:
            self.down.append(DownsampleModule(current_ch, ch))
            current_ch = ch
            
        self.skip = nn.ModuleList()
        for i in range(len(chs_skip)):
            if chs_skip[i] > 0:
                self.skip.append(SkipConnection(chs_down[i], chs_skip[i]))
            else:
                self.skip.append(None)
                
        self.up = nn.ModuleList()
        for i in range(len(chs_up)):
            idx_skip = len(chs_skip) - 1 - i
            skip_ch = chs_skip[idx_skip] if (idx_skip >= 0 and chs_skip[idx_skip] > 0) else 0
            
            input_ch = chs_down[-1] if i == 0 else chs_up[i-1]
            
            self.up.append(UpsampleModule(input_ch + skip_ch, chs_up[i]))
            
        self.out = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(chs_up[-1], out_ch, 3), nn.Sigmoid())

    def forward(self, x):
        down_res = []
        # Down Path
        for m in self.down:
            x = m(x)
            down_res.append(x)
        
        # Bottom bottleneck
        x = down_res[-1]
        
        # Up Path
        for i, m in enumerate(self.up):
            idx_skip = len(down_res) - 1 - i
            
            if idx_skip >= 0:
                skip_feat = down_res[idx_skip]
                skip_module = self.skip[idx_skip]
                if skip_module is not None:
                    skip_feat = skip_module(skip_feat)
                    if x.shape[2:] != skip_feat.shape[2:]:
                        x = F.interpolate(x, size=skip_feat.shape[2:], mode='nearest')
                    
                    x = torch.cat([x, skip_feat], dim=1)
            
            x = m(x)
            
        return self.out(x)


def get_bernoulli_mask(shape, density):
    h, w = shape[0], shape[1]
    mask = (np.random.rand(h, w, 1) < density).astype(np.float32)
    return np.repeat(mask, 3, axis=2) 

def get_box_mask(shape, size=(60, 60), margin=(70, 70)):
    h, w = shape[0], shape[1]
    mask = np.ones((h, w, 3), dtype=np.float32)
    
    margin_left, margin_top = margin
    
    top = margin_top
    bottom = margin_top + size[1]
    left = margin_left
    right = margin_left + size[0]
    
    top = max(0, top)
    left = max(0, left)
    bottom = min(h, bottom)
    right = min(w, right)

    mask[top:bottom, left:right, :] = 0
    
    return mask

def np_to_torch(img_np):
    return torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).float()

def torch_to_np(img_var):
    return img_var[0].detach().cpu().numpy().transpose(1, 2, 0)


def run_inpainting(image_path, output_dir, mask_type='box', mask_param=100, num_iter=2000):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    K_stages = 200      
    N_steps = 2       
    reg_lambda = 1
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Load Image
    try:
        img_pil = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return

    # w, h = img_pil.size
    # new_w = (w // 32) * 32
    # new_h = (h // 32) * 32
    # img_pil = img_pil.resize((new_w, new_h))
    img_pil = img_pil.resize((512, 512), Image.BICUBIC)
    img_np = np.array(img_pil) / 255.0
    
    # 2. Generate Mask
    print(f"Generating {mask_type} mask...")
    if mask_type == 'bernoulli':
        mask_np = get_bernoulli_mask(img_np.shape, density=mask_param)
    else:
        mask_np = get_box_mask(img_np.shape, size=(int(mask_param), int(mask_param)))
        
    img_corrupted_np = (img_np * mask_np) + (0.4 * (1 - mask_np))
    
    img_gt_torch = np_to_torch(img_np).to(device)
    mask_torch = np_to_torch(mask_np).to(device)
    img_corrupted_torch = np_to_torch(img_corrupted_np).to(device)
    
    Image.fromarray((img_corrupted_np * 255).astype(np.uint8)).save(os.path.join(output_dir, 'input_corrupted.png'))

    net = SkipArchitecture(in_ch=3, out_ch=3, 
                           chs_down=[128]*5, chs_up=[128]*5, chs_skip=[4]*5).to(device)
    
    # 4. Optimization
    optimizer = optim.Adam(net.parameters(), lr=3e-4)
    mse_loss_fn = nn.MSELoss() 
    
    net_input = img_corrupted_torch.clone()
    
    psnr_history = [] 
    global_step = 0

    best_psnr = 0.0
    best_epoch = 0
    best_step = 0
    
    print(f"Starting aSeqDIP: K={K_stages} stages, N={N_steps} steps/stage...")
    img_gt_np_cpu = img_gt_torch.detach().cpu().numpy()[0]
    
    for k in range(K_stages):
        
        current_z = net_input.detach() 

        for i in range(N_steps):
            optimizer.zero_grad()
            
            out = net(current_z)
            
            loss_data = mse_loss_fn(out * mask_torch, img_corrupted_torch)
            loss_reg = mse_loss_fn(out, current_z)
            
            loss = loss_data + (reg_lambda * loss_reg)
            
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                # 1, 3, H, W] to [3, H, W]
                out_np_cpu = out.detach().cpu().numpy()[0]
                
                psnr_val = compute_psnr(img_gt_np_cpu, out_np_cpu, data_range=1)
                
                psnr_history.append(psnr_val)

                if psnr_val > best_psnr:
                    best_psnr = psnr_val
                    best_epoch = k + 1
                    best_step = global_step
            
            global_step += 1
            if global_step % 100 == 0:
                print(f"Stage {k+1}/{K_stages} | Step {i} | Loss: {loss.item():.6f} (Data: {loss_data.item():.6f}, Reg: {loss_reg.item():.6f}) | PSNR: {psnr_val:.2f} dB")

        with torch.no_grad():
            net_input = net(current_z).detach() 

        out_np = torch_to_np(net_input)
        final_np = (img_np * mask_np) + (out_np * (1 - mask_np))
        final_np = np.clip(final_np, 0, 1)
        Image.fromarray((final_np * 255).astype(np.uint8)).save(os.path.join(output_dir, f'stage_{k+1}_result.png'))

    plt.figure()
    plt.plot(psnr_history)
    plt.title('PSNR during aSeqDIP training')
    plt.xlabel('Global Steps')
    plt.ylabel('PSNR (dB)')
    plt.savefig(os.path.join(output_dir, 'psnr_curve.png'))
    plt.close()

    txt_path = os.path.join(output_dir, 'best_psnr.txt')
    with open(txt_path, 'w') as f:
        f.write(f"Best PSNR: {best_psnr:.4f} dB\n")
        f.write(f"Best Stage (Epoch): {best_epoch}\n")
        f.write(f"Global Step: {best_step}\n")
    print(f"Training Done. Best PSNR: {best_psnr:.4f} at Stage {best_epoch}. Saved to {txt_path}")
    
    # Save Final Comparison
    out_np = torch_to_np(net_input)
    final_np = (img_np * mask_np) + (out_np * (1 - mask_np))
    final_np = np.clip(final_np, 0, 1)
    
    gap = np.ones((img_np.shape[0], 10, 3))
    comparison = np.concatenate([img_np, gap, img_corrupted_np, gap, final_np], axis=1)
    Image.fromarray((np.clip(comparison, 0, 1) * 255).astype(np.uint8)).save(os.path.join(output_dir, 'comparison.png'))

if __name__ == "__main__":
    run_inpainting('CBSD68/original_png/0000.png', 'results', mask_type='box', mask_param=190)