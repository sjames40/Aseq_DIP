import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pydicom
from skimage.metrics import peak_signal_noise_ratio as compute_psnr

from unet import MediumUNet


# ============================================================
# 0) Config
# ============================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

img_path = "input-path"

save_dir = "output-path"
os.makedirs(save_dir, exist_ok=True)

NUM_VIEWS = 30
FULL_VIEWS = 180
USE_PAD725 = True

K_STAGES = 3000
N_STEPS_PER_STAGE = 5
LAMBDA_AE = 1.0
LR = 1e-4

TV_WEIGHT = 1e-4

SHOW_EVERY = 50
SAVE_EVERY = 200

SEED = 0


# ============================================================
# 1) Utils
# ============================================================
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def inner_prod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.reshape(-1) * b.reshape(-1)).sum()

def tv_loss(x: torch.Tensor) -> torch.Tensor:
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx.abs().mean() + dy.abs().mean()

def save_img(path: str, img2d: np.ndarray, title: str = None):
    plt.figure(figsize=(6, 6))
    plt.imshow(img2d, cmap="gray")
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def psnr_fixed_full(gt: torch.Tensor, out: torch.Tensor) -> float:
    scale = gt.abs().max().item()
    gt01 = (gt / (scale + 1e-8)).clamp(0, 1).squeeze().detach().cpu().numpy()
    out01 = (out / (scale + 1e-8)).clamp(0, 1).squeeze().detach().cpu().numpy()
    return float(compute_psnr(gt01, out01, data_range=1.0))


# ============================================================
# 2) Discrete Radon forward + EXACT discrete adjoint
# ============================================================
class RadonAutogradAdjoint:
    def __init__(self, img_size: int, angles_deg: np.ndarray,
                 align_corners: bool = True,
                 padding_mode: str = "zeros"):
        self.N = img_size
        self.align_corners = align_corners
        self.padding_mode = padding_mode

        angles = np.deg2rad(angles_deg.astype(np.float32))
        self.angles = torch.tensor(angles, dtype=torch.float32, device=device)
        self.V = len(self.angles)

        self.grids = []
        for theta in self.angles:
            c = torch.cos(theta)
            s = torch.sin(theta)
            R = torch.tensor([[ c, -s, 0],
                              [ s,  c, 0]], dtype=torch.float32, device=device).unsqueeze(0)
            grid = F.affine_grid(R, torch.Size([1, 1, self.N, self.N]), align_corners=self.align_corners)
            self.grids.append(grid)

    def A(self, x: torch.Tensor) -> torch.Tensor:
        B, C, N, _ = x.shape
        y = torch.zeros((B, C, N, self.V), device=x.device, dtype=x.dtype)
        for i, grid in enumerate(self.grids):
            rotated = F.grid_sample(
                x, grid.repeat(B, 1, 1, 1),
                mode="bilinear",
                padding_mode=self.padding_mode,
                align_corners=self.align_corners
            )
            y[..., i] = rotated.sum(dim=2)
        return y

    def AH(self, s: torch.Tensor) -> torch.Tensor:
        B, C, N, V = s.shape
        assert N == self.N and V == self.V
        with torch.enable_grad():
            x = torch.zeros((B, C, self.N, self.N), device=s.device, dtype=s.dtype, requires_grad=True)
            Ax = self.A(x)
            dot = inner_prod(Ax, s)
            (g,) = torch.autograd.grad(dot, x, create_graph=False, retain_graph=False)
        return g.detach()


# ============================================================
# 3) Network
# ============================================================
class NetWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = MediumUNet(n_channels=1, n_classes=1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, z):
        x = self.net(z)
        x = torch.sigmoid(x)
        return x


# ============================================================
# 4) Load CT slice
# ============================================================
set_seed(SEED)

ds = pydicom.dcmread(img_path)
img_np = ds.pixel_array.astype(np.float32)
img_np = img_np / (np.max(np.abs(img_np)) + 1e-8)
gt = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
N = gt.shape[-1]

save_img(os.path.join(save_dir, "gt.png"),
         gt.squeeze().detach().cpu().numpy(),
         title="GT (scaled)")


# ============================================================
# 5) Angles
# ============================================================
theta_full = np.linspace(0.0, 180.0, FULL_VIEWS, endpoint=False)
idx = np.linspace(0, FULL_VIEWS - 1, NUM_VIEWS, dtype=int)
theta_sparse = theta_full[idx]


# ============================================================
# 6) Operator + padding
# ============================================================
if USE_PAD725:
    det = int(math.ceil(math.sqrt(2) * N))
    pad = (det - N) // 2
    gt_pad = F.pad(gt, (pad, det - N - pad, pad, det - N - pad))
    op = RadonAutogradAdjoint(img_size=det, angles_deg=theta_sparse)
else:
    det = N
    pad = 0
    gt_pad = gt
    op = RadonAutogradAdjoint(img_size=N, angles_deg=theta_sparse)


# ============================================================
# 7) Measurement
# ============================================================
with torch.no_grad():
    y = op.A(gt_pad)

y_scale = float(y.abs().mean().item())
y_scale = max(y_scale, 1e-8)
y_n = y / y_scale

print("y shape:", tuple(y.shape), "y_scale:", y_scale)


# ============================================================
# 8) Init z0 = A^H y
# ============================================================
with torch.no_grad():
    z0_pad = op.AH(y_n)
    z0_pad = z0_pad / (z0_pad.abs().max() + 1e-8)
    if USE_PAD725:
        z0 = z0_pad[:, :, pad:pad+N, pad:pad+N]
    else:
        z0 = z0_pad


# ============================================================
# 9) aSeqDIP training
# ============================================================
net = NetWrapper().to(device)
optm = optim.Adam(net.parameters(), lr=LR)
mse = nn.MSELoss()

z = z0.detach().clone()
best_full = -1.0
best_img = None

psnr_full_hist = []
loss_hist = []

t0 = time.time()
for k in range(K_STAGES):
    for _ in range(N_STEPS_PER_STAGE):
        optm.zero_grad(set_to_none=True)

        xhat = net(z)

        if USE_PAD725:
            xhat_pad = F.pad(xhat, (pad, det - N - pad, pad, det - N - pad))
            yhat = op.A(xhat_pad) / y_scale
        else:
            yhat = op.A(xhat) / y_scale

        loss = mse(yhat, y_n) + LAMBDA_AE * mse(xhat, z)

        if TV_WEIGHT > 0:
            loss = loss + TV_WEIGHT * tv_loss(xhat)

        loss.backward()
        optm.step()

    with torch.no_grad():
        z = net(z).detach()

        p_full = psnr_fixed_full(gt, z)

        psnr_full_hist.append(p_full)
        loss_hist.append(float(loss.item()))

        if p_full > best_full:
            best_full = p_full
            best_img = z.squeeze().detach().cpu().numpy()

    if (k % SHOW_EVERY) == 0:
        msg = f"[k={k}] loss={loss.item():.4e} psnr_full={p_full:.4f} best_full={best_full:.4f}"
        print(msg)

    if (k % SAVE_EVERY) == 0:
        save_img(os.path.join(save_dir, f"iter_{k:04d}.png"),
                 z.squeeze().detach().cpu().numpy(),
                 title=f"k={k}, PSNR(full)={p_full:.2f}")

dt = time.time() - t0
print(f"Training done in {dt:.2f}s")
print(f"Best PSNR(full): {best_full}")


# ============================================================
# 10) Save results
# ============================================================
final_img = z.squeeze().detach().cpu().numpy()
plt.imsave(os.path.join(save_dir, "final_reconstruction.png"), final_img, cmap="gray")
if best_img is not None:
    plt.imsave(os.path.join(save_dir, "best_psnr_reconstruction.png"), best_img, cmap="gray")

plt.figure()
plt.plot(psnr_full_hist, label="PSNR(full)")
plt.legend()
plt.title("PSNR curves (fixed-scale)")
plt.xlabel("stage k")
plt.ylabel("PSNR (dB)")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "psnr_curves.png"), dpi=150)
plt.close()

plt.figure()
plt.plot(loss_hist)
plt.title("Loss curve")
plt.xlabel("stage k")
plt.ylabel("loss")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=150)
plt.close()

print("Saved to:", save_dir)