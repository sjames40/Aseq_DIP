import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import GaussianBlur
from .radon import Radon, IRadon
import scipy
from torch import nn
import leapctype as leap
import leaptorch

class CT():
    def __init__(self, img_width, radon_view, uniform=True, circle=False, device='cuda:0'):
        if uniform:
            theta = np.linspace(0, 180, radon_view, endpoint=False)
            theta_all = np.linspace(0, 180, 180, endpoint=False)
        else:
            theta = torch.arange(radon_view)
            theta_all = torch.arange(radon_view)

        self.radon = Radon(img_width, theta, circle).to(device)
        self.radon_all = Radon(img_width, theta_all, circle).to(device)
        self.iradon_all = IRadon(img_width, theta_all, circle).to(device)
        self.iradon = IRadon(img_width, theta, circle).to(device)
        self.radont = IRadon(img_width, theta, circle, use_filter=None).to(device)

    def A(self, x):
        return self.radon(x)

    def A_all(self, x):
        return self.radon_all(x)

    def A_all_dagger(self, x):
        return self.iradon_all(x)

    def A_dagger(self, y):
        return self.iradon(y)

    def AT(self, y):
        return self.radont(y)
    
class PBCT:

    def __init__(self, num_views, num_rows,  num_cols, batch_size, device='cpu', angles=None):

        pixelHeight = 1
        pixelWidth = 1
        centerRow = num_rows//2
        centerCol = num_cols//2
        device = torch.device(device)

        if device == torch.device('cpu'):
            self.proj = leaptorch.Projector(forward_project=True, use_gpu=False, gpu_device=device, batch_size= batch_size)
        else:
            self.proj = leaptorch.Projector(forward_project=True, use_gpu=True, gpu_device=device, batch_size= batch_size)
        
        if angles is None:
            phis = self.proj.leapct.setAngleArray(num_views, 180.0)
        else:
            phis = angles

        self.proj.leapct.set_parallelbeam(num_views, num_rows, num_cols, pixelHeight, pixelWidth, centerRow, centerCol, phis)

        self.proj.leapct.set_diameterFOV(1.0e7)

        self.proj.set_default_volume()
        self.proj.allocate_batch_data()
        #self.proj.leapct.set_truncatedScan(True)

    def A(self, x):
        x = x.float().contiguous()
        return self.proj(x).clone()

    def A_T(self, y):
        y = y.float().contiguous()
        self.proj.forward_project = False
        x = self.proj(y)
        self.proj.forward_project = True
        return x.clone()

    def A_pinv(self, y):
        y = y.float().contiguous()
        return self.proj.fbp(y)
class CBCT:

    def __init__(self, num_views, num_rows, num_cols, sod, sdd, device='cpu', angles=None):

        pixelHeight = 10
        pixelWidth = 10
        centerRow = num_rows//2
        centerCol = num_rows//2
        device = torch.device(device)

        if device == torch.device('cpu'):
            self.proj = leaptorch.Projector(forward_project=True, use_gpu=False, gpu_device=device, batch_size=1)
        else:
            self.proj = leaptorch.Projector(forward_project=True, use_gpu=True, gpu_device=device, batch_size=1)
        
        if angles is None:
            phis = self.proj.leapct.setAngleArray(num_views, 180.0)
        else:
            phis = angles

        self.proj.leapct.set_conebeam(num_views, num_rows, num_cols, pixelHeight, pixelWidth, centerRow, centerCol,
                                phis, sod, sdd)

        self.proj.leapct.set_diameterFOV(1.0e7)

        self.proj.set_default_volume()
        self.proj.allocate_batch_data()
        self.proj.leapct.set_truncatedScan(True)
        #self.proj.leapct.set_offsetScan(True)

    def A(self, x):
        x = x.float().contiguous()
        return self.proj(x).clone()

    def A_T(self, y):
        y = y.float().contiguous()
        self.proj.forward_project = False
        x = self.proj(y)
        self.proj.forward_project = True
        return x.clone()

    def A_pinv(self, y):
        y = y.float().contiguous()
        return self.proj.fbp(y).clone()
    
# class PBCT:

#     def __init__(self, num_views, num_rows, num_cols, device='cpu', angles=None):

#         pixelHeight = 1
#         pixelWidth = 1
#         centerRow = num_rows//2
#         centerCol = num_rows//2
#         device = torch.device(device)

#         if device == torch.device('cpu'):
#             self.proj = leaptorch.Projector(forward_project=True, use_static=False, use_gpu=False, gpu_device=device, batch_size=1)
#         else:
#             self.proj = leaptorch.Projector(forward_project=True, use_static=False, use_gpu=True, gpu_device=device, batch_size=1)
        
#         if angles is None:
#             phis = self.proj.leapct.setAngleArray(num_views, 180.0)
#         else:
#             phis = angles

#         self.proj.leapct.set_parallelbeam(num_views, num_rows, num_cols, pixelHeight, pixelWidth, centerRow, centerCol, phis)

#         self.proj.leapct.set_diameterFOV(1.0e7)

#         self.proj.set_default_volume()
#         self.proj.allocate_batch_data()
#         self.proj.leapct.set_truncatedScan(True)

#     def A(self, x):
#         x = x.float().contiguous()
#         return self.proj(x).clone()

#     def A_T(self, y):
#         y = y.float().contiguous()
#         self.proj.forward_project = False
#         x = self.proj(y)
#         self.proj.forward_project = True
#         return x.clone()

#     def A_pinv(self, y):
#         y = y.float().contiguous()
#         return self.proj.fbp(y).clone()

# class CT_arbitrary_angles():
#     def __init__(self, img_width, theta, circle=False, device='cuda:0'):
#         theta_all = np.linspace(0, 180, 180, endpoint=False)
#         theta = torch.sort(180 * torch.rand(20))[0]
#         self.radon = Radon(img_width, theta, circle).to(device)
#         self.radon_all = Radon(img_width, theta_all, circle).to(device)
#         self.iradon_all = IRadon(img_width, theta_all, circle).to(device)
#         self.iradon = IRadon(img_width, theta, circle).to(device)
#         self.radont = IRadon(img_width, theta, circle, use_filter=None).to(device)

#     def A(self, x):
#         return self.radon(x)

#     def A_all(self, x):
#         return self.radon_all(x)

#     def A_all_dagger(self, x):
#         return self.iradon_all(x)

#     def A_dagger(self, y):
#         return self.iradon(y)

#     def AT(self, y):
#         return self.radont(y)
class CT_arbitrary_angles():
    def __init__(self, img_width, theta, circle=False, device='cuda:0'):
        theta_all = np.linspace(0, 180, 180, endpoint=False)
        
        self.radon = Radon(img_width, theta, circle).to(device)
        self.radon_all = Radon(img_width, theta_all, circle).to(device)
        self.iradon_all = IRadon(img_width, theta_all, circle).to(device)
        self.iradon = IRadon(img_width, theta, circle).to(device)
        self.radont = IRadon(img_width, theta, circle, use_filter=None).to(device)

    def A(self, x):
        return self.radon(x)

    def A_all(self, x):
        return self.radon_all(x)

    def A_all_dagger(self, x):
        return self.iradon_all(x)

    def A_dagger(self, y):
        return self.iradon(y)

    def AT(self, y):
        return self.radont(y)

def get_projections(vol, angles, MAC=1/100):
    
    device = vol.device
    
    ct = CT_arbitrary_angles(vol.shape[-1], angles, device=device)

    y = ct.A(vol) * MAC

    return y

def get_noisy_projections(vol, angles, I_0=10, MAC=1/100, blur_size=21, blur_sigma=1, gaussian_noise_std=5):
    
    device = vol.device
    theta = torch.sort(180 * torch.rand(angles))[0]
    ct = CT_arbitrary_angles(vol.shape[-1], angles, device=device)

    y = ct.A(vol)

    rad = I_0 * torch.exp(-y * MAC)

    rad =  gaussian_blur(rad, n=blur_size, min_sigma=blur_sigma, max_sigma=blur_sigma)

    noisy_rad = torch.poisson(rad)
    noisy_rad += gaussian_noise_std * torch.randn_like(noisy_rad)
    noisy_rad = noisy_rad.clip(1)

    y_obs =-(noisy_rad/I_0).log()

    return y_obs

def get_radiograph(vol, angles, I_0=10, MAC=1/100):
    
    device = vol.device
    
    ct = CT_arbitrary_angles(vol.shape[-1], angles, device=device)

    y = ct.A(vol)

    rad = I_0 * torch.exp(-y * MAC)

    return rad

def get_noisy_radiograph(vol, angles, I_0=10, MAC=1/100, blur_size=21, blur_sigma=1, gaussian_noise_std=5):
    
    device = vol.device
    
    ct = CT_arbitrary_angles(vol.shape[-1], angles, device=device)

    y = ct.A(vol)

    rad = I_0 * torch.exp(-y * MAC)
    conv = Blurkernel(blur_type='gaussian',
                               kernel_size=blur_size,
                               std=blur_sigma,
                               device=device).to(device)
    #rad =  gaussian_blur(rad, n=blur_size, min_sigma=blur_sigma, max_sigma=blur_sigma)
    rad = conv(rad)
    noisy_rad = torch.poisson(rad)
    noisy_rad += gaussian_noise_std * torch.randn_like(noisy_rad)

    return noisy_rad

def gaussian_blur(radiographs, n=3, min_sigma=1, max_sigma=1):

    radiographs = radiographs.permute(0,3,1,2)

    radiographs = GaussianBlur((n, n), (min_sigma, max_sigma))(radiographs)

    radiographs = radiographs.permute(0,2,3,1)

    return radiographs

class Blurkernel(nn.Module):
    def __init__(self, blur_type='gaussian', kernel_size=31, std=3.0, device=None):
        super().__init__()
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.std = std
        self.device = device
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.kernel_size//2),
            nn.Conv2d(256,256, self.kernel_size, stride=1, padding=0, bias=False, groups=256)
        )

        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        if self.blur_type == "gaussian":
            n = np.zeros((self.kernel_size, self.kernel_size))
            n[self.kernel_size // 2,self.kernel_size // 2] = 1
            k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
            noise = np.random.normal(0, 0.05, k.shape)  # Gaussian noise with mean 0 and std 0.01
            k += noise
        
        # Ensure non-negativity and normalize
            k = np.clip(k, a_min=0, a_max=None)  # Clip negative values to 0
            k /= k.sum()  # Normalize to ensure the kernel sum is 1

            k = torch.from_numpy(k)#.float().to(self.device)
#         k = k.unsqueeze(0).unsqueeze(0)  # Add channel dimensions
#         k = k.repeat(256, 1, 1, 1)  # Match the 
            #k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)
        elif self.blur_type == "motion":
            k = Kernel(size=(self.kernel_size, self.kernel_size), intensity=self.std).kernelMatrix
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)

    def update_weights(self, k):
        if not torch.is_tensor(k):
            k = torch.from_numpy(k).to(self.device)
        for name, f in self.named_parameters():
            f.data.copy_(k)

    def get_kernel(self):
        return self.k

class CT_LA():
    """
    Limited Angle tomography
    """
    def __init__(self, img_width, radon_view, uniform=True, circle=False, device='cuda:0'):
        if uniform:
            theta = np.linspace(0, 180, radon_view, endpoint=False)
        else:
            theta = torch.arange(radon_view)
        self.radon = Radon(img_width, theta, circle).to(device)
        self.iradon = IRadon(img_width, theta, circle).to(device)
        self.radont = IRadon(img_width, theta, circle, use_filter=None).to(device)

    def A(self, x):
        return self.radon(x)

    def A_dagger(self, y):
        return self.iradon(y)

    def AT(self, y):
        return self.radont(y)
