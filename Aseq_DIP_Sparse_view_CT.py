
import numpy as np
import matplotlib.pyplot as plt
import torch
from unet import UNet, FullUNet, MediumUNet, OneLayerUNet, ExtraDeepUNet, SuperDeepUNet
import torch.optim as optim
import torch.fft as fft
import torch.nn as nn
from tqdm.notebook import tqdm
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
import glob
#import two_channel_dataset_DIP
from torch.nn import init
from torch.autograd import Variable
import pydicom
from skimage.transform import radon, iradon
from physics.radon import Radon, IRadon




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


img_dir = '/../Downloads/L067_FD_1_1.CT.0001.0001.2015.12.22.18.09.40.840353.358074219.IMA'


num_angles =30


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



image = pydicom.dcmread(img_dir).pixel_array # Ground Truth
theta_in = np.linspace(0., 180., num_angles, endpoint=False)
sinogram_in = radon(image, theta=theta_in,circle=False) # measurements
filter_name='ramp'
recon = iradon(sinogram_in,theta=theta_in,filter_name=filter_name) # FBP


radon_view = 30
img_width =image.shape[0]
theta = np.linspace(0, 180, radon_view, endpoint=False)
theat2 = np.linspace(0, 180, 80, endpoint=False)
circle=False


torch_radon = Radon(img_width, theta, circle).to(device)


torch_iradon = IRadon(img_width, theta, circle).to(device)



img_torch = torch.tensor(image.astype(np.float32)).unsqueeze(0).unsqueeze(0)


img_torch = img_torch/torch.max(torch.abs(img_torch))



out = torch_radon(img_torch)



from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((725, 360))
])


out2 = transform(out)


recon_out = torch_iradon(out2)



net = MediumUNet(n_channels=1, n_classes=1).to(device)
init_weights(net, init_type='normal',init_gain=0.02)
num_epochs = 500
learning_rate = 2e-4
show_every = 50



optimizer = optim.Adam(net.parameters(), lr = learning_rate)


width,length = image.shape[0],image.shape[1]


ref = Variable(torch.rand((1,1,width,length)).cuda(), requires_grad=True)



optimizer2 = optim.Adam([ref], lr = 1e-1)



with torch.no_grad():
    scale_factor = torch.linalg.norm(net(ref.to(device)))/torch.linalg.norm(recon_out.to(device))
    target_out = scale_factor * out2.to(device)
    print('K-space scaled by: ', scale_factor)

recon_out2 = torch_iradon(target_out )


alpha =1



MSE = nn.MSELoss()


# In[33]:
noise_max = 1e-4
eplision = noise_max * torch.rand(*ref.shape, device=device)
losses = []
psnrs = []
avg_psnrs = []
exp_weight = .99
out_avg = torch.zeros_like(img_torch)#.to(device)

for epoch in tqdm(range(2000)):
    optimizer.zero_grad()
    for i in range(20):
        net_output = net(ref)
        pred_out = torch_radon(net_output)
        pred_out = transform(pred_out)
        loss = torch.linalg.norm(target_out - pred_out)+ alpha * torch.linalg.norm(ref - net_output)
        loss.backward()
        optimizer.step()
    ref = net_output.detach()
    with torch.no_grad():
        out = net_output#torch_iradon(pred_out)
        out /= torch.max(out)
        #out = net_couput_final
        psnr = compute_psnr(np.array(img_torch, dtype=np.float32)/float(torch.max(img_torch)),
                            np.array(out.cpu()))
        psnrs.append(psnr)

        losses.append(loss.item())
       
        out_avg = out_avg * exp_weight + out.cpu() * (1 - exp_weight)
        avg_psnr = compute_psnr(np.array(img_torch, dtype=np.float32)/float(torch.max(img_torch)), 
                                np.array(out_avg)/float(out_avg.max().item()))
        #avg_psnr = compute_psnr(gt1, out_avg/float(out_avg.max().item())
        avg_psnrs.append(avg_psnr)
        
        #avg_ksp = avg_ksp * exp_weight + pred_ksp * (1 - exp_weight)
    
        if epoch%show_every == 0:
            plt.figure(figsize=(12,12))
            plt.subplot(131)
            #plt.imshow((out_avg.cpu()/float(out_avg.max().item()))[0][0])
            plt.imshow(out[0][0].cpu())
            plt.title('Sliding Average\nPSNR = ' + str(round(psnr, 2)))
            plt.colorbar(fraction=0.02)
            plt.subplot(132)
            plt.imshow(img_torch[0][0])
            plt.title('Ground Truth')
            #plt.colorbar()
                            
            plt.show()


# In[34]:


print(np.max(avg_psnrs))


# In[35]:


plt.imshow(out[0][0].cpu().detach().numpy(),cmap='gray')


# In[ ]:




