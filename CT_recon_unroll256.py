
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



img_dir = '/../Downloads/L067_FD_1_1.CT.0001.0211.2015.12.22.18.09.40.840353.358079259.IMA'



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
theta_all = np.linspace(0, 180, 180, endpoint=False)
circle=False



from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((256, 256))
])




torch_radon = Radon(img_width//2, theta, circle).to(device)



torch_iradon = IRadon(img_width//2, theta, circle).to(device)





torch_radon_all = Radon(img_width//2, theta_all, circle).to(device)



torch_iradon_all = IRadon(img_width//2, theta_all, circle).to(device)



sparsity = 6





det_count = int((image.shape[0]//2* (2*torch.ones(1)).sqrt()).ceil())




mask = torch.zeros([1, 1, det_count, 180]).to(device)
mask[..., ::sparsity] = 1



img_torch = torch.tensor(image.astype(np.float32)).unsqueeze(0).unsqueeze(0)



img_torch  = transform(img_torch )




ori_out = torch_radon_all(img_torch)




recon_out = torch_iradon_all(ori_out)



rand_full_out = ori_out*mask.cpu()#+torch_radon_all(img_torch)*(1-mask).cpu()



inverse_full_out = torch_iradon_all(rand_full_out)



inverse_full_out = inverse_full_out#/torch.max(inverse_full_out )




net = MediumUNet(n_channels=1, n_classes=1).to(device)
init_weights(net, init_type='normal',init_gain=0.02)
num_epochs = 500
learning_rate = 3e-4
show_every = 50




optimizer = optim.Adam(net.parameters(), lr = learning_rate)




ref = (inverse_full_out/torch.max(inverse_full_out )).cuda()




with torch.no_grad():
    scale_factor = torch.linalg.norm(net(ref.to(device)))/torch.linalg.norm(recon_out.to(device))
    target_out = scale_factor * ori_out.to(device)
    print('K-space scaled by: ', scale_factor)




alpha =1



losses = []
psnrs = []
for epoch in tqdm(range(8000)):
    optimizer.zero_grad()
    net_output = net(ref).contiguous()
    pred_out = torch_radon_all(net_output)
    new_pred_out = (1 - mask).to(device) * pred_out.detach() +ori_out.to(device) *(mask.to(device))
    ref = torch_iradon_all(new_pred_out)
    loss = torch.linalg.norm(mask.to(device) *target_out.to(device) - mask.to(device)*pred_out)
    + alpha * torch.linalg.norm(ref - net_output)
    #loss = MSE(inverse_full_out.to(device) ,net_output_final )+ alpha * MSE(ref, net_output_final)
    for i in range(2):
        loss.backward(retain_graph=True)
        optimizer.step()
    with torch.no_grad():
        out = net_output
        out /= torch.max(out)
        psnr = compute_psnr(np.array(img_torch, dtype=np.float32)/float(torch.max(img_torch)),
                            np.array(out.cpu()))
        psnrs.append(psnr)

        losses.append(loss.item())
       
        if epoch%show_every == 0:
            plt.figure(figsize=(12,12))
            plt.subplot(131)
            #plt.imshow((out.cpu()/float(out_avg.max().item()))[0][0])
            plt.imshow(out[0][0].cpu())
            plt.title('Sliding Average\nPSNR = ' + str(round(psnr, 2)))
            plt.colorbar(fraction=0.02)
            plt.subplot(132)
            plt.imshow(img_torch[0][0])
            plt.title('Ground Truth')
            #plt.colorbar()
                            
            plt.show()






