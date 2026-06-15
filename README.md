# Image Reconstruction Via Autoencoding Sequential Deep Image Prior (aSeqDIP)

[![NeurIPS](https://img.shields.io/badge/NeurIPS-2024-blue.svg)](https://neurips.cc/virtual/2024/poster/95683)

Repository with code to reproduce the results for **aSeqDIP** in our paper:

### Image Reconstruction Via Autoencoding Sequential Deep Image Prior (aSeqDIP)

**NeurIPS 2024**

**Authors**: Ismail R. Alkhouri*, Shijun Liang*, Evan Bell, Qing Qu, Rongrong Wang, and Saiprasad Ravishankar.

**Paper page**

Paper link: [NeurIPS 2024](https://neurips.cc/virtual/2024/poster/95683)


### Abstract: 
Recently, Deep Image Prior (DIP) has emerged as an effective unsupervised one-shot learner, delivering competitive results across various image recovery problems. This method only requires the noisy measurements and a forward operator, relying solely on deep networks initialized with random noise to learn and restore the structure of the data. However, DIP is notorious for its vulnerability to overfitting due to the overparameterization of the network. Building upon insights into the impact of the DIP input and drawing inspiration from the gradual denoising process in cutting-edge diffusion models, we introduce the Autoencoding Sequential DIP (aSeqDIP) for image reconstruction by progressively denoising and reconstructing the image through a sequential optimization of multiple network architectures. This is achieved using an input-adaptive DIP objective, combined with an autoencoding regularization term. Our approach differs from the Vanilla DIP by not relying on a single-step denoising process. Compared to diffusion models, our method does not require pre-training and outperforms DIP methods in mitigating noise overfitting while maintaining the same number of parameter updates as Vanilla DIP. Through extensive experiments, we validate the effectiveness of our method in various imaging reconstruction tasks, such as MRI and CT reconstruction, as well as in image restoration tasks like image denoising, inpainting, and brown non-linear non-uniform deblurring.


## aSeqDIP Illustrative Diagram:
![Alt text](aSeqDIP_Diagram.png)

## Results:
![Alt text](aSeqDIP_visuals.png)

### For MRI: 
Download the [fastMRI](https://github.com/microsoft/fastmri-plus/tree/main) dataset. 

### For sparse view CT: 
Download the [AAPM](https://www.aapm.org/grandchallenge/lowdosect/) dataset.

### For image restoration: 
Setup the [DPS](https://github.com/DPS2022/diffusion-posterior-sampling) code for the forward models. 

## Running the Code

Run all commands from the repository root:

```bash
cd /Aseq_DIP-main
```

Before running the scripts, update the hard-coded data paths and CUDA device ids inside the target script as needed for your machine. 
The MRI loader also requires updating `Kspace_data_name` and `mask_data_name` in `two_channel_dataset_DIP_github.py`.

```bash
# MRI reconstruction task.
python Aseq_DIP_MRI.py

# Sparse-view CT reconstruction task.
python Aseq_DIP_CT.py

# Image denoising task.
python Aseq_DIP_denoising.py

# Image inpainting task.
python Aseq_DIP_inpainting.py

# Non-linear motion deblurring task.
python Aseq_DIP_Deblurring.py
```

The Jupyter notebook can be opened with:

```bash
jupyter notebook Aseq_DIP.ipynb
```

## Blur Operator Source

In `Aseq_DIP_Deblurring.py`, the class `BlurOperator (torch.nn.Module)` is sourced from the following open-source repository:

https://github.com/VinAIResearch/blur-kernel-space-exploring/tree/main

The blur operator is used to construct the forward degradation model in the deblurring task.

## U-Net Architecture Usage

Different U-Net architectures are used for different tasks in this repository.

### MRI and CT Reconstruction

For MRI and CT reconstruction tasks, we use the **Medium U-Net** architecture provided in the original aSeqDIP repository:

- Medium U-Net implementation:  
  https://github.com/sjames40/Aseq_DIP/blob/main/unet/medium_unet.py


### Deblurring, Denoising, and Inpainting

For image deblurring, denoising, and inpainting tasks, we use the standard **U-Net** architecture from the same repository:

- U-Net implementation:  
  https://github.com/sjames40/Aseq_DIP/blob/main/unet/unet.py


### To cite our paper, use the following: 
```bibtex
@inproceedings{alkhouriNeuIPS24,
  author    = {Alkhouri, Ismail and Linag, Shijun and Bell, Evan and Qu, Qing and Wang, Rongrong and Ravishankar, Saiprasad },
  title     = {Image Reconstruction Via Autoencoding Sequential Deep Image Prior},
  booktitle   = {Advances in neural information processing systems (NeurIPS)},
  year      = {2024}
}
```
