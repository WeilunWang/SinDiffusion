# SinDiffusion: Learning a Diffusion Model from a Single Natural Image

Official PyTorch implementation of "SinDiffusion: Learning a Diffusion Model from a Single Natural Image".
The code aims to allow the users to reproduce and extend the results reported in the study. Please cite the paper when reporting, reproducing or extending the results.

[[Arxiv (Comming Soon)]()] [[Project (Comming Soon)]()]

# Overview

This repository implements the SinDiffusion model, leveraging denoising diffusion models to capture internal distribution of patches from a single natural image. 
SinDiffusion significantly improves the quality and diversity of generated samples compared with existing GAN-based approaches. 
It is based on two core designs. 
First, SinDiffusion is trained with a single model at a single scale instead of multiple models with progressive growing of scales which serves as the default setting in prior work. 
This avoids the accumulation of errors, which cause characteristic artifacts in generated results.
Second, we identify that a patch-level receptive field of the diffusion network is crucial and effective for capturing the image's patch statistics, therefore we redesign the network structure of the diffusion model.
Extensive experiments on a wide range of images demonstrate the superiority of our proposed method for modeling the patch distribution.

<p align="center">
<img src="teaser.png" >
</p>


## Setup

## Datasets

## Training the model

## Testing the model

## Measuring SIFID and LPIPS

## Pretrained models

# Additional information

## Citation
If you use this work please cite
```
```

## License

## Acknowledge

## Contact
Please feel free to open an issue or contact us personally if you have questions, need help, or need explanations.
Write to one of the following email addresses: **wwlustc** at **mail** dot **ustc** dot **cn**.
