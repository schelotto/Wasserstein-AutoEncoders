# Wasserstein Autoencoders
## Introduction
This is the implementation of [Wasserstein Auto-Encoders](https://arxiv.org/abs/1711.01558) paper in PyTorch.

For simplicity, I just use MNIST data with MLP architecture instead of DC-GAN for the encoder/decoder/discriminator, but you can replace them easily.

## Requirement
* python 3
* PyTorch >= 0.3
* torchvision

## Train
* To train an adversarial autoencoder:
```
python aae.py
```
* To train a WAE-GAN:
```
python wae_gan.py
```
* To train a WAE-MMD:
```
python wae_mmd.py
```
