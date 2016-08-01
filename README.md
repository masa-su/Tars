# Tars

[![Build Status](https://travis-ci.com/masa-su/Tars.svg?branch=master&token=Tzd7hmaLjunaLnEja81y)](https://travis-ci.com/masa-su/Tars)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/6e6c735a1bc9484986a0d5877302042b)](https://www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=masa-su/Tars&amp;utm_campaign=Badge_Grade)

Tars is the deep generative models library. It supports the following models:
* Autoencoder
* VAE
  * Conditional VAE
  * Importance weighted autoencoder
  * VAE-GAN (and conditional VAE-GAN)
  * [Multimodal VAE](https://kaigi.org/jsai/webprogram/2016/paper-727.html)
  * Multimodal VAE-GAN
* GAN
  * Conditional GAN
* VAE-RNN
  * Variational RNN
  * DRAW
  * Convolutional DRAW (NEW!)
* Various lowerbounds
  * The evidence lowerbound (ELBO, which is the same as the original lowerbound)
  * The importance sampling lowerbound 
  * The variational R\'enyi bound
* Multiple latent layers
* Semi-supervised model
  * VAE model
  * VAE-GAN model
* Various distributions (Gaussian, Bernoulli, Laplace,...) 

## Installation
```
$ git clone https://github.com/masa-su/Tars.git
$ pip install -e Tars --process-dependency-links
```
or
```
$ pip install -e git://github.com/masa-su/Tars --process-dependency-links
```
When you execute this command, the following packages will be automatically installed in your environment:
* Theano
* Lasagne
* progressbar2
* matplotlib
* sklearn

## Examples
Please go to the "examples" directory and try to execute some examples.

## Generated images
### MNIST
![](https://github.com/masa-su/Tars/blob/master/examples/mnist.jpg?raw=true)
### MNIST (conditional)
![](https://github.com/masa-su/Tars/blob/master/examples/mnist_conditional.jpg?raw=true)
### CelebA
![](https://github.com/masa-su/Tars/blob/master/examples/celeba.jpg?raw=true)
### CelebA (GAN)
![](https://github.com/masa-su/Tars/blob/master/examples/celeba_gan.jpg?raw=true)
