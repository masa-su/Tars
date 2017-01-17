# Tars

[![Build Status](https://travis-ci.com/masa-su/Tars.svg?branch=master&token=Tzd7hmaLjunaLnEja81y)](https://travis-ci.com/masa-su/Tars)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/6e6c735a1bc9484986a0d5877302042b)](https://www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=masa-su/Tars&amp;utm_campaign=Badge_Grade)

Tars is the deep generative models library. It has the following features:
* Various distributions
  * Gaussian, Bernoulli, Laplace, Gamma, Beta, Dirichlet, Bernoulli, Categorical, and so on.
  * We can draw samples from these distributions by **the reparameterization trick** .
* Various models
  * Autoencoder
  * VAE
     * Conditional VAE
     * Importance weighted autoencoder
     * [JMVAE](https://arxiv.org/abs/1611.01891v1)
     * Multiple latent layers
  * Multimodal VAE-GAN
  * GAN, Conditional GAN
  * VAE-GAN, conditional VAE-GAN
  * VAE-RNN
    * Variational RNN
    * DRAW, Convolutional DRAW
  * Various lower bounds
    * The evidence lower bound (ELBO, which is the same as the original lowerbound)
    * The importance sampling lowerbound 
    * The variational R'enyi bound

Note: Some of the implementations of the above models have not yet been released in this version.

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
Please go to the "examples" directory and try to run some examples.
