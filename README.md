# Tars
Tars is a simple variational autoencoder library. It supports the following models:
* VAE
* Conditional VAE
* Various lowerbounds
  * The evidence lowerbound (ELBO, which is the same as the original lowerbound)
  * The importance sampling lowerbound 
  * The variational R\'enyi bound

We're going to support multiple latent layers in a few days.

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
