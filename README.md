# Tars
Tars is a simple variational autoencoder library. It supports the following models:
* VAE
* Conditional VAE
* Various lowerbounds
  * The evidence lowerbound (ELBO, original lowerbound)
  * The importance sampling lowerbound 
  * The variational R\'enyi bound

We're going to support multiple latent layers in a few days.

## Installation
This library depends on Theano and Lasagne.
```
$ git clone https://github.com/masa-su/Tars.git
$ pip install -e Tars
```
or
```
$ pip install -e git://github.com/masa-su/Tars
```
## Examples
Please go to the "examples" directory and try to execute some examples.
