import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from Tars.util import tolist
import lasagne

class Concatenate(object):

    def __init__(self, distributions):
        self.distributions = distributions
        self.inputs = self.distributions[0].inputs
        #TODO: check distributions[0].inputs == distributions[0].inputs

    def get_params(self):
        params = []
        for d in self.distributions:
            params += d.get_params()
        return params

    def fprop(self, x, srng=None, deterministic=False):
        """
        inputs : x
        outputs : mean
        """
        samples = []
        for d in self.distributions:
            samples.append(d.fprop(x, srng, deterministic=deterministic))
        return T.concatenate(samples,axis=-1)

    def sample_given_x(self, x, srng, deterministic=False):
        samples = []
        for d in self.distributions:
            samples.append(d.sample_given_x(x, srng, deterministic=deterministic)[-1])
        return [x, T.concatenate(samples,axis=-1)]

    def sample_mean_given_x(self, x, srng, deterministic=False):
        samples = []
        for d in self.distributions:
            samples.append(d.sample_mean_given_x(x, srng, deterministic=deterministic)[-1])
        return [x, T.concatenate(samples,axis=-1)]

    def log_likelihood_given_x(self, samples, deterministic=False):
        """
        inputs : [[x,y,...],sample]
        outputs : p(sample|x)
        """
        loglikes = 0
        start = 0
        for i, d in enumerate(self.distributions):
            shape = d.get_output_shape()[-1]
            loglikes += d.log_likelihood_given_x([samples[0],samples[1][:,start:start+shape]], deterministic=deterministic)
            start += shape
        return loglikes

# TODO: conplicated conditional version
class Multilayer(object):
    """
    Multiple layer distribution
    p(x|z) = p(x|z1)p(z1|z2)...p(zn-1|zn)
    q(z|x) = q(zn|zn-1)...q(z2|z1)q(z1|x)
    """

    def __init__(self, distributions):
        self.distributions = distributions
        self.inputs = self.distributions[0].inputs

    def get_params(self):
        params = []
        for d in self.distributions:
            params += d.get_params()
        return params

    def __sampling(self, x, srng, deterministic):
        """
        inputs : x
        outputs : [x,z1,...,zn-1]
        """
        samples = [x]
        for i, d in enumerate(self.distributions[:-1]):
            sample = d.sample_given_x(samples[i], srng, deterministic=deterministic)
            samples.append(sample[-1])
        return samples

    def fprop(self, x, srng, deterministic=False):
        """
        inputs : x
        outputs : mean
        """
        samples = self.__sampling(x, srng, deterministic)
        sample = self.distributions[-1].fprop(tolist(samples[-1]), deterministic=deterministic)
        return sample

    def sample_given_x(self, x, srng, deterministic=False):
        """
        inputs : x
        outputs : [x,z1,...,zn]
        """
        samples = self.__sampling(x, srng, deterministic)        
        samples += self.distributions[-1].sample_given_x(tolist(samples[-1]), srng, deterministic=deterministic)[-1:]
        return samples

    def sample_mean_given_x(self, x, srng, deterministic=False):
        """
        inputs : x
        outputs : [x,z1,...,zn]
        """
        mean = self.__sampling(x, srng, deterministic)
        mean += self.distributions[-1].sample_mean_given_x(tolist(mean[-1]), deterministic=deterministic)[-1:]
        return mean

    def log_likelihood_given_x(self, samples, deterministic=False):
        """
        inputs : [[x,y,...],z1,z2,...,zn] or [[zn,y,...],zn-1,...,x]
        outputs : 
           log_likelihood (q) : [q(z1|[x,y,...]),...,q(zn|zn-1)]
           log_likelihood (p) : [p(zn-1|[zn,y,...]),...,p(x|z1)]
        """
        all_log_likelihood = 0
        for x, sample, d in zip(samples, samples[1:], self.distributions):
            log_likelihood = d.log_likelihood_given_x([tolist(x),sample])
            all_log_likelihood += log_likelihood
        return all_log_likelihood
