import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams


class Model(object):
    __metaclass__ = ABCMeta

    def __init__(self, seed=1234):
        self.set_seed(seed)
        
    @abstractmethod    
    def train(self):
        pass

    def set_seed(self, seed=1234):
        self.rng = np.random.RandomState(seed)
        self.srng = RandomStreams(seed)

    def inverse_samples(self, samples):
        """
        inputs : [[x,y],z1,z2,...zn]
        outputs : [[zn,y],zn-1,...x]
        """
        inverse_samples = samples[::-1]
        inverse_samples[0] = [inverse_samples[0]] + inverse_samples[-1][1:]
        inverse_samples[-1] = inverse_samples[-1][0]
        return inverse_samples    
