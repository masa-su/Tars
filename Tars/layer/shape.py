import theano.tensor as T
from lasagne.layers import Layer

__all__ = [
    "RepeatLayer"
]


class RepeatLayer(Layer):

    def __init__(self, incoming, n, **kwargs):
        '''
        The input is expected to be a 2D tensor of shape
        (num_batch, num_features). The input is repeated
        n times such that the output will be
        (num_batch, n, num_features)
        '''
        super(RepeatLayer, self).__init__(incoming, **kwargs)
        self.n = n

    def get_output_shape_for(self, input_shape):
        return tuple([input_shape[0], self.n] + list(input_shape[1:]))

    def get_output_for(self, input, **kwargs):
        # repeat the input n times
        tensors = [input]*self.n
        stacked = T.stack(*tensors)
        dim = [1, 0] + range(2, input.ndim + 1)
        return stacked.dimshuffle(dim)
