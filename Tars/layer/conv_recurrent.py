import numpy as np
import theano
import theano.tensor as T

from lasagne import nonlinearities
from lasagne import init
from lasagne.layers import MergeLayer, Layer, InputLayer, Conv2DLayer
from lasagne.layers import helper
from lasagne.layers import Gate
from lasagne.layers.conv import conv_output_length
from lasagne.utils import as_tuple

from ..utils import t_repeat

__all__ = [
    "ConvLSTMCell"
]


class ConvLSTMCell(MergeLayer):

    def __init__(self, x, cell_previous, hid_previous,
                 filter_size, stride=(1,1),
                 pad='same', flip_filters=True, n=None, convolution=T.nnet.conv2d,
                 ingate=Gate(W_in=init.GlorotUniform(), 
                             W_hid=init.GlorotUniform()),
                 forgetgate=Gate(W_in=init.GlorotUniform(), 
                                 W_hid=init.GlorotUniform()),
                 cell=Gate(W_in=init.GlorotUniform(), 
                           W_hid=init.GlorotUniform(), 
                           W_cell=None, 
                           nonlinearity=nonlinearities.tanh),
                 outgate=Gate(),
                 nonlinearity=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 learn_init=False,
                 peepholes=True,
                 grad_clipping=0,
                 **kwargs):

        # Initialize parent layer
        super(ConvLSTMCell, self).__init__([x, cell_previous, hid_previous], **kwargs)

        # If the provided nonlinearity is None, make it linear
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        # Retrieve the dimensionality of the incoming layer
        self.input_shape_x = self.input_shapes[0]
        self.input_shape_c = self.input_shapes[1]
        self.input_shape_h = self.input_shapes[2]

        # From ConvNet
        if n is None:
            n = len(self.input_shape_x) - 2
        elif n != len(self.input_shape_x) - 2:
            raise ValueError("Tried to create a %dD convolution layer with "
                             "input shape %r. Expected %d input dimensions "
                             "(batchsize, channels, %d spatial dimensions)." %
                             (n, self.input_shape_x, n+2, n))
        self.n = n
        self.pad = pad
        self.num_filters = self.input_shape_h[1]
        self.filter_size = as_tuple(filter_size, n, int)
        self.flip_filters = flip_filters
        self.stride = as_tuple(stride, n, int)
        self.convolution = convolution

        if self.pad == 'same':
            if any(s % 2 == 0 for s in self.filter_size):
                raise NotImplementedError(
                    '`same` padding requires odd filter size.')
        else:
            raise ValueError("You must use the 'same' padding")

        self.learn_init = learn_init
        self.peepholes = peepholes
        self.grad_clipping = grad_clipping

        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, self.get_W_shape(self.input_shape_x),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, self.get_W_shape(self.input_shape_h),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (self.num_filters, ),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        # Add in parameters from the supplied Gate instances
        (self.W_in_to_ingate, self.W_hid_to_ingate, self.b_ingate,
         self.nonlinearity_ingate) = add_gate_params(ingate, 'ingate')

        (self.W_in_to_forgetgate, self.W_hid_to_forgetgate, self.b_forgetgate,
         self.nonlinearity_forgetgate) = add_gate_params(forgetgate,
                                                         'forgetgate')

        (self.W_in_to_cell, self.W_hid_to_cell, self.b_cell,
         self.nonlinearity_cell) = add_gate_params(cell, 'cell')

        (self.W_in_to_outgate, self.W_hid_to_outgate, self.b_outgate,
         self.nonlinearity_outgate) = add_gate_params(outgate, 'outgate')

        # If peephole (cell to gate) connections were enabled, initialize
        # peephole connections.  These are elementwise products with the cell
        # state, so they are represented as vectors.

        if self.peepholes:
            self.W_cell_to_ingate = self.add_param(
                ingate.W_cell, (self.num_filters, ), name="W_cell_to_ingate")

            self.W_cell_to_forgetgate = self.add_param(
                forgetgate.W_cell, (self.num_filters, ), name="W_cell_to_forgetgate")

            self.W_cell_to_outgate = self.add_param(
                outgate.W_cell, (self.num_filters, ), name="W_cell_to_outgate")

        # Setup initial values for the cell and the hidden units
        self.cell_init = self.add_param(
            cell_init, (1, ) + self.input_shape_c[1:], name="cell_init",
            trainable=learn_init, regularizable=False)

        self.hid_init = self.add_param(
            hid_init, (1, ) + self.input_shape_h[1:], name="hid_init",
            trainable=learn_init, regularizable=False)

    def get_W_shape(self, input_shape):
        num_input_channels = input_shape[1]
        return (self.num_filters, num_input_channels) + self.filter_size

    def get_hid_init(self, num_batch):
        return t_repeat(self.hid_init, num_batch, axis=0)

    def get_cell_init(self, num_batch):
        return t_repeat(self.cell_init, num_batch, axis=0)

    def get_output_shape_for(self, input_shapes):
        pad = self.pad if isinstance(self.pad, tuple) else (self.pad,) * self.n
        input_shape_h = self.input_shapes[2]
        batchsize = input_shape_h[0]
        return ((batchsize, self.num_filters) +
                tuple(conv_output_length(input, filter, stride, p)
                      for input, filter, stride, p
                      in zip(input_shape_h[2:], self.filter_size,
                             self.stride, pad)))

    def get_output_for(self, inputs, **kwargs):
        """
        inputs: [x_t, cell_previous, h_previous]
        """
        # Retrieve the layer input
        input_n, cell_previous, hid_previous = inputs
        
        def get_gates(input_n, hid, W0, W1, b, **kwargs):
            border_mode = 'half' if self.pad == 'same' else self.pad
            input_conved = self.convolution(input_n, W0,
                                            self.input_shape_x, 
                                            self.get_W_shape(self.input_shape_x),
                                            subsample=self.stride,
                                            border_mode=border_mode,
                                            filter_flip=self.flip_filters)

            hid_conved = self.convolution(hid, W1,
                                          self.input_shape_h, 
                                          self.get_W_shape(self.input_shape_h),
                                          subsample=self.stride,
                                          border_mode=border_mode,
                                          filter_flip=self.flip_filters)

            gate = input_conved+hid_conved
            gate = gate + b.dimshuffle(('x', 0) + ('x',) * self.n)

            # Clip gradients
            if self.grad_clipping:
                gate = theano.gradient.grad_clip(
                    gate, -self.grad_clipping, self.grad_clipping)

            return gate

        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def step(input_n, cell_previous, hid_previous, *args):

            ingate = get_gates(input_n, hid_previous, self.W_in_to_ingate, self.W_hid_to_ingate, self.b_ingate)
            forgetgate = get_gates(input_n, hid_previous, self.W_in_to_forgetgate, self.W_hid_to_forgetgate, self.b_forgetgate)
            cell_input = get_gates(input_n, hid_previous, self.W_in_to_cell, self.W_hid_to_cell, self.b_cell)
            outgate = get_gates(input_n, hid_previous, self.W_in_to_outgate, self.W_hid_to_outgate, self.b_outgate)

            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous*self.W_cell_to_ingate.dimshuffle(('x', 0) + ('x',) * self.n)
                forgetgate += cell_previous*self.W_cell_to_forgetgate.dimshuffle(('x', 0) + ('x',) * self.n)

            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)

            # Compute new cell value
            cell = forgetgate*cell_previous + ingate*cell_input

            if self.peepholes:
                outgate += cell*self.W_cell_to_outgate.dimshuffle(('x', 0) + ('x',) * self.n)
            outgate = self.nonlinearity_outgate(outgate)

            # Compute new hidden unit activation
            hid = outgate*self.nonlinearity(cell)
            return [cell, hid]

        return step(input_n, cell_previous, hid_previous)
