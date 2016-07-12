import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from progressbar import ProgressBar

from ..utils import gauss_unitgauss_kl


class DRAW(object):

    def __init__(self, q, q_rnn, p, p_rnn, write, n_batch, optimizer, glimpses=64, random=1234):
        self.q = q
        self.q_rnn = q_rnn
        self.p = p
        self.p_rnn = p_rnn
        self.write = write
        self.n_batch = n_batch
        self.optimizer = optimizer
        self.glimpses = glimpses

        np.random.seed(random)
        self.srng = RandomStreams(seed=random)

        self.p_sample_mean_given_x()
        self.lowerbound()

    def step(self, cell_enc, cell_dec, hid_enc, hid_dec, canvas, x, deterministic=False):
        x_shape = np.array(x.shape[1:])

        # encoder
        x_err = x - self.p.fprop([canvas], self.srng, deterministic=deterministic)

        new_cell_enc, new_hid_enc = self.q_rnn.fprop([x, x_err, cell_enc, hid_enc, hid_dec], self.srng, deterministic=deterministic)
        mean, var = self.q.fprop([new_hid_enc], self.srng, deterministic=deterministic)
        kl = gauss_unitgauss_kl(mean, var).mean()

        z = self.q.sample_given_x([new_hid_enc], self.srng, deterministic=deterministic)[-1]

        # decoder
        new_cell_dec, new_hid_dec = self.p_rnn.fprop([z, cell_dec, hid_dec], self.srng, deterministic=deterministic)

        # write
        new_canvas = self.write.fprop([new_hid_dec], self.srng, deterministic=deterministic)
        new_canvas = canvas + new_canvas

        return new_cell_enc, new_cell_dec, new_hid_enc, new_hid_dec, new_canvas, z, kl

    def lowerbound(self):
        x = T.fmatrix('x')
        init_cell_enc = self.q_rnn.mean_network.get_hid_init(x.shape[0])
        init_cell_dec = self.q_rnn.mean_network.get_cell_init(x.shape[0])
        init_hid_enc = self.p_rnn.mean_network.get_hid_init(x.shape[0])
        init_hid_dec = self.p_rnn.mean_network.get_cell_init(x.shape[0])
        init_canvas = T.ones_like(x)
        outputs_info = [init_cell_enc, init_cell_dec, init_hid_enc, init_hid_dec, init_canvas, None, None]

        [cell_enc, cell_dec, hid_enc, hid_dec, canvas, z, kl], scan_updates =\
            theano.scan(fn=self.step,
                        sequences=None,
                        outputs_info=outputs_info,
                        non_sequences=x,
                        n_steps=self.glimpses)

        log_like_T = self.p.log_likelihood_given_x([[canvas[-1]], x]).mean()
        kl_T = T.sum(kl)

        lowerbound = [-kl_T, log_like_T]
        loss = -np.sum(lowerbound)

        p_rnn_params = self.p_rnn.get_params()
        q_rnn_params = self.q_rnn.get_params()
        p_params = self.p.get_params()
        q_params = self.q.get_params()
        write_params = self.write.get_params()
        params = p_rnn_params + p_params + q_rnn_params + q_params + write_params

        updates = self.optimizer(loss, params) + scan_updates

        self.get_log_likelihood = theano.function(
            inputs=[x], outputs=lowerbound, updates=scan_updates, on_unused_input='ignore')

        canvas_all = self.p.fprop([canvas.dimshuffle(1, 0, 2)], self.srng, deterministic=True)
        self.reconst = theano.function(
            inputs=[x], outputs=canvas_all, updates=scan_updates, on_unused_input='ignore')

        self.lowerbound_train = theano.function(
            inputs=[x], outputs=lowerbound, updates=updates, on_unused_input='ignore')

    def train(self, train_set):
        n_x = train_set[0].shape[0]
        nbatches = n_x // self.n_batch
        lowerbound_train = []

        pbar = ProgressBar(maxval=nbatches).start()
        for i in range(nbatches):
            start = i * self.n_batch
            end = start + self.n_batch

            batch_x = [_x[start:end] for _x in train_set]
            train_L = self.lowerbound_train(*batch_x)
            lowerbound_train.append(np.array(train_L))
            pbar.update(i)
        lowerbound_train = np.mean(lowerbound_train, axis=0)

        return lowerbound_train

    def log_likelihood_test(self, test_set):
        print "start sampling"

        n_x = test_set[0].shape[0]
        nbatches = n_x // self.n_batch

        pbar = ProgressBar(maxval=nbatches).start()
        all_log_likelihood = []
        for i in range(nbatches):
            start = i * self.n_batch
            end = start + self.n_batch
            batch_x = [_x[start:end] for _x in test_set]
            log_likelihood = get_log_likelihood(*batch_x)
            all_log_likelihood = np.r_[all_log_likelihood, log_likelihood]
            pbar.update(i)

        return all_log_likelihood

    def p_sample_mean_given_x(self):
        z = T.tensor3('z')
        z_dimshuffle = z.dimshuffle(1, 0, 2)
        init_cell_dec = self.p_rnn.mean_network.get_cell_init(z.shape[0])
        init_hid_dec = self.p_rnn.mean_network.get_cell_init(z.shape[0])
        init_canvas = T.ones((z.shape[0],) + self.write.mean_network.output_shape[1:])

        def p_step(z, cell_dec, hid_dec, canvas):
            new_cell_dec, new_hid_dec = self.p_rnn.fprop([z, cell_dec, hid_dec], self.srng, deterministic=True)
            # write
            new_canvas = self.write.fprop([new_hid_dec], self.srng, deterministic=True)
            new_canvas = canvas + new_canvas

            return new_cell_dec, new_hid_dec, new_canvas

        [cell_dec, hid_dec, canvas], scan_updates =\
            theano.scan(fn=p_step,
                        sequences=[z_dimshuffle],
                        outputs_info=[init_cell_dec, init_hid_dec, init_canvas])

        canvas = self.p.fprop([canvas.dimshuffle(1, 0, 2)], self.srng, deterministic=True)
        self.p_sample_mean_x = theano.function(
            inputs=[z], outputs=canvas, updates=scan_updates, on_unused_input='ignore')
