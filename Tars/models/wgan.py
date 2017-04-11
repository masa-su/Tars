import numpy as np
import theano
import theano.tensor as T

from ..models import GAN
from ..distributions.distribution_samples import UnitUniformSample, mean_sum_samples

"""
Improved WGAN
https://arxiv.org/pdf/1704.00028.pdf
"""


class WGAN(GAN):

    def __init__(self, p, d, prior=UnitUniformSample(),
                 random_number_eps=UnitUniformSample(a=0),
                 gradient_penalty_lambda=10,
                 p_critic=lambda gt: -gt,
                 d_critic=lambda t, gt: -t+gt,
                 **kwargs):
        self.random_number_eps = random_number_eps
        self.gradient_penalty_lambda = gradient_penalty_lambda

        super(WGAN, self).__init__(p, d, prior=prior,
                                   p_critic=p_critic, d_critic=d_critic,
                                   **kwargs)

    def _loss(self, x, deterministic=False):
        # G loss
        # z~p(z)
        z = []
        if self.prior is not None:
            z.append(self.prior.sample(self.z_shape))

        # gx~p(x|z,y,...)
        gx = self.p.sample_mean_given_x(
            z+x[1:], deterministic=deterministic)[-1]

        # gt~d(t|gx,y,...)
        gt = self.d.sample_mean_given_x(
            [gx] + x[1:], deterministic=deterministic)[-1]

        p_loss = mean_sum_samples(self.p_critic(gt)).mean()

        # D loss
        # z~p(z)
        z = []
        if self.prior is not None:
            z.append(self.prior.sample(self.z_shape))

        # gx~p(x|z,y,...)
        gx = self.p.sample_mean_given_x(
            z+x[1:], deterministic=deterministic)[-1]

        # gt~d(t|gx,y,...)
        gt = self.d.sample_mean_given_x(
            [gx] + x[1:], deterministic=deterministic)[-1]

        # gx~p(x|z,y,...)
        gx = self.p.sample_mean_given_x(
            z+x[1:], deterministic=deterministic)[-1]

        # t~d(t|x,y,...)
        t = self.d.sample_mean_given_x(
            x, deterministic=deterministic)[-1]

        d_loss = T.mean(self.d_critic(t, gt))

        # Gradient penalty
        if deterministic is False: 
            epsilon = self.random_number_eps.sample((x[0].shape[0], 1)) # 2-D
            """
            epsilon = self.random_number_eps.sample((x[0].shape[0])) # 1-D

            def _iter(i, epsilon, gx, x, *args):
                y = [_y[i][np.newaxis,:] for y in args]
                x_hat = x[i]*epsilon + gx[i]*(1-epsilon)  #1-D
                t_hat = T.mean(self.d.sample_mean_given_x(
                    [x_hat[np.newaxis,:]] + y, deterministic=deterministic)[-1])  # 0-D
                return T.grad(t_hat, x_hat)
                
            delta_t_hat, updates =\
                theano.scan(fn=_iter,
                            sequences=[T.arange(gx.shape[0]), epsilon],
                            non_sequences=[gx]+x)  # 2-D
            l2_delta_t_hat = T.sum(delta_t_hat, axis=-1)**2  # 1-D

            """
            x_hat = x[0]*epsilon + gx*(1-epsilon)  # 2-D
            t_hat = self.d.sample_mean_given_x(
                [x_hat] + x[1:], deterministic=deterministic)[-1]  # 2-D
            t_hat = T.mean(t_hat, axis=-1)  # 1-D

            delta_t_hat, updates =\
                theano.scan(lambda i, t, x : T.grad(t[i], x)[i],
                            sequences=T.arange(x_hat.shape[0]),
                            non_sequences=[t_hat, x_hat])  # 2-D
            l2_delta_t_hat = T.sum(delta_t_hat, axis=-1)**2  # 1-D

            grad_penalty = l2_delta_t_hat - 1
            d_loss += self.gradient_penalty_lambda * grad_penalty**2

        d_loss = d_loss.mean()

        if deterministic is False and len(z) > 1:
            p_loss +=\
                self.l1_lambda * mean_sum_samples(T.abs_(x[0]-gx)).mean()

        p_params = self.p.get_params()
        d_params = self.d.get_params()

        return [p_loss, d_loss], [p_params, d_params]
