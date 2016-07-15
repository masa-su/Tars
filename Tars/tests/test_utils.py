from unittest import TestCase

import mock
import numpy as np
import theano.tensor as T


class TestEpsilon(TestCase):

    def test_it(self):
        from ..utils import epsilon, set_epsilon
        dummy_eps = 10
        with mock.patch('Tars.utils._EPSILON', dummy_eps):
            self.assertEquals(epsilon(), dummy_eps)
            new_eps = 30
            set_epsilon(new_eps)
            self.assertEquals(epsilon(), new_eps)


class TestToList(TestCase):

    def test_it(self):
        from ..utils import tolist
        sample_list = [1, 2, 3, 4]
        self.assertEqual(sample_list, tolist(sample_list))
        sample_value = 1
        self.assertEqual([sample_value], tolist(sample_value))


class TestLogMeanExp(TestCase):

    # https://github.com/blei-lab/edward/blob/c584c423251316a227fbfb2f669aaf45ac236c40/tests/test_log_mean_exp.py
    def test_1d(self):
        from ..utils import log_mean_exp
        x = T.constant([-1.0, -2.0, -3.0, -4.0])
        val_ed = log_mean_exp(x)
        val_true = -1.9461046625586951
        self.assert_(np.allclose(val_ed.eval(), val_true))

    def test_2d(self):
        from ..utils import log_mean_exp
        x = T.constant([[-1.0], [-2.0], [-3.0], [-4.0]])
        val_ed = log_mean_exp(x)
        val_true = -1.9461046625586951
        self.assert_(np.allclose(val_ed.eval(), val_true))


class TestLogSumExp(TestCase):

    # https://github.com/blei-lab/edward/blob/c584c423251316a227fbfb2f669aaf45ac236c40/tests/test_log_sum_exp.py
    def test_1d(self):
        from ..utils import log_sum_exp
        x = T.constant([-1.0, -2.0, -3.0, -4.0])
        val_ed = log_sum_exp(x)
        val_true = -0.5598103014388045
        assert np.allclose(val_ed.eval(), val_true)

    def test_2d(self):
        from ..utils import log_sum_exp
        x = T.constant([[-1.0], [-2.0], [-3.0], [-4.0]])
        val_ed = log_sum_exp(x)
        val_true = -0.5598103014388045
        assert np.allclose(val_ed.eval(), val_true)


class TestTRepeat(TestCase):

    def test_2d(self):
        from ..utils import t_repeat
        x = T.constant([[-1.0], [-2.0], [-3.0], [-4.0]])
        self.assertEqual(t_repeat(x, 1, 0), x)
        x_rep2_axis0 = np.array([[-1.0], [-1.0], [-2.0], [-2.0], [-3.0], [-3.0], [-4.0], [-4.0]])
        self.assert_(np.array_equal(t_repeat(x, 2, 0).eval(), x_rep2_axis0))
        x_rep2_axis1 = np.array([[-1.0, -1.0], [-2.0, -2.0], [-3.0, -3.0], [-4.0, -4.0]])
        self.assert_(np.array_equal(t_repeat(x, 2, 1).eval(), x_rep2_axis1))
