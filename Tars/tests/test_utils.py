import unittest
import mock


class TestEpsilon(unittest.TestCase):

    def test_it(self):
        from ..utils import epsilon, set_epsilon
        dummy_eps = 10
        with mock.patch('Tars.utils._EPSILON', dummy_eps):
            self.assertEquals(epsilon(), dummy_eps)
            new_eps = 30
            set_epsilon(new_eps)
            self.assertEquals(epsilon(), new_eps)
