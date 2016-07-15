from unittest import TestCase
import mock


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
