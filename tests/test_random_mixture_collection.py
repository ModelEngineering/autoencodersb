from iplane.random_mixture_collection import PCollectionMixture, DCollectionMixture  # type: ignore

from matplotlib import pyplot as plt  # type: ignore
import numpy as np
import unittest

IGNORE_TESTS = True
IS_PLOT = False
NUM_SAMPLE = 1000


#######################################
class TestPCollectionMixture(unittest.TestCase):

    def testParameterMGaussian(self):
        """Test the ParameterMGaussian class."""
        #if IGNORE_TESTS:
        #    return
        dct = {
            'mean_arr': np.array([0, 1]),
            'covariance_arr': np.array([[0.5, 0], [0, 0.5]]),
            'weight_arr': np.array([0.5, 0.5]),
            'num_component': 2,
            'random_state': 42}
        pcollection = PCollectionMixture(parameter_dct=dct)
        #
        self.assertTrue(pcollection == pcollection)


#######################################
class TestDCollectionMixture(unittest.TestCase):

    def setUp(self):
        pass


if __name__ == '__main__':
    unittest.main()