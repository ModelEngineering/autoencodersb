import iplane.constants as cn  # type: ignore
from iplane.random_mixture_collection import PCollectionMixture, DCollectionMixture  # type: ignore

from matplotlib import pyplot as plt  # type: ignore
import numpy as np
import unittest

IGNORE_TESTS = False
IS_PLOT = False
NUM_SAMPLE = 1000
MEAN_ARR = np.array([[0, 1], [10, 20]])
COVARIANCE_ARR = np.array([[[0.5, 0], [0, 0.5]], [[0.5, 0], [0, 0.5]]])
WEIGHT_ARR = np.array([0.5, 0.5])


#######################################
class TestPCollectionMixture(unittest.TestCase):

    def setUp(self):
        """Test the ParameterMGaussian class."""
        #if IGNORE_TESTS:
        #    return
        self.pcollection = PCollectionMixture(
            mean_arr=MEAN_ARR,
            covariance_arr=COVARIANCE_ARR,
            weight_arr=WEIGHT_ARR)
        
    def testConstructor(self):
        """Test the constructor of PCollectionMixture."""
        if IGNORE_TESTS:
            return
        self.assertEqual(self.pcollection.get(cn.PC_MEAN_ARR).shape, (2, 2))
        self.assertEqual(self.pcollection.get(cn.PC_COVARIANCE_ARR).shape, (2, 2, 2))
        self.assertEqual(self.pcollection.get(cn.PC_WEIGHT_ARR).shape, (2,))

    def testGetAll(self):
        """Test the getAll method of PCollectionMixture."""
        if IGNORE_TESTS:
            return
        mean_arr, covariance_arr, weight_arr = self.pcollection.getAll()
        self.assertTrue(np.array_equal(mean_arr, MEAN_ARR))
        self.assertTrue(np.array_equal(covariance_arr, COVARIANCE_ARR))
        self.assertTrue(np.array_equal(weight_arr, WEIGHT_ARR))

    def testGetComponentAndDimension(self):
        """Test the getComponentAndDimension method of PCollectionMixture."""
        if IGNORE_TESTS:
            return
        num_component, num_dimension = self.pcollection.getComponentAndDimension()
        self.assertEqual(num_component, 2)
        self.assertEqual(num_dimension, 2)

    def testSelect(self):
        """Test the select method of PCollectionMixture."""
        if IGNORE_TESTS:
            return
        selected_pcollection = self.pcollection.select(dimensions=[0])
        import pdb; pdb.set_trace()
        self.assertTrue(selected_pcollection.isAllValid())
        self.assertEqual(selected_pcollection.get(cn.PC_MEAN_ARR).shape, (2, 2))
        self.assertEqual(selected_pcollection.get(cn.PC_WEIGHT_ARR).shape, (2,))
        with self.assertRaises(ValueError):
            selected_pcollection.get(cn.PC_COVARIANCE_ARR)


#######################################
class TestDCollectionMixture(unittest.TestCase):

    def setUp(self):
        pass


if __name__ == '__main__':
    unittest.main()