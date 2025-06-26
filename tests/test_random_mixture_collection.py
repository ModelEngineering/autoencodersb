import iplane.constants as cn  # type: ignore
from iplane.random_mixture_collection import PCollectionMixture, DCollectionMixture  # type: ignore

import itertools
from matplotlib import pyplot as plt  # type: ignore
import numpy as np
from scipy.stats import multivariate_normal # type: ignore
import unittest

IGNORE_TESTS = True
IS_PLOT = False
NUM_COORDINATE = 100
NUM_DIMENSION = 2
NUM_SAMPLE = NUM_COORDINATE ** NUM_DIMENSION
MEAN1D_ARR = np.array([0, 1])
COVARIANCE1D_ARR = np.array(range(10))
MEAN_ARR = np.array([[0, 1], [10, 20]])
COVARIANCE1C_ARR = np.array([[0.5, 0], [0, 0.5]])
COVARIANCE_ARR = np.array([[[0.5, 0], [0, 0.5]], [[0.5, 0], [0, 0.5]]])
WEIGHT_ARR = np.array([0.5, 0.5])
linspaces = [np.linspace(-2, 2, 100), np.linspace(-1, 3, 100)]  # Create linspaces for each dimension
DX1D_ARR = np.array([np.mean(np.diff(linspaces[n])) for n in range(len(linspaces))])  # Calculate the differential for each dimension
DX2D_ARR = np.array([DX1D_ARR, DX1D_ARR])  # Create a differential for each dimension
VARIATE2D_ARR = np.array(list(itertools.product(*linspaces)))  # Create a grid of variates
VARIATE1D_ARR = VARIATE2D_ARR[:, 0]
cov = np.reshape(COVARIANCE_ARR[0, :, :], (2,2))
mvn = multivariate_normal(mean=MEAN_ARR[0], cov=cov)   # type: ignore
DENSITY_ARR = mvn.pdf(VARIATE2D_ARR)  # type: ignore
ENTROPY = 20
#plt.scatter(VARIATE_ARR[:, 0], VARIATE_ARR[:, 1], c=DENSITY_ARR, cmap='viridis')
#plt.show()


#######################################
class TestPCollectionMixture(unittest.TestCase):

    def setUp(self):
        """Test the ParameterMGaussian class."""
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
        shape = self.pcollection.getShape()
        self.assertEqual(shape.num_component, 2)
        self.assertEqual(shape.num_dimension, 2)

    def testSelect(self):
        """Test the select method of PCollectionMixture."""
        if IGNORE_TESTS:
            return
        selected_pcollection = self.pcollection.select(dimensions=[0])
        self.assertTrue(selected_pcollection.isAllValid())
        self.assertEqual(selected_pcollection.get(cn.PC_MEAN_ARR).shape, (2,1))
        self.assertEqual(selected_pcollection.get(cn.PC_COVARIANCE_ARR).shape, (2,1,1))
        self.assertEqual(selected_pcollection.get(cn.PC_WEIGHT_ARR).shape, (2,))

    def testReshape(self):
        """Test the reshape method of PCollectionMixture."""
        #if IGNORE_TESTS:
        #    return
        mean_arr = self.pcollection.reshape(cn.PC_MEAN_ARR, MEAN1D_ARR, has_component=True, has_dimension=False)
        self.assertEqual(mean_arr.shape, (2, 1))
        #
        mean_arr = self.pcollection.reshape(cn.PC_MEAN_ARR, MEAN1D_ARR, has_component=False, has_dimension=True)
        self.assertEqual(mean_arr.shape, (1, 2))
        #
        mean_arr = self.pcollection.reshape(cn.PC_MEAN_ARR, 30, has_component=False, has_dimension=True)
        self.assertEqual(mean_arr.shape, (1, 1))
        #
        covariance_arr = self.pcollection.reshape(cn.PC_COVARIANCE_ARR, MEAN1D_ARR, has_component=False, has_dimension=True)
        self.assertEqual(covariance_arr.shape, (1, 2, 2))
        #
        covariance_arr = self.pcollection.reshape(cn.PC_COVARIANCE_ARR, np.array([1, 2, 3]),
                has_component=True, has_dimension=False)
        self.assertEqual(covariance_arr.shape, (3, 1, 1))
        #
        covariance_arr = self.pcollection.reshape(cn.PC_COVARIANCE_ARR, np.array([4]),
                has_component=True, has_dimension=False)
        self.assertEqual(covariance_arr.shape, (1, 1, 1))


#######################################
class TestDCollectionMixture(unittest.TestCase):

    def setUp(self):
        """Test the ParameterMGaussian class."""
        #if IGNORE_TESTS:
        #    return
        self.dcollection = DCollectionMixture(
            variate_arr=VARIATE2D_ARR,
            density_arr=DENSITY_ARR,
            dx_arr=DX1D_ARR,
            entropy=ENTROPY)
        
    def testConstructor(self):
        """Test the constructor of DCollectionMixture."""
        if IGNORE_TESTS:
            return
        self.assertTrue(np.array_equal(self.dcollection.get(cn.DC_VARIATE_ARR), VARIATE2D_ARR))
        self.assertTrue(np.array_equal(self.dcollection.get(cn.DC_DENSITY_ARR), DENSITY_ARR))
        self.assertTrue(np.array_equal(self.dcollection.get(cn.DC_DX_ARR), DX1D_ARR))
        self.assertEqual(self.dcollection.get(cn.DC_ENTROPY), ENTROPY)

    def testGetAll(self):
        """Test the getAll method of DCollectionMixture."""
        if IGNORE_TESTS:
            return
        variate_arr, density_arr, dx_arr, entropy = self.dcollection.getAll()
        self.assertTrue(np.array_equal(variate_arr, VARIATE2D_ARR))
        self.assertTrue(np.array_equal(density_arr, DENSITY_ARR))
        self.assertTrue(np.array_equal(dx_arr, DX1D_ARR))
        self.assertTrue(np.array_equal(entropy, ENTROPY))

    def testGetShape(self):
        """Test the getComponentAndDimension method of DCollectionMixture."""
        if IGNORE_TESTS:
            return
        shape = self.dcollection.getShape()
        self.assertEqual(shape.num_sample, NUM_SAMPLE)
        self.assertEqual(shape.num_dimension, NUM_DIMENSION)


if __name__ == '__main__':
    unittest.main()