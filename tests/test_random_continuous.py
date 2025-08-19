from iplane.random_continuous import RandomContinuous, PCollectionContinuous, DCollectionContinuous  # type: ignore
from iplane.random_mixture import RandomMixture  # type: ignore
from iplane.random_mixture_collection import PCollectionMixture  # type: ignore
import iplane.constants as cn  # type: ignore

from matplotlib import pyplot as plt  # type: ignore
import numpy as np
from scipy.stats import multivariate_normal # type: ignore
from typing import List, Any, Tuple
import unittest

IGNORE_TESTS = True
IS_PLOT = False
NUM_SAMPLE = 50
STD = 4
GAUSSIAN_SAMPLE_ARR = np.random.normal(loc=5, scale=STD, size=NUM_SAMPLE).reshape(-1, 1)  # Sample for testing
UNIFORM_SAMPLE_ARR = np.random.uniform(low=0, high=8, size=NUM_SAMPLE).reshape(-1, 1)  # Uniform sample for testing


class TestPCollectionContinuous(unittest.TestCase):

    def testConstructor(self):
        """Test the constructor of PCollectionContinuous."""
        if IGNORE_TESTS:
            return
        dct = dict(
            mean_arr=np.array(range(NUM_SAMPLE)).reshape(-1, 1),
            covariance_arr= np.array([[[STD**2]]]),
            weight_arr=np.array([1.0]),
        )
        pcollection = PCollectionContinuous(cn.PC_MIXTURE_NAMES, dct)
        self.assertIsInstance(pcollection, PCollectionContinuous)
        self.assertEqual(pcollection.get(cn.PC_WEIGHT_ARR).shape, (1,))


###################################################
class TestDCollectionContinuous(unittest.TestCase):

    def setUp(self):
        self.entropy = 5
        self.dcollection = DCollectionContinuous(
            variate_arr=np.array(range(NUM_SAMPLE)).reshape(-1, 1),
            density_arr=GAUSSIAN_SAMPLE_ARR,
            dx_arr=np.array(np.repeat(1, 1)),
            entropy=self.entropy,
        )

    def testConstructor(self):
        if IGNORE_TESTS:
            return
        self.assertIsInstance(self.dcollection, DCollectionContinuous)
        self.assertEqual(self.dcollection.get(cn.DC_VARIATE_ARR).shape, (NUM_SAMPLE, 1))
    
    def testNumDimension(self):
        if IGNORE_TESTS:
            return
        num_dimension = self.dcollection.num_dimension
        self.assertEqual(num_dimension, 1)
        #
        single_variate_arr = np.array(range(NUM_SAMPLE))
        dcollection = DCollectionContinuous(
            variate_arr=np.array([single_variate_arr, single_variate_arr]).T,
            density_arr=GAUSSIAN_SAMPLE_ARR,
            dx_arr=np.array(np.repeat(1, 1)),
            entropy=self.entropy,
        )
        num_dimension = dcollection.num_dimension
        self.assertEqual(num_dimension, 2)

    def testGetAll(self):
        """Test the getAll method of DCollectionContinuous."""
        if IGNORE_TESTS:
            return
        variate_arr, density_arr, dx_arr, entropy = self.dcollection.getAll()
        self.assertIsInstance(variate_arr, np.ndarray)
        self.assertIsInstance(density_arr, np.ndarray)
        self.assertIsInstance(dx_arr, np.ndarray)
        self.assertIsInstance(entropy, (int, float))
        self.assertEqual(variate_arr.shape, (NUM_SAMPLE, 1))
        self.assertEqual(density_arr.shape, (NUM_SAMPLE, 1))


###################################################
class TestRandomContinuous(unittest.TestCase):

    def setUp(self):
        dct = dict(
            mean_arr=np.array(range(NUM_SAMPLE)).reshape(-1, 1),
            covariance_arr= np.array([[[STD**2]]]),
            weight_arr=np.array([1.0]),
        )
        self.pcollection = PCollectionContinuous(cn.PC_MIXTURE_NAMES, dct)
        self.random = RandomContinuous(pcollection=self.pcollection, num_variate_sample=NUM_SAMPLE)

    def testConstructor(self):
        """Test the constructor of RandomContinuous."""
        #if IGNORE_TESTS:
        #    return
        self.assertIsInstance(self.random, RandomContinuous)
        self.assertIsNotNone(self.random.pcollection)
        self.assertEqual(self.random.pcollection.get(cn.PC_WEIGHT_ARR).shape, (1,))  # type: ignore

    def testMakeVariate(self):
        """Test the creation of DCollectionContinuous."""
        #if IGNORE_TESTS:
        #    return
        min_point = np.array([1, 2])
        max_point = np.array([10, 20])
        num_sample = 100
        variate_arr, dx_arr = self.random.makeVariate(min_point, max_point, num_sample)
        self.assertEqual(variate_arr.shape, (num_sample, 2))
        self.assertEqual(dx_arr.shape, (2,))
        if IS_PLOT:
            plt.scatter(variate_arr[:, 0], variate_arr[:, 1])
            plt.title("Variate Array")
            plt.show()

    def testMakeEntropy(self):
        """Test the calculation of entropy."""
        #if IGNORE_TESTS:
        #    return
        num_dimension = 5
        mean = 100
        num_sample = 8**num_dimension
        #
        mean_arr = np.repeat(mean, num_dimension)
        diagonal = np.array([2.0] * num_dimension)
        covariance_arr = np.diag(diagonal)
        min_point = np.repeat(mean-5, num_dimension)
        max_point = np.repeat(mean+5, num_dimension)
        variate_arr, dx_arr = self.random.makeVariate(min_point, max_point, num_sample)
        mvn = multivariate_normal(mean=mean_arr, cov=covariance_arr)  # type: ignore
        density_arr = mvn.pdf(variate_arr)
        if IS_PLOT:
            plt.scatter(variate_arr[:, 0], variate_arr[:, 1], 10*density_arr, c='red', marker='^', alpha=0.6)
            plt.show()
        # Calculate entropy
        pcollection  = PCollectionMixture(
            mean_arr=mean_arr.reshape(1, -1),
            covariance_arr=np.array([covariance_arr]),
            weight_arr=np.array([1.0]),
        )
        random_mixture = RandomMixture(pcollection=pcollection)
        calculated_entropy = random_mixture.calculateEntropy(pcollection)
        entropy = self.random.makeEntropy(density_arr, dx_arr)
        #print(f"Calculated Entropy: {calculated_entropy}, Entropy from makeEntropy: {entropy}")
        self.assertIsInstance(entropy, (int, float))
        self.assertAlmostEqual(entropy, calculated_entropy, places=2)



if __name__ == '__main__':
    unittest.main()