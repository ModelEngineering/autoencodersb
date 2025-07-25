from iplane.random_empirical import RandomEmpirical, PCollectionEmpirical, DCollectionEmpirical  # type: ignore
from iplane.random_mixture import RandomMixture, PCollectionMixture  # type: ignore
import iplane.constants as cn  # type: ignore

from matplotlib import pyplot as plt  # type: ignore
import numpy as np
from typing import List, Any, Tuple
import unittest

IGNORE_TESTS = False
IS_PLOT = False
NUM_SAMPLE = 50
STD = 4
GAUSSIAN_SAMPLE_ARR = np.random.normal(loc=5, scale=STD, size=NUM_SAMPLE).reshape(-1, 1)  # Sample for testing
UNIFORM_SAMPLE_ARR = np.random.uniform(low=0, high=8, size=NUM_SAMPLE).reshape(-1, 1)  # Uniform sample for testing


class TestRandomEmpirical(unittest.TestCase):

    def setUp(self):
        self.pcollection = PCollectionEmpirical(training_arr=GAUSSIAN_SAMPLE_ARR)
        self.random = RandomEmpirical(total_num_sample=NUM_SAMPLE)
        self.random_mixture = RandomMixture()

    def testEstimatePCollection(self):
        """Test the estimation of PCollectionEmpirical."""
        if IGNORE_TESTS:
            return
        pcollection = self.random.makePCollection(GAUSSIAN_SAMPLE_ARR)
        self.assertIsInstance(pcollection, PCollectionEmpirical)
        self.assertTrue(np.array_equal(pcollection.get(cn.PC_TRAINING_ARR), GAUSSIAN_SAMPLE_ARR))

    def testMakeDCollection(self):
        """Test the creation of DCollectionEmpirical."""
        if IGNORE_TESTS:
            return
        _ = self.random.makePCollection(GAUSSIAN_SAMPLE_ARR)
        dcollection = self.random.makeDCollection(pcollection=self.pcollection)
        pcollection_mixture = PCollectionMixture(
            mean_arr=np.array([[0]]),
            covariance_arr=np.array([[[STD**2]]]),
            weight_arr=np.array([1.0])
        )
        entropy_mixture = self.random_mixture.calculateEntropy(pcollection_mixture)
        entropy_empirical = self.random.calculateEntropy(dcollection)
        print(f"Entropy Mixture: {entropy_mixture}, Entropy Empirical: {entropy_empirical}")
        # Do uniform
        self.random = RandomEmpirical(total_num_sample=NUM_SAMPLE)
        self.assertIsInstance(dcollection, DCollectionEmpirical)

    def testCalculateEntropy(self):
        """Test the calculation of entropy."""
        if IGNORE_TESTS:
            return
        return
        self.random.makePCollection(GAUSSIAN_SAMPLE_ARR)
        dcollection = self.random.makeDCollection()
        entropy = self.random.calculateEntropy(dcollection)
        self.assertIsInstance(entropy, float)

    def testMakeCDFUnivariate(self):
        """Test the creation of CDF from variate array."""
        if IGNORE_TESTS:
            return
        STD = 1
        pcollection = PCollectionMixture(
            mean_arr=np.array([[0]]),
            covariance_arr=np.array([[[STD**2]]]),
            weight_arr=np.array([1.0])
        )
        sample_arr = self.random_mixture.generateSample(pcollection, 500) 
        random_empirical = RandomEmpirical(total_num_sample=500)
        random_empirical.makePCollection(sample_arr)
        cdf = random_empirical.makeCDF(sample_arr)
        self.assertTrue(isinstance(cdf.variate_arr, np.ndarray))
        self.assertTrue(isinstance(cdf.cdf_arr, np.ndarray))
        self.assertEqual(cdf.variate_arr.shape[0], cdf.cdf_arr.shape[0])
        if IS_PLOT:
            plt.step(cdf.variate_arr, cdf.cdf_arr)
            plt.title('CDF of Empirical Distribution')
            plt.xlabel('Variate')
            plt.ylabel('Cumulative Probability')
            plt.grid()
            plt.show()  

    def testMakeCDFBivariate(self):
        """Test the creation of CDF from variate array."""
        if IGNORE_TESTS:
            return
        STD = 2
        matrix = np.array([[STD, 0.1], [0.1, STD]])  # Identity matrix for covariance
        pcollection = PCollectionMixture(
            mean_arr=np.array([[0, 5]]),
            covariance_arr=np.array([matrix]),
            weight_arr=np.array([1.0])
        )
        sample_arr = self.random_mixture.generateSample(pcollection, 500) 
        random_empirical = RandomEmpirical(total_num_sample=500)
        random_empirical.makePCollection(sample_arr)
        cdf = random_empirical.makeCDF(sample_arr)
        self.assertTrue(isinstance(cdf.variate_arr, np.ndarray))
        self.assertTrue(isinstance(cdf.cdf_arr, np.ndarray))
        self.assertEqual(cdf.variate_arr.shape[0], cdf.cdf_arr.shape[0])
        if IS_PLOT:
            plt.scatter(cdf.variate_arr[:, 0], cdf.variate_arr[:, 1], cdf.cdf_arr,  c='blue')
            plt.grid()
            plt.show()  

if __name__ == '__main__':
    unittest.main()