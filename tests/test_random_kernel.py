from iplane.random_kernel import RandomKernel, PCollectionKernel, DCollectionKernel  # type: ignore
from iplane.random_mixture import RandomMixture  # type: ignore
from iplane.random_mixture_collection import PCollectionMixture, DCollectionMixture  # type: ignore
import iplane.constants as cn  # type: ignore

from collections import namedtuple
import itertools
from matplotlib import pyplot as plt  # type: ignore
import numpy as np
from scipy.stats import gaussian_kde  # type: ignore
from typing import Tuple, cast
import unittest

IGNORE_TESTS = True
IS_PLOT = True
NUM_SAMPLE = 100
mixture = RandomMixture()
MEAN = 100
STD = 4
MEAN_ARR = np.array([[MEAN]])
COVARIANCE_ARR = np.array([[[STD**2]]])
PCOLLECTION_MIXTURE = PCollectionMixture(
    mean_arr=MEAN_ARR,
    covariance_arr=COVARIANCE_ARR,
    weight_arr=np.array([1])
)
TRAINING_ARR = mixture.generateSample(PCOLLECTION_MIXTURE, NUM_SAMPLE)
TRAINING2_ARR = mixture.generateSample(PCOLLECTION_MIXTURE, NUM_SAMPLE)
PCOLLECTION_BIVARIATE_MIXTURE = PCollectionMixture(
    mean_arr=np.array([[MEAN, MEAN]]),
    covariance_arr=np.array([[[STD**2, 0], [0, STD**2]]]),
    weight_arr=np.array([1.0])
)
BIVARIATE_TRAINING_ARR = mixture.generateSample(PCOLLECTION_BIVARIATE_MIXTURE, NUM_SAMPLE)
GAUSSIAN_SAMPLE_ARR = np.random.normal(loc=5, scale=STD, size=NUM_SAMPLE).reshape(-1, 1)  # Sample for testing
UNIFORM_SAMPLE_ARR = np.random.uniform(low=0, high=8, size=NUM_SAMPLE).reshape(-1, 1)  # Uniform sample for testing


BivariateResult = namedtuple('BivariateResult', ['kernel', 'sample_arr', 'cdf', 'pcollection'])


####################################
class TestRandomMixturePCollection(unittest.TestCase):

    def setUp(self):
        random_kernel = RandomKernel()
        self.pcollection = random_kernel.makePCollection(TRAINING_ARR)

    def testMakePCollection(self):
        if IGNORE_TESTS:
            return
        self.assertTrue(np.all(self.pcollection.get(cn.PC_TRAINING_ARR) == TRAINING_ARR))
        self.assertTrue(isinstance(self.pcollection.get(cn.PC_KDE), gaussian_kde))

    def test_num_dimension(self):
        if IGNORE_TESTS:
            return
        self.assertEqual(self.pcollection.num_dimension, 1)
        #
        training_arr = np.array([TRAINING_ARR.flatten(), TRAINING2_ARR.flatten()]).T
        random_kernel = RandomKernel()
        pcollection = random_kernel.makePCollection(training_arr)
        self.assertEqual(pcollection.num_dimension, 2)


####################################
class TestRandomMixture(unittest.TestCase):

    def setUp(self):
        self.random_kernel = RandomKernel()
        self.training_arr = np.array([TRAINING_ARR.flatten(), TRAINING2_ARR.flatten()]).T
        self.pcollection = self.random_kernel.makePCollection(self.training_arr)
        self.num_sample = 80
        arr = np.linspace(MEAN - 3*STD, MEAN + 3*STD, self.num_sample)
        self.dx = np.mean(np.diff(arr))
        self.variate_arr = np.array(list(itertools.product(arr, arr)))

    def testPredict(self):
        if IGNORE_TESTS:
            return
        density = self.random_kernel.predict(self.variate_arr, self.pcollection)
        if IS_PLOT:
            plt.scatter(self.variate_arr[:, 0], self.variate_arr[:, 1], 100*density, alpha=0.6)
            plt.show()
        integral = np.sum(density)*self.dx*self.dx
        self.assertAlmostEqual(integral, 1.0, places=1)

    def testMakeDCollectionGaussian(self):
        if IGNORE_TESTS:
            return
        random_kernel = RandomKernel()
        pcollection = random_kernel.makePCollection(BIVARIATE_TRAINING_ARR)
        dcollection = self.random_kernel.makeDCollection(
            variate_arr=self.variate_arr,
            pcollection=pcollection
        )
        entropy = dcollection.get(cn.DC_ENTROPY)
        random_mixture = RandomMixture()
        calculated_entropy = random_mixture.calculateEntropy(PCOLLECTION_BIVARIATE_MIXTURE)
        self.assertTrue(np.abs(entropy - calculated_entropy) < 0.5)
    
    def testMakeDCollectionUniform(self, num_sample=100):
        if IGNORE_TESTS:
            return
        num_sample = 100
        random_kernel = RandomKernel(num_variate_sample=num_sample)
        lower, upper = 8, 24 # Difference is 16
        training_arr = np.random.uniform(low=lower, high=upper, size=(num_sample, 1))
        calculated_entropy = np.log2(upper - lower)   #  For uniform distribution in 2D
        pcollection = random_kernel.makePCollection(training_arr)
        variate_arr=np.linspace(lower, upper, num_sample).reshape(-1, 1)
        dcollection = random_kernel.makeDCollection(
            variate_arr=variate_arr,  # type: ignore
            pcollection=pcollection
        )
        density_arr = dcollection.get(cn.DC_DENSITY_ARR)
        entropy = dcollection.get(cn.DC_ENTROPY)
        if IS_PLOT:
            plt.scatter(variate_arr[:, 0], density_arr)
            plt.show()
        self.assertTrue(np.abs(entropy - calculated_entropy) < 0.5)

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
        random_mixture = RandomMixture()
        sample_arr = random_mixture.generateSample(pcollection, 500) 
        random_kernel = RandomKernel(num_variate_sample=500)
        random_kernel.makePCollection(sample_arr)
        cdf = random_kernel.makeCDF(sample_arr)
        self.assertTrue(isinstance(cdf.variate_arr, np.ndarray))
        self.assertTrue(isinstance(cdf.cdf_arr, np.ndarray))
        self.assertEqual(cdf.variate_arr.shape[0], cdf.cdf_arr.shape[0])
        if IS_PLOT:
            plt.scatter(cdf.variate_arr.flatten(), cdf.cdf_arr)
            plt.title('CDF of Empirical Distribution')
            plt.xlabel('Variate')
            plt.ylabel('Cumulative Probability')
            plt.grid()
            plt.show()

    def makeBivariate(self, std=2, num_sample=500)->BivariateResult:
        matrix = np.array([[std, 0], [0, std]])  # Identity matrix for covariance
        pcollection = PCollectionMixture(
            mean_arr=np.array([[0, 0]]),
            covariance_arr=np.array([matrix]),
            weight_arr=np.array([1.0])
        )
        random_mixture = RandomMixture()
        sample_arr = random_mixture.generateSample(pcollection, num_sample=num_sample)
        random_kernel = RandomKernel(num_variate_sample=num_sample)
        pcollection_kernel = random_kernel.makePCollection(sample_arr)
        cdf = random_kernel.makeCDF(sample_arr)
        return BivariateResult(kernel=random_kernel, sample_arr=sample_arr, cdf=cdf, pcollection=pcollection_kernel)

    def testMakeCDFBivariate(self):
        """Test the creation of CDF from variate array."""
        if IGNORE_TESTS:
            return
        bivariate_result = self.makeBivariate()
        _, cdf = bivariate_result.kernel, bivariate_result.cdf
        self.assertTrue(isinstance(cdf.variate_arr, np.ndarray))
        self.assertTrue(isinstance(cdf.cdf_arr, np.ndarray))
        self.assertEqual(cdf.variate_arr.shape[0], cdf.cdf_arr.shape[0])
        if IS_PLOT:
            plt.scatter(cdf.variate_arr[:, 0], cdf.variate_arr[:, 1], 10*cdf.cdf_arr,  c='blue')
            plt.grid()
            plt.show()

    def testFindVariate(self):
        """Test the find variate method."""
        if IGNORE_TESTS:
            return
        def test(cdf_val:float, num_sample:int = 1000):
            bivariate_result = self.makeBivariate(num_sample=num_sample)
            random_kernel, cdf = bivariate_result.kernel, bivariate_result.cdf
            point = random_kernel._findVariate(cdf, cdf_val)
            estimated_cdf_val = random_kernel.calculateCDFValue(point, cdf.variate_arr)
            self.assertTrue(isinstance(point, np.ndarray))
            self.assertEqual(point.shape, (2,))
            self.assertTrue(np.isclose(estimated_cdf_val, cdf_val, atol=1e-1))
        #
        #test(0.1)
        test(0.5)
        #test(0.9)

    def testGenerateSample(self):
        """Test the generation of samples from the empirical distribution."""
        if IGNORE_TESTS:
            return
        def test(num_variate_sample:int):
            bivariate_result = self.makeBivariate(num_sample=num_variate_sample)
            random_kernel, pcollection = bivariate_result.kernel, bivariate_result.pcollection
            dcollection = random_kernel.makeDCollection(pcollection=pcollection)
            sample_arr = random_kernel.generateSample(pcollection, num_variate_sample)
            sample_arr_entropy = RandomKernel.estimateEntropy(sample_arr, num_variate_sample=num_variate_sample)
            dcollection_entropy = dcollection.get(cn.DC_ENTROPY)
            self.assertTrue(np.abs(sample_arr_entropy - dcollection_entropy) < 1)
            if IS_PLOT:
                _, ax = plt.subplots(1, 1, figsize=(8, 6))
                ax.scatter(sample_arr[:, 0], sample_arr[:, 1], alpha=1, color='g')
                ax.scatter(bivariate_result.sample_arr[:, 0], bivariate_result.sample_arr[:, 1], alpha=0.5, color='y')
                ax.set_title(f'Sample from Empirical Distribution (n={num_variate_sample})')
                ax.set_xlabel('X-axis')
                ax.set_ylabel('Y-axis')
                ax.grid()
                plt.show()
        #
        test(100)
    
    def testMakeMarginal(self):
        if IGNORE_TESTS:
            return
        ##
        def plotMarginal(random, dimension:int, ax=None):
            marginal_random = random.makeMarginal(dimensions=[dimension])
            variate_arr, density_arr, entropy, _ = marginal_random.dcollection.getAll()
            if IS_PLOT:
                if ax is None:
                    _, ax = plt.subplots(1, 1)
                ax.scatter(variate_arr, density_arr)
                ax.set_title(f"Marginal Density for Dimension {dimension}")
                ax.set_xlabel("Value")
                ax.set_ylabel("Density")
            return marginal_random, ax
        ##
        num_component = 3
        num_dim = 2
        variance = 1
        covariance_arr = np.zeros((num_dim, num_dim))
        diagonal_arr = np.array([(n+1)*variance for n in range(num_dim)])
        np.fill_diagonal(covariance_arr, diagonal_arr)
        component_covariance_arr = np.array([covariance_arr]*num_component)
        mean_arr = np.array([np.array(range(num_dim))]*num_component)
        weight_arr = np.repeat(1/num_component, num_component)
        pcollection = PCollectionMixture(
            mean_arr = mean_arr,
            covariance_arr = component_covariance_arr,
            weight_arr = weight_arr,
        )
        random = RandomMixture()
        random.pcollection = pcollection
        _ = random.makeDCollection(pcollection=pcollection)
        marginal1_random = random.makeMarginal(dimensions=[1])
        marginal0_random, ax = plotMarginal(random, dimension=0)
        marginal1_random, _ = plotMarginal(random, dimension=1, ax=ax)
        entropy0 = marginal0_random.dcollection.get(cn.DC_ENTROPY)
        entropy1 = marginal1_random.dcollection.get(cn.DC_ENTROPY)
        self.assertGreater(entropy1, entropy0)
        plt.show()

    def testMakeMutualInformation(self):
        """Test the mutual information calculation between two Random instances."""
        if IGNORE_TESTS:
            return
        mean_arr = [0, 5]
        error_mean_arr = [0, 0]
        covariance_arr = np.array([[1, 0.5], [0.5, 1]])
        error_covariance_arr = np.array([[0.1, 0], [0, 0.1]])
        sample_arr1 = np.random.multivariate_normal(mean_arr, covariance_arr, size=100)
        sample_arr2 = 4*sample_arr1 + np.random.multivariate_normal(error_mean_arr, error_covariance_arr, size=100)
        mutual_info = RandomKernel.makeNormalizedMutualInformation(sample_arr1, sample_arr2, min_num_dimension_coordinate=3)
        self.assertAlmostEqual(mutual_info, 1.0, places=1)
    
    def testMakeMutualInformationDiscrete(self):
        """Test the mutual information calculation between two Random instances."""
        #if IGNORE_TESTS:
        #    return
        mean_arr = [0, 5]
        covariance_arr = np.array([[1, 0], [0, 1]])
        sample_arr1 = np.random.multivariate_normal(mean_arr, covariance_arr, size=100000)
        sample_arr2 = np.array([(x > 0) and (y > 5) for x, y in sample_arr1])
        mutual_info = RandomKernel.makeNormalizedMutualInformation(sample_arr1, sample_arr2.reshape(-1, 1), min_num_dimension_coordinate=3)
        print(f"Mutual Information: {mutual_info}")

if __name__ == '__main__':
    unittest.main()