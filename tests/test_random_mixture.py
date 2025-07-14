from iplane.random_mixture import RandomMixture  # type: ignore
import iplane.constants as cn  # type: ignore
from iplane.random_mixture_collection import PCollectionMixture, DCollectionMixture  # type: ignore

from matplotlib import pyplot as plt  # type: ignore
import numpy as np
from typing import List, Any, Tuple
import unittest

IGNORE_TESTS = False
IS_PLOT = False
NUM_SAMPLE = 1000


class TestRandomMixture(unittest.TestCase):

    def setUp(self):
        self.random = RandomMixture(total_num_sample=int(1e6), random_state=42)

    def makeMixture(self,
            num_component:int=2,
            component_sample_size:int=NUM_SAMPLE,
            num_dim:int=1,
            variance:float=0.5,
            covariance:float=0.0,  # Between non-identical dimensions
    )-> np.ndarray:
        pcollection = PCollectionMixture.make(
            num_component=num_component,
            num_dim=num_dim,
            variance=variance,
            covariance=covariance,
        )
        arr = self.random.generateSample(pcollection=pcollection, num_sample=component_sample_size)
        if IS_PLOT:
            if num_dim == 1:
                plt.hist(arr, bins=30)
                plt.show()
            else:
                for ndim in range(num_dim):
                    plt.figure()
                    plt.title(f"Dimension {ndim}")
                    plt.xlabel("Value")
                    plt.ylabel("Frequency")
                    plt.hist(arr[:, ndim], bins=30)
                plt.figure()
                plt.scatter(arr[:, 0], arr[:, 1], alpha=0.5)
                plt.show()
        return arr

    def testGenerateMixture1Dim1Component(self):
        """Test the generateMixture method."""
        if IGNORE_TESTS:
            return
        sample_arr = self.makeMixture(num_dim=1)
        self.assertEqual(sample_arr.shape[0], NUM_SAMPLE)

    def testGenerateMixture1Dim2Components(self):
        """Test the generateMixture method."""
        if IGNORE_TESTS:
            return
        num_dim = 2
        num_component = 3
        sample_arr = self.makeMixture(num_dim=num_dim, variance=0.5, covariance=-0.5, num_component=num_component)
        self.assertTrue(np.abs(sample_arr.shape[0] - NUM_SAMPLE) < 2)
        self.assertEqual(sample_arr.shape[1], num_dim)
    
    def testEstimatePCollection1d(self):
        if IGNORE_TESTS:
            return
        NUM_COMPONENT = 2
        NUM_DIM = 1
        VARIANCE = 0.5
        random = RandomMixture(num_component=NUM_COMPONENT, random_state=42)
        sample_arr = self.makeMixture(num_dim=NUM_DIM, num_component=NUM_COMPONENT, variance=VARIANCE,
                component_sample_size=10000)
        pcollection = random.makePCollection(sample_arr)
        _, covariance_arr, _ = pcollection.getAll()
        flat_covariances = np.array(covariance_arr).flatten()
        trues = np.isclose(np.abs(flat_covariances), VARIANCE, atol=0.5)
        self.assertTrue(all(trues), f"Covariances do not match: {flat_covariances} != {VARIANCE}")

    def testEstimatePCollection2d(self):
        if IGNORE_TESTS:
            return
        NUM_COMPONENT = 2
        NUM_DIM = 2
        VARIANCE = 0.5
        COVARIANCE = -0.5
        for idx in range(2):
            random = RandomMixture(num_component=NUM_COMPONENT, random_state=42)
            sample_arr = self.makeMixture(num_dim=NUM_DIM, num_component=NUM_COMPONENT, variance=VARIANCE,
                    covariance=COVARIANCE, component_sample_size=10000)
            pcollection = random.makePCollection(sample_arr)
            _, covariance_arr, __ = pcollection.getAll()
            flat_covariances = np.abs(np.array(covariance_arr).flatten())
            frac = np.mean(np.isclose(np.abs(flat_covariances), VARIANCE, atol=0.2))
            if frac < 0.5:
                print(f"Iteration {idx}: Covariances do not match: {flat_covariances} != {VARIANCE}")
            self.assertGreater(frac, 0.6, f"Covariances do not match: {flat_covariances} != {VARIANCE}")
            # Covariance should be negative for non-identical dimensions
            covariance_arr = pcollection.get(cn.PC_COVARIANCE_ARR)
            flat_covariance_arr = covariance_arr.flatten()
            self.assertTrue(np.allclose(np.abs(flat_covariance_arr), VARIANCE, atol=0.2))
            self.assertEqual(np.sum(covariance_arr < 0), NUM_COMPONENT * NUM_DIM)

    def testMakeDistribution1Component1Dimension(self):
        if IGNORE_TESTS:
            return
        MEAN_ARR = np.array([[5]])
        COVARIANCE_ARR = np.array([[[0.5]]])
        pcollection = PCollectionMixture(
            mean_arr=MEAN_ARR,
            covariance_arr=COVARIANCE_ARR,
            weight_arr=np.array([1])
        )
        dcollection = self.random.makeDCollection(pcollection=pcollection)
        expected_entropy = self.random.calculateEntropy(pcollection)
        self.assertAlmostEqual(dcollection.get(cn.DC_ENTROPY), expected_entropy, delta=0.1)

    def testCalculateUnivariateGaussianEntropy(self):
        if IGNORE_TESTS:
            return
        # Test the calculation of univariate Gaussian entropy
        ##
        def calculate(variance:float) -> float:
            weight_arr = np.array([1])
            mean_arr = np.array([[0]])
            pcollection = PCollectionMixture(
                mean_arr=mean_arr,
                covariance_arr=np.array([[[variance]]]),
                weight_arr=weight_arr,
                )
            return self.random.calculateEntropy(pcollection)
        ##
        variances = range(1, 10)
        last_entropy = calculate(variances[0])
        for variance in variances[1:]:
            entropy = calculate(variance)
            self.assertGreater(entropy, last_entropy, f"Entropy should increase with variance: {entropy} <= {last_entropy}")
            last_entropy = entropy

    def makeDistribution(self, mean_arr:np.ndarray, covariance_arr:np.ndarray, weight_arr:np.ndarray
            ) -> Tuple[np.ndarray, np.ndarray, float, PCollectionMixture]:
        """
        Creates a distribution from the given parameters and returns variate, density, and entropy.
        """
        pcollection = PCollectionMixture(
            mean_arr=mean_arr,
            covariance_arr=covariance_arr,
            weight_arr=weight_arr
        )
        dcollection = self.random.makeDCollection(pcollection=pcollection)
        variate_arr, density_arr, _, entropy = dcollection.getAll()
        return variate_arr, density_arr, entropy, pcollection
    
    def testMakeDensityNComponent1Dimension(self):
        # Evaluates calculations for 1-dimensional mixture and multiple components to the mixture.
        if IGNORE_TESTS:
            return
        def test(num_component:int=2):
            mean_arr = np.array([(100*n) for n in range(num_component)])
            mean_arr = np.reshape(mean_arr, (num_component, 1))
            # Independent dimensions
            covariance_arr = np.array([(n+1)*0.1 for n in range(num_component)])
            covariance_arr = np.reshape(covariance_arr, (num_component, 1, 1))
            weight_arr = np.repeat(1/num_component, num_component)
            variate_arr, density_arr, entropy, pcollection = self.makeDistribution(mean_arr=mean_arr,
                    covariance_arr=covariance_arr, weight_arr=weight_arr)
            if IS_PLOT:
                plt.plot(variate_arr, density_arr)
                plt.title("Mixture Density")
                plt.xlabel("Value")
                plt.ylabel("Density")
                plt.show()
            # Assume that all weights are equal
            expected_entropy = self.random.calculateEntropy(pcollection)
            self.assertAlmostEqual(expected_entropy, entropy, delta=0.1)
        ##
        test(1)
        test(3)
        test(8)
    
    def testMakeDensity1Component2Dimension(self):
        # Evaluates calculations for 1-dimensional mixture and multiple components to the mixture.
        if IGNORE_TESTS:
            return
        def test(num_dimension:int=2):
            mean_arr = np.array([(20*n)*0.5 for n in range(num_dimension)], dtype=float)
            mean_arr = np.reshape(mean_arr, (1, num_dimension))
            # Independent dimensions
            covariance_arr = np.zeros((num_dimension, num_dimension), dtype=float)
            diagonal_arr = np.array([(n+1)*0.1 for n in range(num_dimension)])
            np.fill_diagonal(covariance_arr, diagonal_arr)
            covariance_arr = np.reshape(covariance_arr, (1, num_dimension, num_dimension))
            weight_arr = np.array([1])
            variate_arr, density_arr, entropy, pcollection = self.makeDistribution(mean_arr=mean_arr,
                    covariance_arr=covariance_arr, weight_arr=weight_arr)
            if IS_PLOT:
                plt.plot(variate_arr, density_arr)
                plt.title("Mixture Density")
                plt.xlabel("Value")
                plt.ylabel("Density")
                plt.show()
            expected_entropy = self.random.calculateEntropy(pcollection)
            self.assertAlmostEqual(expected_entropy, entropy, delta=0.1)
        ##
        test(2)
        test(3)
        test(6)

    def testMakeDensity2Component2Dimension(self):
        # Evaluates calculations for 2-dimensional mixture and multiple components to the mixture.
        if IGNORE_TESTS:
            return
        def test(num_dimension:int=2, num_component:int=3):
            mean_arr = np.array([(20*n)*0.5 for n in range(num_dimension)], dtype=float)
            mean_arr = np.array([mean_arr + n*10 for n in range(num_component)])
            # Independent dimensions
            covariance_arr = np.zeros((num_dimension, num_dimension), dtype=float)
            diagonal_arr = np.array([(n+1)*0.1 for n in range(num_dimension)])
            np.fill_diagonal(covariance_arr, diagonal_arr)
            covariance_arr = np.array([covariance_arr]*num_component)
            weight_arr = np.repeat(1/num_component, num_component)
            variate_arr, density_arr, entropy, pcollection = self.makeDistribution(mean_arr=mean_arr,
                    covariance_arr=covariance_arr, weight_arr=weight_arr)
            if IS_PLOT:
                plt.plot(variate_arr, density_arr)
                plt.title("Mixture Density")
                plt.xlabel("Value")
                plt.ylabel("Density")
                plt.show()
            # Assume that all weights are equal
        ##
        test()

    def testCalculateEntropy2Component2Dimension(self):
        # Test the calculation of entropy for a mixture distribution
        # Test is done for a single component so that an exact calculation can be done.
        if IGNORE_TESTS:
            return
        num_component = 2
        sample_arr = self.makeMixture(num_dim=5, num_component=num_component, variance=0.1,
                component_sample_size=100000)
        pcollection = self.random.makePCollection(sample_arr)
        expected_entropy = self.random.calculateEntropy(pcollection)
        dcollection = self.random.makeDCollection(pcollection=pcollection)
        # Use a big delta on comparison because of the "roughness" of assuming non-overlapping distributions of components
        self.assertAlmostEqual(expected_entropy, dcollection.get(cn.DC_ENTROPY), delta=1)


if __name__ == '__main__':
    unittest.main()