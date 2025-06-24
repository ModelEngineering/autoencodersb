from iplane.random_mixture import RandomMixture  # type: ignore
import iplane.constants as cn  # type: ignore
from iplane.random_mixture_collection import PCollectionMixture, DCollectionMixture  # type: ignore

from matplotlib import pyplot as plt  # type: ignore
import numpy as np
from typing import List, Any, Tuple
import unittest

IGNORE_TESTS = True
IS_PLOT = False
NUM_SAMPLE = 1000


class TestRandomMixture(unittest.TestCase):

    def setUp(self):
        self.random = RandomMixture()

    def makeMixture(self,
            num_component:int=2,
            component_sample_size:int=NUM_SAMPLE,
            num_dim:int=1,
            variance:float=0.5,
            covariance:float=0.0,  # Between non-identical dimensions
    )-> np.ndarray:
        # Size of sample for the ith component is 100 less than i+1st
        sample_arr = np.array([n*component_sample_size for n in range(1, num_component+1)])
        means:List[Any] = []
        covariances:List[Any] = []
        for n_component in range(num_component):
            if num_dim > 1:
                means.append([10*n_component + 2*n_dim for n_dim in range(1, num_dim + 1)])
                # Covariances
                matrix = np.repeat(covariance, num_dim*num_dim)
                matrix = np.reshape(matrix, (num_dim, num_dim))
                np.fill_diagonal(matrix, variance)
                covariances.append(matrix)
            else:
                covariances.append(variance)
                means.append(10 * n_component)
        mean_arr = np.array(means)
        covariance_arr = np.array(covariances)
        pcollection = PCollectionMixture(mean_arr= mean_arr, covariance_arr=covariance_arr,
                weight_arr= np.repeat(1/num_component, num_component))
        arr = self.random.generateSample(pcollection=pcollection, num_sample=NUM_SAMPLE)
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
        self.assertEqual(sample_arr.shape[0], 3*NUM_SAMPLE)

    def testGenerateMixture1Dim2Components(self):
        """Test the generateMixture method."""
        if IGNORE_TESTS:
            return
        num_dim = 2
        num_component = 3
        sample_arr = self.makeMixture(num_dim=num_dim, variance=0.5, covariance=-0.5, num_component=num_component)
        self.assertEqual(sample_arr.shape[0], 6*NUM_SAMPLE)
        self.assertEqual(sample_arr.shape[1], num_dim)
    
    def testEstimatePCollection1d(self):
        if IGNORE_TESTS:
            return
        NUM_COMPONENT = 2
        NUM_DIM = 1
        VARIANCE = 0.5
        MEANS = np.array([0, 10])
        random = RandomMixture(num_component=NUM_COMPONENT, random_state=42)
        sample_arr = self.makeMixture(num_dim=NUM_DIM, num_component=NUM_COMPONENT, variance=VARIANCE)
        pcollection = random.estimatePCollection(sample_arr)
        mean_arr, covariance_arr, weight_arr = pcollection.getAll()
        trues = [any([np.abs(m - x) < 0.1 for m in MEANS]) for x in mean_arr]
        self.assertTrue(all(trues), f"Means do not match: {mean_arr} != {MEANS}")
        flat_covariances = np.array(covariance_arr).flatten()
        trues = np.isclose(np.abs(flat_covariances), VARIANCE, atol=0.1)
        self.assertTrue(all(trues), f"Covariances do not match: {flat_covariances} != {VARIANCE}")

    def testEstimatePCollection2d(self):
        if IGNORE_TESTS:
            return
        NUM_COMPONENT = 2
        NUM_DIM = 2
        VARIANCE = 0.5
        COVARIANCE = -0.5
        MEANS = np.array([ [12, 14], [2, 4]])
        for idx in range(2):
            random = RandomMixture(num_component=NUM_COMPONENT, random_state=42)
            sample_arr = self.makeMixture(num_dim=NUM_DIM, num_component=NUM_COMPONENT, variance=VARIANCE,
                    covariance=COVARIANCE)
            pcollection = random.estimatePCollection(sample_arr)
            mean_arr, covariance_arr, weight_arr = pcollection.getAll()
            flat_covariances = np.abs(np.array(covariance_arr).flatten())
            frac = np.mean(np.isclose(np.abs(flat_covariances), VARIANCE, atol=0.2))
            if frac < 0.5:
                print(f"Iteration {idx}: Covariances do not match: {flat_covariances} != {VARIANCE}")
            self.assertGreater(frac, 0.6, f"Covariances do not match: {flat_covariances} != {VARIANCE}")
            # Means
            flattened_sorted_means = np.array(mean_arr).flatten()
            flattened_sorted_means.sort()
            flattened_sorted_expected_means = MEANS.flatten()
            flattened_sorted_expected_means.sort()
            for actual, expected in zip(flattened_sorted_means, flattened_sorted_expected_means):
                is_true = np.abs(actual - expected) < 0.1
                if not is_true:
                    print(f"Actual mean {actual} does not match expected mean {expected}")
                self.assertTrue(is_true, f"Means do not match: {mean_arr} != {MEANS}")

    def testMakeDistribution1Component1Dimension(self):
        if IGNORE_TESTS:
            return
        MEAN_ARR = np.reshape(np.array([5]), (1, -1))
        COVARIANCE_ARR = np.reshape(np.array([0.5]), (1, -1))
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
            pcollection = PCollectionMixture(covariance_arr=np.array([variance]))
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
        dcollection = self.random.makeDCollection(pcollection)
        variate_arr, density_arr, dx_arr, entropy = dcollection.getAll()
        return variate_arr, density_arr, entropy, pcollection
    
    def testMakeDensity2Component1Dimension(self):
        # Evaluates calculations for 1-dimensional mixture and multiple components to the mixture.
        if IGNORE_TESTS:
            return
        def test(num_component:int=2):
            mean_arr = np.array([(20*n)*0.5 for n in range(num_component)])
            mean_arr = np.reshape(mean_arr, (-1, 1))
            # Independent dimensions
            covariance_arr = np.array([(n+1)*0.1 for n in range(num_component)])
            covariance_arr = np.reshape(covariance_arr, (-1, 1))
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
        test(10)
        test(3)
    
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
        test(6)
        test(2)

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

    def testCalculateEntropy1Component2Dimension(self):
        # Test the calculation of entropy for a mixture distribution
        # Test is done for a single component so that an exact calculation can be done.
        if IGNORE_TESTS:
            return
        num_component = 1
        sample_arr = self.makeMixture(num_dim=5, num_component=num_component, variance=1.5)
        mixture_entropy = RandomMixture(
            num_component=num_component,
            random_state=42
        )
        pcollection = self.random.estimatePCollection(sample_arr)
        expected_entropy = self.random.calculateEntropy(pcollection)
        dcollection = self.random.makeDCollection(pcollection=pcollection)
        self.assertAlmostEqual(expected_entropy, dcollection.get(cn.DC_ENTROPY), delta=0.01)


if __name__ == '__main__':
    unittest.main()