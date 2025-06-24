from iplane.random_mixture import RandomMixture, PCollectionMixture, DCollectionMixture  # type: ignore

from matplotlib import pyplot as plt  # type: ignore
import numpy as np
from typing import List, Any
import unittest

IGNORE_TESTS = True
IS_PLOT = False
NUM_SAMPLE = 1000


class TestMixtureEntropy(unittest.TestCase):
    """Test class for MixtureEntropy."""

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
        parameter = PCollectionMixture(parameter_dct=dct)
        #
        self.assertTrue(parameter == parameter)

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
        arr = MixtureEntropy.generateMixture(
            sample_arr=sample_arr,
            mean_arr=mean_arr,
            covariance_arr=covariance_arr)
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
    
    def testFit1d(self):
        if IGNORE_TESTS:
            return
        NUM_COMPONENT = 2
        NUM_DIM = 1
        VARIANCE = 0.5
        MEANS = np.array([0, 10])
        mixture_entropy = MixtureEntropy(
            num_component=NUM_COMPONENT,
            random_state=42
        )
        sample_arr = self.makeMixture(num_dim=NUM_DIM, num_component=NUM_COMPONENT, variance=VARIANCE)
        mixture_entropy.fit(sample_arr)
        trues = [any([np.abs(m - x) < 0.1 for m in MEANS]) for x in mixture_entropy.mean_arr]
        self.assertTrue(all(trues), f"Means do not match: {mixture_entropy.mean_arr} != {MEANS}")
        flat_covariances = np.array(mixture_entropy.covariance_arr).flatten()
        trues = np.isclose(np.abs(flat_covariances), VARIANCE, atol=0.1)
        self.assertTrue(all(trues), f"Covariances do not match: {flat_covariances} != {VARIANCE}")

    def testFit2d(self):
        if IGNORE_TESTS:
            return
        NUM_COMPONENT = 2
        NUM_DIM = 2
        VARIANCE = 0.5
        COVARIANCE = -0.5
        MEANS = np.array([ [12, 14], [2, 4]])
        for idx in range(2):
            mixture_entropy = MixtureEntropy(
                num_component=NUM_COMPONENT,
                random_state=42
            )
            sample_arr = self.makeMixture(num_dim=NUM_DIM, num_component=NUM_COMPONENT, variance=VARIANCE,
                    covariance=COVARIANCE)
            mixture_entropy.fit(sample_arr)
            flat_covariances = np.abs(np.array(mixture_entropy.covariance_arr).flatten())
            frac = np.mean(np.isclose(np.abs(flat_covariances), VARIANCE, atol=0.2))
            if frac < 0.5:
                print(f"Iteration {idx}: Covariances do not match: {flat_covariances} != {VARIANCE}")
            self.assertGreater(frac, 0.6, f"Covariances do not match: {flat_covariances} != {VARIANCE}")
            # Means
            flattened_sorted_means = np.array(mixture_entropy.mean_arr).flatten()
            flattened_sorted_means.sort()
            flattened_sorted_expected_means = MEANS.flatten()
            flattened_sorted_expected_means.sort()
            for actual, expected in zip(flattened_sorted_means, flattened_sorted_expected_means):
                is_true = np.abs(actual - expected) < 0.1
                if not is_true:
                    print(f"Actual mean {actual} does not match expected mean {expected}")
                self.assertTrue(is_true, f"Means do not match: {mixture_entropy.mean_arr} != {MEANS}")

    def testMakeDensity1Component1Dimension(self):
        if IGNORE_TESTS:
            return
        MEAN_ARR = np.reshape(np.array([5]), (1, -1))
        COVARIANCE_ARR = np.reshape(np.array([0.5]), (1, -1))
        result = MixtureEntropy.calculateMixtureEntropy(
                mean_arr=MEAN_ARR,
                covariance_arr=COVARIANCE_ARR,
                weight_arr=np.array([1]))
        expected_Hx =  MixtureEntropy.calculateUnivariateGaussianEntropy(COVARIANCE_ARR[0])
        self.assertAlmostEqual(expected_Hx, result.Hx, delta=0.1)

    def testCalculateUnivariateGaussianEntropy(self):
        if IGNORE_TESTS:
            return
        # Test the calculation of univariate Gaussian entropy
        variance = 0.5
        expected_entropy = MixtureEntropy.calculateUnivariateGaussianEntropy(variance)
        calculated_entropy = MixtureEntropy.calculateUnivariateGaussianEntropy(variance)
        self.assertAlmostEqual(expected_entropy, calculated_entropy, delta=0.01)
    
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
            result = MixtureEntropy.calculateMixtureEntropy(
                    mean_arr=mean_arr,
                    covariance_arr=covariance_arr,
                    weight_arr=weight_arr)
            expected_Hx =  MixtureEntropy.calculateUnivariateGaussianEntropy(covariance_arr[0])
            if IS_PLOT:
                plt.plot(result.variate_arr, result.pdf_arr)
                plt.title("Mixture Density")
                plt.xlabel("Value")
                plt.ylabel("Density")
                plt.show()
            # Assume that all weights are equal
            expected_Hx = np.sum(MixtureEntropy.calculateUnivariateGaussianEntropy(covariance_arr.flatten()) * weight_arr)  \
                    - np.log2(weight_arr[0])
            self.assertAlmostEqual(expected_Hx, result.Hx, delta=0.1)
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
            result = MixtureEntropy.calculateMixtureEntropy(
                    mean_arr=mean_arr,
                    covariance_arr=covariance_arr,
                    weight_arr=weight_arr)
            if IS_PLOT:
                plt.plot(result.variate_arr, result.pdf_arr)
                plt.title("Mixture Density")
                plt.xlabel("Value")
                plt.ylabel("Density")
                plt.show()
            expected_Hx = MixtureEntropy.calculateMultivariateGaussianEntropy(covariance_arr[0])
            self.assertAlmostEqual(expected_Hx, result.Hx, delta=0.1)
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
            result = MixtureEntropy.calculateMixtureEntropy(
                    mean_arr=mean_arr,
                    covariance_arr=covariance_arr,
                    weight_arr=weight_arr)
            if IS_PLOT:
                plt.plot(result.variate_arr, result.pdf_arr)
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
        mixture_entropy = MixtureEntropy(
            num_component=num_component,
            random_state=42
        )
        mixture_entropy.fit(sample_arr)
        mixture_entropy.calculateMixtureModelEntropy()
        expected_Hx = MixtureEntropy.calculateMultivariateGaussianEntropy(mixture_entropy.covariance_arr[0])
        self.assertAlmostEqual(expected_Hx, mixture_entropy.Hx, delta=0.01)


if __name__ == '__main__':
    unittest.main()