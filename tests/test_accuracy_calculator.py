from iplane.accuracy_calculator import AccuracyCalculator, AccuracyResult, PERCENTILE_STATISTICS  # type: ignore

import numpy as np  # type: ignore
import pandas as pd    # type: ignore
import unittest

IGNORE_TESTS = False
IS_PLOT = False
ERROR_ARR = np.random.rand(1000)  # Simulated fractional errors


class TestAccuracyCalculator(unittest.TestCase):

    def setUp(self):
        self.calculator = AccuracyCalculator(ERROR_ARR)

    def testConstructor(self):
        if IGNORE_TESTS:
            return
        self.assertIsInstance(self.calculator, AccuracyCalculator)
        self.assertIsInstance(self.calculator.error_arr, np.ndarray)

    def testCalculateCDF(self):
        if IGNORE_TESTS:
            return
        result = self.calculator.calculateCDF()
        self.assertIsInstance(result, AccuracyResult)
        self.assertIsInstance(result.cdf_df, pd.DataFrame)

    def testGetStatistics(self):
        if IGNORE_TESTS:
            return
        result = AccuracyCalculator.getStatistics(ERROR_ARR)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(PERCENTILE_STATISTICS))
        for idx in range(len(result)):
            self.assertIsInstance(result[idx], float)
            self.assertGreaterEqual(result[idx], 0.0)
            self.assertLessEqual(result[idx], 1.0)
            if idx > 0:
                self.assertGreaterEqual(result[idx], result[idx - 1])

    def testPlotCDF(self):
        if IGNORE_TESTS or not IS_PLOT:
            return
        self.calculator.plotCDF(is_plot=IS_PLOT)
        # Check if the plot was created without errors
        self.assertTrue(True)

    def testPlotCDFComparison(self):
        if IGNORE_TESTS:
            return
        error_arr = np.random.normal(0, 1, 1000)  # Simulated fractional errors
        other_calculator = AccuracyCalculator(error_arr)
        self.calculator.plotCDFComparison(other_calculator,
                names=['uniform', 'normal'], is_plot=IS_PLOT)


if __name__ == '__main__':
    unittest.main()