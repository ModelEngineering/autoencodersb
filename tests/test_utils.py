import autoencodersb.utils as utils  # type: ignore
import utils_test # type: ignore
import autoencodersb.constants as cn  # type: ignore

import numpy as np  # type: ignore
import pandas as pd    # type: ignore
import unittest

IGNORE_TESTS = False
IS_PLOT = False


class TestFunction(unittest.TestCase):

    def testDataloaderToDataFrame(self):
        if IGNORE_TESTS:
            return
        num_independent_feature = 3
        num_dependent_feature = 3
        dataloader, _ = utils_test.makeAutocoderData(num_independent_feature=num_independent_feature,
                num_dependent_feature=num_dependent_feature)
        df = utils.dataloaderToDataframe(dataloader)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df.columns), num_independent_feature + num_dependent_feature)
        self.assertEqual(len(df), len(dataloader.dataset))  # type: ignore

    def testCalculateMaximumRelativeError(self):
        if IGNORE_TESTS:
            return
        reference_arr = np.array([[1, 2, 3], [4, 5.5, 6]])
        target_arr = np.array([[1.5, 2.1, 3], [4, 5, 3]])
        max_rel_error_arr = utils.calculateMaximumRelativeError(reference_arr, target_arr)
        expected = np.array([0.5, -0.5])
        np.testing.assert_array_equal(max_rel_error_arr, expected)

if __name__ == '__main__':
    unittest.main()