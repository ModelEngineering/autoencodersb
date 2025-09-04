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
        self.assertTrue(np.allclose(max_rel_error_arr, expected, atol=1e-3))

    def testGetLocalURL(self):
        if IGNORE_TESTS:
            return
        local_file_contents = utils.getLocalURL(model_num=895)
        self.assertIsInstance(local_file_contents, str)
        #
        url = "https://www.ebi.ac.uk/biomodels/services/download/get-files/MODEL3352181362/3/BIOMD0000000206_url.xml"
        local_file_contents = utils.getLocalURL(url=url)
        if local_file_contents is None:
            print("Got None!")
        else:
            self.assertIsInstance(local_file_contents, str)
        #
        url = "https://www.ebi.ac.uk/biomodels/services/download/get-files/MODEL3352181362/3/BIOMD000000020A_url.xml"
        local_file_contents = utils.getLocalURL(url=url)
        self.assertIsNone(local_file_contents, str)

if __name__ == '__main__':
    unittest.main()