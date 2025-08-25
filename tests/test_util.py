import autoencodersb.util as util  # type: ignore
import utils_test # type: ignore
import autoencodersb.constants as cn  # type: ignore

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
        df = util.dataloaderToDataframe(dataloader)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df.columns), num_independent_feature + num_dependent_feature)
        self.assertEqual(len(df), len(dataloader.dataset))  # type: ignore

if __name__ == '__main__':
    unittest.main()