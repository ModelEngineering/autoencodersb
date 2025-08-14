from tests import utils_test as util
import iplane.constants as cn  # type: ignore

import pandas as pd  # type: ignore
import numpy as np
from typing import List, Any, Tuple
import unittest

IGNORE_TESTS = False


########################################
class TestFunctions(unittest.TestCase):

    def setUp(self):
        pass

    def testMakeAutocoderData(self):
        """Test the make_autocoder_data function."""
        if IGNORE_TESTS:
            return
        def test(**kwargs) -> pd.DataFrame:
            """Test the make_autocoder_data function."""
            dataloader = util.makeAutocoderData(**kwargs)
            self.assertIsNotNone(dataloader)
            self.assertTrue(len(dataloader) > 0)
            df = dataloader.dataset.data_df # type: ignore
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(df.shape[0], kwargs.get("num_sample", df.shape[0]))
            return df
        #
        _ = test(num_sample=6, num_independent_feature=2, num_dependent_feature=8, num_value=10)
        _ = test(num_sample=600, num_independent_feature=10, num_dependent_feature=5, num_value=10)
        #        # Test with noise
        noise_std = 2.0
        df = test(num_sample=10000, num_independent_feature=2, num_dependent_feature=5, num_value=10,
                noise_std=noise_std, is_multiplier=False)
        var_ser = df.var()
        self.assertTrue(np.abs(var_ser['D_k_0'] - var_ser['I_0'] - noise_std**2) < 0.5)
        self.assertTrue(np.abs(var_ser['D_k_1'] - var_ser['I_1'] - noise_std**2) < 0.5)
        

if __name__ == '__main__':
    unittest.main()