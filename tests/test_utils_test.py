from tests import utils_test as util
import autoencodersb.constants as cn  # type: ignore

import matplotlib.pyplot as plt
import pandas as pd  # type: ignore
import numpy as np
from typing import List, Any, Tuple
import unittest

IGNORE_TESTS = True


########################################
class TestFunctions(unittest.TestCase):

    def setUp(self):
        pass

    def calculateFractionalError(self, actual: Any, expected: Any) -> float:
        return np.abs(actual - expected) / np.abs(expected) if expected != 0 else np.inf

    def testMakeAutocoderData(self):
        """Test the make_autocoder_data function."""
        if IGNORE_TESTS:
            return
        def test(num_independent_feature:int=2, num_dependent_feature:int=2,
                 **kwargs) -> Tuple[pd.DataFrame, dict]:
            """Test the make_autocoder_data function."""
            dataloader, dct = util.makeAutocoderData(num_independent_feature=num_independent_feature,
                    num_dependent_feature=num_dependent_feature, **kwargs)
            self.assertIsNotNone(dataloader)
            self.assertTrue(len(dataloader) > 0)
            df = dataloader.dataset.data_df # type: ignore
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(df.shape[0], kwargs.get("num_sample", df.shape[0]))
            #
            self.assertEqual(len(dct), len(df.columns))
            independent_features = df.columns[:num_independent_feature]
            dependent_features = df.columns[num_independent_feature:num_independent_feature + num_dependent_feature]
            for feature in independent_features:
                self.assertIn(feature, dct)
                self.assertTrue(dct[feature] == 1)
            for feature in dependent_features:
                self.assertIn(feature, dct)
                self.assertTrue(dct[feature] >0)
            return df, dct
        #
        _ = test(num_sample=6, num_independent_feature=2, num_dependent_feature=8, num_value=10)
        _ = test(num_sample=600, num_independent_feature=4, num_dependent_feature=5, num_value=10)
        #        # Test with noise
        noise_std = 1.0
        df, dct = test(num_sample=1000, num_independent_feature=2, num_dependent_feature=5, num_value=10,
                noise_std=noise_std)
        var_ser = df.var()
        frac = self.calculateFractionalError(var_ser['D_k_0'], var_ser['I_0']*dct['D_k_0']**2 + noise_std**2)
        self.assertTrue(frac < 0.5)
        frac = self.calculateFractionalError(var_ser['D_k_1'], var_ser['I_1']*dct['D_k_1']**2 + noise_std**2)
        self.assertTrue(frac < 0.5)

    def testMakeAutocoderDataWithReuseConstants(self):
        """Test the make_autocoder_data function with noise."""
        #if IGNORE_TESTS:
        #    return
        ##
        def test(num_independent_feature:int=4):
            """Test the make_autocoder_data function with noise."""
            dataloader, dct = util.makeAutocoderData(num_independent_feature=num_independent_feature,
                    num_dependent_feature=num_independent_feature, num_value=10, noise_std=0.0)
            df = dataloader.dataset.data_df # type: ignore
            dkeys = [k for k in df.columns if k.startswith('D_k_')]
            for dkey in dkeys:
                idx = int(dkey.split('_')[-1])
                ikey = f'I_{idx}'
                self.assertIn(ikey, df.columns)
                self.assertIn(dkey, df.columns)
                iunique = np.unique(df[ikey]*dct[dkey])
                dunique = np.unique(df[dkey])
                self.assertTrue(np.all(np.isin(iunique, dunique)))
                self.assertTrue(np.all(np.isin(dunique, iunique)))
        ##
        test(num_independent_feature=4) # First 2 dkeys are Michaelis-Menten
        test(num_independent_feature=8)


if __name__ == '__main__':
    unittest.main()