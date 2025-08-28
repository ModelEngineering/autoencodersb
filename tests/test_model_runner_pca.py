from autoencodersb.model_runner_pca import ModelRunnerPCA, RunnerResult  # type: ignore
import autoencodersb.constants as cn  # type: ignore
from autoencodersb.dataset_csv import DatasetCSV  # type: ignore
from tests.utils_test import makeAutocoderData  # type: ignore

import numpy as np  # type: ignore
from torch.utils.data import DataLoader
import pandas as pd    # type: ignore
from typing import cast
import unittest

IGNORE_TESTS = False
IS_PLOT = False
NUM_EPOCH = 5
NUM_EPOCH = 20000

TARGET_COLUMN = "target"  # Assuming the target column is named 'target'
NUM_DEPENDENT_FEATURE = 12
NUM_OUTPUT_FEATURE = 2
NUM_INPUT_FEATURE = NUM_DEPENDENT_FEATURE + NUM_OUTPUT_FEATURE
NUM_SAMPLE = 1000
df = pd.DataFrame({'I1': np.random.randint(1, 11, NUM_SAMPLE), 'I2': np.random.randint(1, 11, NUM_SAMPLE)})
df['D1'] = df['I1'] + df['I2']
dataset = DatasetCSV(df)
TRAIN_LINEAR_DL = DataLoader(dataset, batch_size=10)
TRAIN_DL = makeAutocoderData(num_sample=NUM_SAMPLE, num_independent_feature=NUM_OUTPUT_FEATURE,
        num_dependent_feature=NUM_DEPENDENT_FEATURE, num_value=10)
TEST_DL = makeAutocoderData(num_sample=1000, num_independent_feature=NUM_OUTPUT_FEATURE,
                            num_dependent_feature=NUM_DEPENDENT_FEATURE, num_value=10)


class TestModelRunnerPCA(unittest.TestCase):

    def setUp(self):
        self.runner = ModelRunnerPCA(n_components=2, random_state=42)

    def testFit(self):
        """Test fitting PCA model"""
        def test(num_dependent_feature: int = 2):
            # Make data
            df = pd.DataFrame({'I1': np.random.randint(1, 11, NUM_SAMPLE), 'I2': np.random.randint(1, 11, NUM_SAMPLE)})
            for i in range(num_dependent_feature):
                aaa = np.random.uniform(0, 1, 1)
                bbb = np.random.uniform(0, 1, 1)
                df[f'D{i+1}'] = aaa*df['I1'] + bbb*df['I2']
            dataset = DatasetCSV(df)
            train_loader = DataLoader(dataset, batch_size=10)
            result = self.runner.fit(train_loader)
            self.assertIsInstance(result, RunnerResult)
            self.assertTrue(cast(float, result.mean_absolute_error) < 1e-5)

        test(3)
        test(30)

if __name__ == '__main__':
    unittest.main()