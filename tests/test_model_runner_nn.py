from iplane.dataset_csv import DatasetCSV # type: ignore
from iplane.model_runner_nn import ModelRunnerNN, RunnerResultPredict  # type: ignore
from iplane.model_runner import RunnerResult  # type: ignore
from iplane.autoencoder import Autoencoder  # type: ignore
import iplane.constants as cn  # type: ignore

import numpy as np  # type: ignore
import torch
from torch.utils.data import DataLoader
import pandas as pd    # type: ignore
import unittest

IGNORE_TESTS = False
IS_PLOT = False
NUM_EPOCH = 2500
NUM_EPOCH = 5

TARGET_COLUMN = "target"  # Assuming the target column is named 'target'
NUM_INPUT_FEATURE = 5
NUM_OUTPUT_FEATURE = 2
NUM_SAMPLE = 1000
MODEL = Autoencoder(layer_dimensions=[NUM_INPUT_FEATURE,
        10*NUM_INPUT_FEATURE, 10*NUM_OUTPUT_FEATURE, NUM_OUTPUT_FEATURE])  # Example model
##
def makeData(num_sample:int=NUM_SAMPLE, num_feature:int=NUM_INPUT_FEATURE) -> DataLoader:
    """Generate a DataFrame with random data."""
    arr = np.random.randint(1, 11, (num_sample, num_feature)).astype(np.float32)
    # Make the arrays
    arrs = [arr[:, 0], arr[:, 1], arr[:, 0]**2, arr[:, 0]**2, arr[:, 0]*arr[:, 1]]
    # Select columns
    big_arr = np.array(arrs)[:num_feature].T
    columns = [f"feature{i+1}" for i in range(big_arr.shape[1])]
    # Make the dataloader
    df = pd.DataFrame(big_arr, columns=columns, dtype=np.float32)
    dataloader = DataLoader(DatasetCSV(csv_input=df, target_column=None), batch_size=10)
    return dataloader
##
TEST_DL = makeData(num_sample=1000, num_feature=NUM_INPUT_FEATURE)
TRAIN_DL = makeData(num_sample=NUM_SAMPLE, num_feature=NUM_INPUT_FEATURE)


class TestModelRunner(unittest.TestCase):

    def setUp(self):
        self.runner = ModelRunnerNN(model=MODEL, num_epoch=NUM_EPOCH,
                learning_rate=1e-5, is_normalized=True)

    def testTrain(self):
        if IGNORE_TESTS:
            return
        result = self.runner.fit(TRAIN_DL)
        self.assertIsInstance(result, RunnerResultPredict)
        self.assertIsInstance(result.losses, list)
    
    def testPredict(self):
        if IGNORE_TESTS:
            return
        _ = self.runner.fit(TRAIN_DL)
        feature_tnsr = [x[0] for x in TRAIN_DL][0]
        prediction_tnsr = self.runner.predict(feature_tnsr)
        self.assertIsInstance(prediction_tnsr, torch.Tensor)

    def testEvaluate(self):
        if IGNORE_TESTS:
            return
        _ = self.runner.fit(TRAIN_DL)
        evaluate_result = self.runner.evaluate(TEST_DL)
        self.assertIsInstance(evaluate_result, RunnerResult)
        self.assertIsInstance(evaluate_result.losses, list)
    
    def testplotEvaluate(self):
        if IGNORE_TESTS:
            return
        _ = self.runner.fit(TRAIN_DL)
        self.runner.plotEvaluate(TEST_DL, is_plot=IS_PLOT)

if __name__ == '__main__':
    unittest.main()