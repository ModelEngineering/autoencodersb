from iplane.dataset_csv import DatasetCSV # type: ignore
from iplane.model_runner_nn import ModelRunnerNN, RunnerResultPredict  # type: ignore
from iplane.model_runner import RunnerResult  # type: ignore
from iplane.autoencoder import Autoencoder  # type: ignore
import iplane.constants as cn  # type: ignore
from tests.utils_test import makeAutocoderData  # type: ignore

import numpy as np  # type: ignore
import torch
from torch.utils.data import DataLoader
import pandas as pd    # type: ignore
import unittest

IGNORE_TESTS = True
IS_PLOT = True
NUM_EPOCH = 20000
NUM_EPOCH = 1000

TARGET_COLUMN = "target"  # Assuming the target column is named 'target'
NUM_DEPENDENT_FEATURE = 6
NUM_OUTPUT_FEATURE = 2
NUM_INPUT_FEATURE = NUM_DEPENDENT_FEATURE + NUM_OUTPUT_FEATURE
NUM_SAMPLE = 1000
MODEL = Autoencoder(layer_dimensions=[NUM_INPUT_FEATURE, 10*NUM_INPUT_FEATURE,
        10*NUM_OUTPUT_FEATURE, NUM_OUTPUT_FEATURE])  # Example model
TRAIN_DL = makeAutocoderData(num_sample=NUM_SAMPLE, num_independent_feature=NUM_OUTPUT_FEATURE,
        num_dependent_feature=NUM_DEPENDENT_FEATURE, num_value=10)
TEST_DL = makeAutocoderData(num_sample=1000, num_independent_feature=NUM_OUTPUT_FEATURE,
                            num_dependent_feature=NUM_DEPENDENT_FEATURE, num_value=10)


class TestModelRunner(unittest.TestCase):

    def setUp(self):
        self.runner = ModelRunnerNN(model=MODEL, num_epoch=NUM_EPOCH,
                learning_rate=1e-5, is_normalized=True,
                max_fractional_error=0.1)

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
        #if IGNORE_TESTS:
        #    return
        runner_result_fit = self.runner.fit(TRAIN_DL)
        self.runner.plotEvaluate(TEST_DL, is_plot=IS_PLOT)

if __name__ == '__main__':
    unittest.main()