from autoencodersb.dataset_csv import DatasetCSV # type: ignore
from autoencodersb.model_runner_nn import ModelRunnerNN, RunnerResultPredict  # type: ignore
from autoencodersb.model_runner import RunnerResult  # type: ignore
from autoencodersb.autoencoder import Autoencoder  # type: ignore
import autoencodersb.constants as cn  # type: ignore
from tests.utils_test import makeAutocoderData  # type: ignore

import numpy as np  # type: ignore
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd    # type: ignore
import unittest

IGNORE_TESTS = False
IS_PLOT = False
NUM_EPOCH = 2000
NUM_EPOCH = 1000
TARGET_COLUMN = "target"  # Assuming the target column is named 'target'
NUM_DEPENDENT_FEATURE = 6
NUM_OUTPUT_FEATURE = 2
NUM_INPUT_FEATURE = NUM_DEPENDENT_FEATURE + NUM_OUTPUT_FEATURE
NUM_SAMPLE = 1000

def makeModel():
    """Creates a model for testing."""
    return Autoencoder(layer_dimensions=[NUM_INPUT_FEATURE, 10*NUM_INPUT_FEATURE,
        10*NUM_OUTPUT_FEATURE, NUM_OUTPUT_FEATURE])  # Example model

MODEL = makeModel()
TRAIN_DL, dct = makeAutocoderData(num_sample=NUM_SAMPLE, num_independent_feature=NUM_OUTPUT_FEATURE,
        num_dependent_feature=NUM_DEPENDENT_FEATURE, num_value=10)
TEST_DL = makeAutocoderData(num_sample=1000, num_independent_feature=NUM_OUTPUT_FEATURE,
                            num_dependent_feature=NUM_DEPENDENT_FEATURE, num_value=10,
                            multiplier_dct=dct)[0]
TEST_DIR = os.path.abspath(os.path.dirname(__file__))
SERIALIZE_PATH = os.path.join(TEST_DIR, 'model_runner_nn_test.pkl')


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
        if IGNORE_TESTS:
            return
        runner_result_fit = self.runner.fit(TRAIN_DL)
        self.runner.plotEvaluate(TEST_DL, is_plot=IS_PLOT)

    def testSerializeDeserialize(self):
        if IGNORE_TESTS:
            return
        runner = ModelRunnerNN(model=MODEL, num_epoch=NUM_EPOCH,
                learning_rate=1e-5, is_normalized=True,
                max_fractional_error=0.1)
        runner.serialize(SERIALIZE_PATH)
        model = makeModel()
        new_runner = ModelRunnerNN.deserialize(model, SERIALIZE_PATH)
        self.assertTrue(new_runner.isSameModel(runner.model))
    
    def testIsSameModel(self):
        if IGNORE_TESTS:
            return
        self.assertTrue(self.runner.isSameModel(self.runner.model))
        self.assertFalse(self.runner.isSameModel(makeModel()))

if __name__ == '__main__':
    unittest.main()